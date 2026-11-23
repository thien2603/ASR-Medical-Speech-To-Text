"""Dependency-light local web UI for Vietnamese medical speech-to-text."""

from __future__ import annotations

import cgi
import json
import os
import tempfile
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import soundfile as sf

MODEL_ID = "kyle/vi-asr-fastconformer-114m"
MODEL_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vietnamese Medical Speech-to-Text</title><style>
:root{font-family:Inter,system-ui,sans-serif}body{margin:0;background:#f3f6f8;color:#17212b}main{max-width:760px;margin:48px auto;padding:0 20px}.card{background:white;border-radius:18px;padding:28px;box-shadow:0 12px 35px #10203014}h1{margin:0 0 8px;font-size:28px}p{color:#52616d;line-height:1.55}input[type=file]{width:100%;box-sizing:border-box;padding:18px;border:2px dashed #b6c4ce;border-radius:12px;background:#fafcfd}audio{width:100%;margin:18px 0 8px}button{border:0;border-radius:10px;padding:12px 20px;background:#087f5b;color:white;font-size:16px;font-weight:650;cursor:pointer;margin-right:8px}button.secondary{background:#4263eb}button.stop{background:#c92a2a}button:disabled{opacity:.55;cursor:wait}label{display:block;margin:22px 0 8px;font-weight:650}textarea{width:100%;min-height:170px;resize:vertical;box-sizing:border-box;border:1px solid #cbd5dc;border-radius:10px;padding:14px;font:16px/1.55 system-ui}#status{min-height:24px;margin-top:14px;color:#52616d}.error{color:#c92a2a!important}
</style></head><body><main><div class="card">
<h1>Vietnamese Medical Speech-to-Text</h1>
<p>Tải audio tiếng Việt để thử <code>vi-asr-fastconformer-114m</code>. Khuyến nghị WAV/FLAC/MP3, một người nói và đoạn ngắn.</p>
<input id="audioFile" type="file" accept="audio/*"><p><button id="record" class="secondary">🎙 Bắt đầu thu âm</button><button id="stop" class="stop" disabled>■ Dừng thu âm</button></p><audio id="player" controls hidden></audio>
<p><button id="run">Nhận dạng</button></p><div id="status">Bạn có thể chọn file hoặc thu âm trực tiếp. Model sẽ được nạp ở lần chạy đầu tiên.</div>
<label for="text">Transcript</label><textarea id="text" readonly placeholder="Kết quả sẽ xuất hiện ở đây…"></textarea>
</div></main><script>
const file=document.querySelector('#audioFile'),player=document.querySelector('#player'),button=document.querySelector('#run'),recordButton=document.querySelector('#record'),stopButton=document.querySelector('#stop'),statusBox=document.querySelector('#status'),textBox=document.querySelector('#text');
let chosenAudio=null,audioContext=null,stream=null,source=null,processor=null,chunks=[];
file.onchange=()=>{if(file.files[0]){chosenAudio=file.files[0];player.src=URL.createObjectURL(chosenAudio);player.hidden=false;statusBox.textContent='Đã chọn file audio.';}};
function wavBlob(buffers,sampleRate){let length=buffers.reduce((n,b)=>n+b.length,0),samples=new Float32Array(length),offset=0;for(const b of buffers){samples.set(b,offset);offset+=b.length;}const data=new ArrayBuffer(44+samples.length*2),v=new DataView(data);const s=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i));};s(0,'RIFF');v.setUint32(4,36+samples.length*2,true);s(8,'WAVE');s(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);s(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,x<0?x*32768:x*32767,true);}return new Blob([data],{type:'audio/wav'});}
recordButton.onclick=async()=>{try{stream=await navigator.mediaDevices.getUserMedia({audio:true});audioContext=new AudioContext();source=audioContext.createMediaStreamSource(stream);processor=audioContext.createScriptProcessor(4096,1,1);chunks=[];processor.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));source.connect(processor);processor.connect(audioContext.destination);recordButton.disabled=true;stopButton.disabled=false;statusBox.className='';statusBox.textContent='Đang thu âm…';}catch(error){statusBox.className='error';statusBox.textContent='Không truy cập được microphone: '+error.message;}};
stopButton.onclick=async()=>{processor.disconnect();source.disconnect();stream.getTracks().forEach(t=>t.stop());const rate=audioContext.sampleRate;await audioContext.close();chosenAudio=new File([wavBlob(chunks,rate)],'microphone.wav',{type:'audio/wav'});player.src=URL.createObjectURL(chosenAudio);player.hidden=false;recordButton.disabled=false;stopButton.disabled=true;statusBox.textContent='Đã thu âm xong. Hãy nghe lại hoặc bấm Nhận dạng.';};
button.onclick=async()=>{if(!chosenAudio){statusBox.textContent='Hãy chọn file hoặc thu âm trước.';statusBox.className='error';return;}button.disabled=true;textBox.value='';statusBox.className='';statusBox.textContent='Đang nhận dạng… lần đầu có thể mất khoảng một phút để nạp model.';const body=new FormData();body.append('audio',chosenAudio);try{const response=await fetch('/transcribe',{method:'POST',body});const result=await response.json();if(!response.ok)throw new Error(result.error||'Lỗi không xác định');textBox.value=result.transcript;statusBox.textContent=result.stats;}catch(error){statusBox.textContent=error.message;statusBox.className='error';}finally{button.disabled=false;}};
</script></body></html>"""


def hypothesis_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, (tuple, list)) and value:
        return hypothesis_text(value[0])
    return str(value)


@lru_cache(maxsize=1)
def load_model():
    import torch
    import nemo.collections.asr as nemo_asr

    requested = os.getenv("ASR_DEVICE", "cpu").lower()
    device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID, map_location=device)
    model.eval()
    return model, device


def transcribe_file(path: Path) -> dict[str, Any]:
    try:
        audio_seconds = float(sf.info(path).duration)
    except Exception:
        audio_seconds = 0.0
    started = time.perf_counter()
    with MODEL_LOCK:
        model, device = load_model()
        output = model.transcribe([str(path)], batch_size=1, verbose=False)
    elapsed = time.perf_counter() - started
    if isinstance(output, tuple):
        output = output[0]
    transcript = hypothesis_text(output[0]).strip() if output else ""
    rtf = elapsed / audio_seconds if audio_seconds else None
    stats = f"Thiết bị: {device.upper()} · Audio: {audio_seconds:.1f}s · Xử lý: {elapsed:.2f}s"
    if rtf is not None:
        stats += f" · RTF: {rtf:.3f}"
    return {"transcript": transcript, "stats": stats}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/health":
            self.send_json(200, {"status": "ok", "model_loaded": load_model.cache_info().currsize == 1})
        else:
            self.send_json(404, {"error": "Không tìm thấy trang."})

    def do_POST(self) -> None:
        if self.path != "/transcribe":
            self.send_json(404, {"error": "Không tìm thấy API."})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(400, {"error": "Tệp trống hoặc lớn hơn 100 MB."})
            return
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                "REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
            item = form["audio"]
            suffix = Path(item.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temp_path = Path(handle.name)
                while chunk := item.file.read(1024 * 1024):
                    handle.write(chunk)
            try:
                self.send_json(200, transcribe_file(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as exc:
            self.send_json(500, {"error": f"Không thể nhận dạng audio: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


if __name__ == "__main__":
    host, port = "127.0.0.1", int(os.getenv("ASR_PORT", "7860"))
    print(f"Giao diện đang chạy tại http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
