"""Standalone equivalent of the adjacent notebook; execution requires --run."""
import argparse


def run_runtime_diagnostic():
    # ---
    # jupyter:
    #   jupytext:
    #     formats: ipynb,py:percent
    #   kernelspec:
    #     display_name: Python 3
    #     language: python
    #     name: python3
    # ---

    # %%
    import importlib.metadata as metadata
    import platform
    import subprocess
    import sys

    print("python", sys.version)
    print("platform", platform.platform())
    for package in [
        "numpy", "scipy", "torch", "torchaudio", "torchmetrics", "lightning",
        "pytorch-lightning", "numba", "librosa", "huggingface-hub", "pandas",
        "protobuf", "fsspec", "datasets"
    ]:
        try:
            print(package, metadata.version(package))
        except metadata.PackageNotFoundError:
            print(package, "NOT_INSTALLED")

    print("\nIMPORT CHECK")
    for module in ["numpy", "scipy", "torch", "torchmetrics", "lightning"]:
        try:
            imported = __import__(module)
            print(module, "OK", getattr(imported, "__version__", "n/a"))
        except Exception as exc:
            print(module, "ERROR", repr(exc))

    subprocess.run(["nvidia-smi"], check=False)

    # %%
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '--upgrade-strategy', 'only-if-needed', 'numpy==2.1.3', 'scipy==1.16.3', 'torchmetrics==1.9.0', 'librosa==0.11.0', 'soundfile>=0.12.1', 'pandas==2.3.3', 'protobuf==5.29.5', 'nemo_toolkit[asr]==2.7.3', 'jiwer==3.1.0'], check=True)

    # %%
    print("\nPOST-INSTALL IMPORT CHECK")
    import numpy as np
    import scipy
    import torchmetrics
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr

    print("numpy", np.__version__)
    print("scipy", scipy.__version__)
    print("torchmetrics", torchmetrics.__version__)
    print("lightning", pl.__version__)
    print("nemo_toolkit", metadata.version("nemo_toolkit"))
    print("NEMO_IMPORT_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Install the notebook dependencies and execute the pipeline')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    run_runtime_diagnostic()


if __name__ == '__main__':
    main()
