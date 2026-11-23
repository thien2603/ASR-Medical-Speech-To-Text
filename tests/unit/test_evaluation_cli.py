"""Protect the boundary between training and evaluation commands."""
import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

from evaluation.datasets.evaluate import parse_args


class EvaluationCommandTests(unittest.TestCase):
    def test_finetuned_checkpoint_is_an_explicit_evaluation_source(self):
        with patch("sys.argv", ["test_datasets.py", "--checkpoint", "best.nemo", "--datasets", "vietmed"]):
            args = parse_args()
        self.assertEqual(str(args.checkpoint), "best.nemo")
        self.assertEqual(args.datasets, ["vietmed"])

    def test_conflicting_model_sources_are_rejected(self):
        with patch("sys.argv", ["test_datasets.py", "--checkpoint", "best.nemo", "--model-id", "another/model"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_importing_training_or_requesting_help_cannot_start_training(self):
        with patch("subprocess.run") as external_command:
            module = importlib.import_module("training.vietmed.train")
            module = importlib.reload(module)
            with patch("sys.argv", ["train.py"]), patch.object(module, "run_training") as training, contextlib.redirect_stdout(io.StringIO()):
                module.main()
            training.assert_not_called()
            external_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
