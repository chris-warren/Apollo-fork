import os
import tempfile
import unittest
from unittest import mock

from dotenv_loader import EnvFileError, load_env, parse_env_file

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(d, text):
    path = os.path.join(d, ".env")
    with open(path, "w") as f:
        f.write(text)
    return path


class ParseTests(unittest.TestCase):
    def test_syntax(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"HOME": "/home/u", "PI": "abc"}):
            path = write(d, "\n".join([
                "# comment",
                "",
                "PLAIN=value",
                "export EXPORTED=1",
                "SPACED = a b  # trailing comment",
                'DOUBLE="has # hash"',
                "SINGLE='quoted value'",
                "TILDE=~/data",
                "VAR=/p/def-$PI/x",
                "EMPTY=",
            ]))
            self.assertEqual(parse_env_file(path), {
                "PLAIN": "value", "EXPORTED": "1", "SPACED": "a b", "DOUBLE": "has # hash",
                "SINGLE": "quoted value", "TILDE": "/home/u/data", "VAR": "/p/def-abc/x", "EMPTY": "",
            })

    def test_malformed_lines_are_rejected_with_line_number(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(EnvFileError, r":2: expected KEY=VALUE"):
                parse_env_file(write(d, "A=1\nnot a setting\n"))
            with self.assertRaisesRegex(EnvFileError, "unterminated quote"):
                parse_env_file(write(d, 'A="open\n'))


class LoadTests(unittest.TestCase):
    def test_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"KEEP": "from-env"}, clear=False):
            os.environ.pop("NEW_VAR_FOR_TEST", None)
            loaded = load_env(write(d, "KEEP=from-file\nNEW_VAR_FOR_TEST=x\n"), verbose=False)
            self.assertEqual(loaded, {"NEW_VAR_FOR_TEST": "x"})
            self.assertEqual(os.environ["KEEP"], "from-env")
            self.assertEqual(os.environ["NEW_VAR_FOR_TEST"], "x")
            del os.environ["NEW_VAR_FOR_TEST"]

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(load_env("/nonexistent/.env", verbose=False), {})

    def test_example_file_parses(self):
        values = parse_env_file(os.path.join(REPO, ".env.example"))
        self.assertEqual(set(values), {"APOLLO_DATA", "WANDB_MODE"})

    @unittest.skipIf(OmegaConf is None, "omegaconf not installed")
    def test_smoke_config_resolves_from_env_file(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APOLLO_DATA", None)
            load_env(write(d, f"APOLLO_DATA={d}/apollo_v1\n"), verbose=False)
            cfg = OmegaConf.load(os.path.join(REPO, "configs", "apollo_smoke.yaml"))
            OmegaConf.resolve(cfg)
            self.assertEqual(cfg.datas.train_dir, f"{d}/apollo_v1/hdf5_train")
            self.assertEqual(cfg.datas.eval_dir, f"{d}/apollo_v1/eval/val.hdf5")


if __name__ == "__main__":
    unittest.main()
