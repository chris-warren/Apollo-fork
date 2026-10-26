import os
import tempfile
import unittest
from unittest import mock

from dotenv_loader import EnvFileError, configure_attention_backend, ddp_backend, load_env, parse_env_file

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
        self.assertEqual(set(values), {"APOLLO_DATA", "WANDB_MODE", "APOLLO_SDPA_BACKEND", "APOLLO_DDP_BACKEND"})

    @unittest.skipIf(OmegaConf is None, "omegaconf not installed")
    def test_smoke_config_resolves_from_env_file(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APOLLO_DATA", None)
            load_env(write(d, f"APOLLO_DATA={d}/apollo_v1\n"), verbose=False)
            cfg = OmegaConf.load(os.path.join(REPO, "configs", "apollo_smoke.yaml"))
            OmegaConf.resolve(cfg)
            self.assertEqual(cfg.datas.train_dir, f"{d}/apollo_v1/hdf5_train")
            self.assertEqual(cfg.datas.eval_dir, f"{d}/apollo_v1/eval/val.hdf5")


class AttentionBackendTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.saved = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                      torch.backends.cuda.math_sdp_enabled())

    def tearDown(self):
        b = self.torch.backends.cuda
        b.enable_flash_sdp(self.saved[0]); b.enable_mem_efficient_sdp(self.saved[1]); b.enable_math_sdp(self.saved[2])

    def test_math_disables_fused_kernels(self):
        with mock.patch.dict(os.environ, {"APOLLO_SDPA_BACKEND": "math"}):
            self.assertEqual(configure_attention_backend(verbose=False), "math")
        b = self.torch.backends.cuda
        self.assertEqual((b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled()),
                         (False, False, True))

    def test_default_leaves_torch_choice(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APOLLO_SDPA_BACKEND", None)
            self.assertEqual(configure_attention_backend(verbose=False), "default")
        self.assertEqual((self.torch.backends.cuda.flash_sdp_enabled(),
                          self.torch.backends.cuda.mem_efficient_sdp_enabled()), self.saved[:2])

    def test_rejects_unknown_value(self):
        with mock.patch.dict(os.environ, {"APOLLO_SDPA_BACKEND": "flash"}):
            with self.assertRaises(ValueError):
                configure_attention_backend(verbose=False)


class DdpBackendTests(unittest.TestCase):
    def test_values(self):
        for value, expected in [("gloo", "gloo"), ("nccl", "nccl"), ("default", None), (" GLOO ", "gloo")]:
            with mock.patch.dict(os.environ, {"APOLLO_DDP_BACKEND": value}):
                self.assertEqual(ddp_backend(), expected)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APOLLO_DDP_BACKEND", None)
            self.assertIsNone(ddp_backend())
        with mock.patch.dict(os.environ, {"APOLLO_DDP_BACKEND": "mpi"}):
            with self.assertRaises(ValueError):
                ddp_backend()


if __name__ == "__main__":
    unittest.main()
