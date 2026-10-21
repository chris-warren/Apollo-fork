"""Load machine-specific settings (data paths, wandb mode) from a .env file.

Configs refer to them as ${oc.env:NAME}. The file holds one KEY=VALUE per line;
blank lines and # comments are ignored, an `export ` prefix is allowed, values may
be quoted, and ~ and $VARS are expanded. Variables already set in the environment
win, so a job script can override the file. Copy .env.example to .env to start.
"""
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))
_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


class EnvFileError(ValueError):
    pass


def parse_env_file(path):
    """Return {key: value} from a .env file; raises EnvFileError on a malformed line."""
    values = {}
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _LINE.match(line)
            if not m:
                raise EnvFileError(f"{path}:{n}: expected KEY=VALUE, got {raw.rstrip()!r}")
            key, value = m.group(1), m.group(2).strip()
            if value[:1] in ("'", '"'):
                end = value.find(value[0], 1)
                if end < 0:
                    raise EnvFileError(f"{path}:{n}: unterminated quote in {raw.rstrip()!r}")
                value = value[1:end]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            values[key] = os.path.expanduser(os.path.expandvars(value))
    return values


def load_env(path=None, verbose=True):
    """Load `path` (default: .env in the repository root) into os.environ.

    Returns the variables that were set from the file. A missing file is not an
    error: everything may already be in the environment.
    """
    path = path or os.path.join(REPO, ".env")
    if not os.path.exists(path):
        if verbose:
            print(f"No {path}; using the environment as is (see .env.example)")
        return {}
    loaded = {}
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    if verbose:
        kept = sorted(set(parse_env_file(path)) - set(loaded))
        print(f"Loaded {sorted(loaded)} from {path}" + (f"; kept existing {kept}" if kept else ""))
    return loaded
