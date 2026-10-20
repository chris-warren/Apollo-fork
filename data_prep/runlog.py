"""Keep <root>/README.md = data_prep/README.md + a log of every generation run.

The README travels with the data (e.g. to DRAC), so whoever finds the folder can see
how it was made: the command line, git commit, seed and counts of each run.
"""
import datetime
import os
import subprocess
import sys

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
MARKER = "<!-- generation runs below are appended automatically -->"


def git_describe() -> str:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return commit + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_run(root: str, step: str, details: dict) -> None:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "README.md")
    runs = ""
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
        if MARKER in old:
            runs = old.split(MARKER, 1)[1]
    with open(TEMPLATE) as f:
        template = f.read().rstrip() + "\n"
    lines = [
        f"\n### {step} — {datetime.datetime.now().isoformat(timespec='seconds')}\n",
        f"- command: `{' '.join(sys.argv)}`",
        f"- git commit: `{git_describe()}`",
    ]
    lines += [f"- {k}: {v}" for k, v in details.items()]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(template + "\n## Generation runs\n\n" + MARKER + runs.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    os.replace(tmp, path)
