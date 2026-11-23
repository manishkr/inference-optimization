"""Build the CUDA extension into .build/lib, skipping the build when sources are unchanged."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / ".build"


def wheel_cuda_home():
    """CUDA toolkit installed from pip wheels into the environment, if present."""
    try:
        import nvidia
    except ImportError:
        return None
    for root in nvidia.__path__:
        home = Path(root) / "cu13"
        if (home / "bin" / "nvcc").exists():
            return home
    return None


def fingerprint():
    digest = hashlib.sha256()
    digest.update(os.environ.get("NESTED_KV_CUDA_ARCHS", "").encode() + b"\0")
    for path in [ROOT / "setup.py", *sorted((ROOT / "csrc").rglob("*"))]:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build(force=False):
    TARGET.mkdir(parents=True, exist_ok=True)
    stamp = TARGET / "source.sha256"
    current = fingerprint()
    libraries = list((TARGET / "lib").glob("_nested_kv*.so"))
    if force or len(libraries) != 1 or not stamp.exists() or stamp.read_text().strip() != current:
        for old in libraries:
            old.unlink()
        env = {**os.environ, "MAX_JOBS": os.environ.get("MAX_JOBS", "4")}
        subprocess.run(
            [sys.executable, "setup.py", "build_ext",
             "--build-temp", str(TARGET / "temp"), "--build-lib", str(TARGET / "lib")],
            cwd=ROOT, env=env, check=True)
        stamp.write_text(current + "\n")
    return TARGET / "lib"


if __name__ == "__main__":
    print(build(force="--force" in sys.argv[1:]))
