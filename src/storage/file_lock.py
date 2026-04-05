"""Cross-process file locking for mutable state files.

Uses ``filelock`` to ensure atomic read-modify-write cycles on JSON
state files (sampling counters, question stores, etc.) that may be
accessed concurrently by multiple pipeline invocations.
"""

from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

_DEFAULT_TIMEOUT = 10  # seconds


@contextmanager
def locked_file(path: Path, *, timeout: float = _DEFAULT_TIMEOUT):
    """Acquire an exclusive lock for *path* before yielding.

    The lock file is placed next to the target file with a ``.lock``
    suffix so it never collides with the actual data.

    Usage::

        with locked_file(counter_path):
            data = json.loads(counter_path.read_text())
            data["count"] += 1
            counter_path.write_text(json.dumps(data))
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield
