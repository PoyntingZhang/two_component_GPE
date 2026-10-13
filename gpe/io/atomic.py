"""Publish complete files; an interrupted write leaves a .partial file."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import numpy as np

@contextmanager
def atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_json(path, value):
    with atomic_file(path) as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

def atomic_npz(path, **arrays):
    with atomic_file(path) as stream:
        np.savez(stream, **arrays)
