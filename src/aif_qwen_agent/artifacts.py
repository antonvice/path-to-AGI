"""Immutable artifact hashing, storage, and provenance."""

from hashlib import sha256
from pathlib import Path

from aif_qwen_agent.schemas import RunTrace


def sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: RunTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(trace.model_dump_json())
            stream.write("\n")

    def get(self, run_id: str) -> RunTrace:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            trace = RunTrace.model_validate_json(line)
            if str(trace.run_id) == run_id:
                return trace
        raise KeyError(run_id)
