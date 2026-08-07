from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from aif_qwen_agent.schemas import (
    ReadFilePolicy,
    ReadFileRequest,
    ReadFileTrace,
    ToolErrorCode,
)
from aif_qwen_agent.tools import ReadFileTool, ReadFileTraceStore


def make_tool(root: Path, trace_path: Path, max_read_bytes: int = 131_072) -> ReadFileTool:
    return ReadFileTool(
        ReadFilePolicy(allowed_roots=[root], max_read_bytes=max_read_bytes),
        ReadFileTraceStore(trace_path),
        cwd=root,
    )


def rejection_code(trace_path: Path, trace_id: str) -> ToolErrorCode:
    trace = ReadFileTraceStore(trace_path).get(trace_id)
    assert trace.rejection is not None
    return trace.rejection.code


def test_valid_read_is_bounded_hashed_verified_and_replayable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("verified evidence", encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"
    tool = make_tool(root, trace_path)

    trace = tool.run(ReadFileRequest(path="note.txt", max_bytes=64))

    assert trace.status == "completed"
    assert trace.authorized and trace.executed and trace.verified
    assert trace.observation is not None
    assert trace.observation.content == "verified evidence"
    assert trace.observation.byte_count == 17
    assert trace.observation.sha256 == sha256(b"verified evidence").hexdigest()
    assert ReadFileTraceStore(trace_path).get(str(trace.trace_id)) == trace


def test_parent_traversal_is_rejected_before_execution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"

    trace = make_tool(root, trace_path).run(ReadFileRequest(path="../outside.txt"))

    assert trace.status == "rejected"
    assert not trace.authorized and not trace.executed
    assert rejection_code(trace_path, str(trace.trace_id)) == ToolErrorCode.OUTSIDE_ALLOWED_ROOT


def test_symlink_escape_is_rejected_before_execution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside)
    trace_path = tmp_path / "traces.jsonl"

    trace = make_tool(root, trace_path).run(ReadFileRequest(path="escape.txt"))

    assert trace.status == "rejected"
    assert not trace.authorized and not trace.executed
    assert rejection_code(trace_path, str(trace.trace_id)) == ToolErrorCode.SYMLINK_ESCAPE


def test_oversize_file_is_rejected_during_execution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "large.txt").write_text("123456", encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"

    trace = make_tool(root, trace_path).run(ReadFileRequest(path="large.txt", max_bytes=5))

    assert trace.authorized and not trace.executed
    assert rejection_code(trace_path, str(trace.trace_id)) == ToolErrorCode.FILE_TOO_LARGE


def test_missing_path_and_directory_have_typed_rejections(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    trace_path = tmp_path / "traces.jsonl"
    tool = make_tool(root, trace_path)

    missing = tool.run(ReadFileRequest(path="missing.txt"))
    directory = tool.run(ReadFileRequest(path="."))

    assert rejection_code(trace_path, str(missing.trace_id)) == ToolErrorCode.NOT_FOUND
    assert rejection_code(trace_path, str(directory.trace_id)) == ToolErrorCode.NOT_FILE


def test_invalid_utf8_and_excessive_request_limit_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(b"\xff\xfe")
    trace_path = tmp_path / "traces.jsonl"
    tool = make_tool(root, trace_path, max_read_bytes=8)

    invalid = tool.run(ReadFileRequest(path="binary.dat", max_bytes=8))
    excessive = tool.run(ReadFileRequest(path="binary.dat", max_bytes=9))

    assert rejection_code(trace_path, str(invalid.trace_id)) == ToolErrorCode.INVALID_ENCODING
    assert (
        rejection_code(trace_path, str(excessive.trace_id))
        == ToolErrorCode.REQUEST_LIMIT_EXCEEDS_POLICY
    )


def test_verification_detects_modified_observation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.txt").write_text("original", encoding="utf-8")
    tool = make_tool(root, tmp_path / "traces.jsonl")
    trace = tool.run(ReadFileRequest(path="note.txt"))
    assert trace.observation is not None

    modified = trace.observation.model_copy(update={"content": "changed"})

    assert not tool.verify(trace.request, modified)

    tampered = trace.model_dump()
    tampered["observation"]["content"] = "changed"
    with pytest.raises(ValidationError, match="byte count does not match"):
        ReadFileTrace.model_validate(tampered)
