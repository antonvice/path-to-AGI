import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from aif_qwen_agent.schemas import (
    ReadFileObservation,
    ReadFilePolicy,
    ReadFileRequest,
    ReadFileTrace,
    ToolErrorCode,
    ToolPhase,
    ToolRejection,
)


class ReadFileRejected(Exception):
    def __init__(
        self,
        code: ToolErrorCode,
        phase: ToolPhase,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.message = message


class ReadFileTraceStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: ReadFileTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(trace.model_dump_json())
            stream.write("\n")

    def get(self, trace_id: str) -> ReadFileTrace:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            trace = ReadFileTrace.model_validate_json(line)
            if str(trace.trace_id) == trace_id:
                return trace
        raise KeyError(trace_id)


class ReadFileTool:
    name = "read_file"
    input_schema = ReadFileRequest
    output_schema = ReadFileObservation

    def __init__(
        self,
        policy: ReadFilePolicy,
        traces: ReadFileTraceStore,
        cwd: Path | None = None,
    ) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        self.policy = policy
        self.traces = traces
        self._lexical_roots = tuple(
            Path(os.path.abspath(root if root.is_absolute() else self.cwd / root))
            for root in policy.allowed_roots
        )
        self._resolved_roots = tuple(root.resolve() for root in self._lexical_roots)

    def run(self, request: ReadFileRequest) -> ReadFileTrace:
        started_at = datetime.now(UTC)
        authorized = False
        executed = False
        verified = False
        observation = None
        rejection = None
        try:
            resolved_path = self._authorize(request)
            authorized = True
            observation = self._execute(request, resolved_path)
            executed = True
            if not self._verify(observation):
                raise ReadFileRejected(
                    ToolErrorCode.VERIFICATION_FAILED,
                    "verification",
                    "content hash or byte count verification failed",
                )
            verified = True
        except ReadFileRejected as rejected:
            observation = None
            rejection = ToolRejection(
                code=rejected.code,
                phase=rejected.phase,
                message=rejected.message,
            )
        trace = ReadFileTrace(
            trace_id=uuid4(),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            request=request,
            allowed_roots=list(self._resolved_roots),
            status="completed" if verified else "rejected",
            authorized=authorized,
            executed=executed,
            verified=verified,
            observation=observation,
            rejection=rejection,
        )
        self.traces.append(trace)
        return trace

    def authorize(self, request: ReadFileRequest, context: dict[str, object]) -> bool:
        try:
            self._authorize(request)
        except ReadFileRejected:
            return False
        return True

    def execute(self, request: ReadFileRequest, sandbox: object) -> ReadFileObservation:
        return self._execute(request, self._authorize(request))

    def verify(self, request: ReadFileRequest, result: ReadFileObservation) -> bool:
        return self._verify(result)

    def _authorize(self, request: ReadFileRequest) -> Path:
        if request.max_bytes > self.policy.max_read_bytes:
            raise ReadFileRejected(
                ToolErrorCode.REQUEST_LIMIT_EXCEEDS_POLICY,
                "authorization",
                f"requested {request.max_bytes} bytes exceeds policy limit",
            )
        requested = Path(request.path)
        candidate = requested if requested.is_absolute() else self.cwd / requested
        lexical = Path(os.path.abspath(candidate))
        if not any(lexical.is_relative_to(root) for root in self._lexical_roots):
            raise ReadFileRejected(
                ToolErrorCode.OUTSIDE_ALLOWED_ROOT,
                "authorization",
                "requested path is outside allowed roots",
            )
        try:
            resolved = lexical.resolve(strict=False)
        except OSError as error:
            raise ReadFileRejected(
                ToolErrorCode.IO_ERROR,
                "authorization",
                f"could not resolve requested path: {error}",
            ) from error
        if not any(resolved.is_relative_to(root) for root in self._resolved_roots):
            raise ReadFileRejected(
                ToolErrorCode.SYMLINK_ESCAPE,
                "authorization",
                "requested path resolves outside allowed roots",
            )
        return resolved

    def _execute(self, request: ReadFileRequest, resolved_path: Path) -> ReadFileObservation:
        try:
            if not resolved_path.exists():
                raise ReadFileRejected(
                    ToolErrorCode.NOT_FOUND,
                    "execution",
                    "requested file does not exist",
                )
            if not resolved_path.is_file():
                raise ReadFileRejected(
                    ToolErrorCode.NOT_FILE,
                    "execution",
                    "requested path is not a regular file",
                )
            with resolved_path.open("rb") as stream:
                payload = stream.read(request.max_bytes + 1)
        except ReadFileRejected:
            raise
        except PermissionError as error:
            raise ReadFileRejected(
                ToolErrorCode.PERMISSION_DENIED,
                "execution",
                "permission denied while reading requested file",
            ) from error
        except OSError as error:
            raise ReadFileRejected(
                ToolErrorCode.IO_ERROR,
                "execution",
                f"I/O error while reading requested file: {error}",
            ) from error
        if len(payload) > request.max_bytes:
            raise ReadFileRejected(
                ToolErrorCode.FILE_TOO_LARGE,
                "execution",
                f"requested file exceeds {request.max_bytes} byte limit",
            )
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReadFileRejected(
                ToolErrorCode.INVALID_ENCODING,
                "execution",
                "requested file is not valid UTF-8",
            ) from error
        return ReadFileObservation(
            resolved_path=resolved_path,
            content=content,
            byte_count=len(payload),
            sha256=sha256(payload).hexdigest(),
        )

    def _verify(self, result: ReadFileObservation) -> bool:
        payload = result.content.encode(result.encoding)
        return len(payload) == result.byte_count and sha256(payload).hexdigest() == result.sha256
