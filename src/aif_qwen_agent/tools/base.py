from typing import Any, Protocol

from pydantic import BaseModel


class Tool(Protocol):
    name: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]

    def authorize(self, request: BaseModel, context: dict[str, Any]) -> bool: ...

    def execute(self, request: BaseModel) -> BaseModel: ...

    def verify(self, request: BaseModel, result: BaseModel) -> bool: ...
