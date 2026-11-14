from typing import Optional, Generic, TypeVar, overload, Any, List
from dataclasses import dataclass, field
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from src.utils.validation import _validate_uuid

T = TypeVar('T')

@dataclass
class QdrantBaseField(Generic[T]):
    """Base class for Qdrant fields."""
    name: str = field(init=False)
    
    def __set_name__(self, owner, name):
        self.name = name

    @overload
    def __get__(self, instance: None, owner: Any) -> 'QdrantBaseField[T]': ...

    @overload
    def __get__(self, instance: object, owner: Any) -> T: ...

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance: Any, value: T) -> None:
        instance.__dict__[self.name] = value

@dataclass
class QdrantUUIDField(QdrantBaseField[str]):
    """The point ID (a UUID string). Not part of the payload, so it is never indexed or filtered on."""
    def __set__(self, instance, value):
        if value is not None:
            value = _validate_uuid(value)
        instance.__dict__[self.name] = value

@dataclass
class QdrantVectorField(QdrantBaseField[List[float]]):
    """Configuration for a Qdrant vector."""
    size: int
    distance: Distance
        
    def to_qdrant_config(self) -> VectorParams:
        """Returns the valid configuration argument for Qdrant's vectors_config."""
        return VectorParams(size=self.size, distance=self.distance)

@dataclass
class QdrantPayloadField(QdrantBaseField[T]):
    """Base class for Qdrant payload fields."""
    index: bool = False
    schema_type: Optional[PayloadSchemaType] = field(init=False, default=None)

@dataclass
class QdrantIntField(QdrantPayloadField[int]):
    """Integer payload field."""
    schema_type: PayloadSchemaType = field(init=False, default=PayloadSchemaType.INTEGER)

@dataclass
class QdrantStringField(QdrantPayloadField[str]):
    """String payload field."""
    schema_type: PayloadSchemaType = field(init=False, default=PayloadSchemaType.KEYWORD)
