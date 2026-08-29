from typing import Dict, Any, Optional, Self
from qdrant_client.models import Record, ScoredPoint, VectorStructOutput
from .fields import QdrantVectorField, QdrantPayloadField

class QdrantBaseModel:
    """
    Base class for Qdrant models, acting similar to SQLAlchemy's declarative base.
    Registers all subclasses that define a `__collection__` attribute.
    """
    
    __metadata: Dict[str, type['QdrantBaseModel']] = {}
    __collection__: Optional[str] = None

    def __init__(self, **kwargs):
        cls = self.__class__
        payload_fields = cls.get_payload_fields()
        vector_fields = cls.get_vector_fields()

        self.score: float = kwargs.pop("score", -1)

        for field_name in payload_fields:
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
            else:
                setattr(self, field_name, None)
                
        for vec_name, vec_field in vector_fields.items():
            if vec_name in kwargs:
                setattr(self, vec_name, kwargs.pop(vec_name))
            else:
                setattr(self, vec_name, [0.0] * vec_field.size)
                
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for {cls.__name__}: {list(kwargs.keys())}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        collection_name = getattr(cls, "__collection__", None) 
        if collection_name:
            cls.__metadata[collection_name] = cls

    @classmethod
    def get_vector_fields(cls) -> Dict[str, QdrantVectorField]:
        """Dynamically inspects the class to find all QdrantVectorField definitions."""
        vectors = {}
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, QdrantVectorField):
                vectors[attr_name] = attr_value
        return vectors

    @classmethod
    def _named_vectors(cls, vector: Optional[VectorStructOutput]) -> Dict[str, Any]:
        if vector is None:
            return {}
        if not isinstance(vector, dict):
            raise TypeError(
                f"{cls.__name__} only supports named vectors, got unnamed vector of type {type(vector).__name__}"
            )
        return vector

    @classmethod
    def from_record(cls, record: Record) -> Self:
        """Builds a model instance from a scroll() result (qdrant_client Record), keeping the actual stored vectors."""
        kwargs = dict(record.payload or {})
        kwargs.update(cls._named_vectors(record.vector))
        return cls(**kwargs)

    @classmethod
    def from_scored_point(cls, point: ScoredPoint) -> Self:
        """
        The raw vector value isn't meaningful here since Qdrant already computed the
        match quality, so each queried vector field is set to the point's score instead.
        """
        kwargs = dict(point.payload or {})
        kwargs.update({name: point.score for name in cls._named_vectors(point.vector)})
        kwargs["score"] = point.score
        return cls(**kwargs)

    @classmethod
    def get_payload_fields(cls) -> Dict[str, QdrantPayloadField]:
        """Dynamically inspects the class to find all QdrantPayloadField definitions."""
        payloads = {}
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, QdrantPayloadField):
                payloads[attr_name] = attr_value
        return payloads

    @classmethod
    def get_create_collection_kwargs(cls) -> Dict[str, Any]:
        """Builds the kwargs for QdrantClient.create_collection() from this model's vector fields."""
        return {
            "collection_name": cls.__collection__,
            "vectors_config": {
                vec_name: vec_field.to_qdrant_config()
                for vec_name, vec_field in cls.get_vector_fields().items()
            },
        }

    @classmethod
    def create_all(cls, client: Any = None):
        """
        Creates all registered collections in Qdrant, along with any payload indexes.

        :param client: A qdrant_client.QdrantClient instance.
        """
        print("Initializing Qdrant collections...")
        if not cls.__metadata:
            print("No collections registered.")
            return

        for collection_name, model_class in cls.__metadata.items():
            print(f" -> Creating collection: '{collection_name}' for model '{model_class.__name__}'")
            client.create_collection(**model_class.get_create_collection_kwargs())

            for payload_name, payload_field in model_class.get_payload_fields().items():
                if not payload_field.index:
                    continue
                print(f"    -> Indexing field: '{payload_name}' ({payload_field.schema_type})")
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=payload_name,
                    field_schema=payload_field.schema_type,
                )
