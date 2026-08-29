from typing import Type, TypeVar, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from src.utils.qdrant.base import QdrantBaseModel
from src.utils.qdrant.fields import QdrantIntField, QdrantStringField


class QdrantSession[T: QdrantBaseModel]:
    """
    A wrapper around QdrantClient providing ORM-like access to collections based on QdrantBaseModel subclasses.
    """
    def __init__(self, client: QdrantClient):
        self.client = client

    def _parse_int_field(self, key: str, value: int) -> FieldCondition:
        if isinstance(value, tuple) and len(value) == 2:
            return FieldCondition(key=key, range=Range(gte=value[0], lte=value[1]))
        return FieldCondition(key=key, match=MatchValue(value=value))

    def _parse_string_field(self, key: str, value: str) -> FieldCondition:
        if isinstance(value, (tuple, list)):
            return FieldCondition(key=key, match=MatchAny(any=list(value)))
        return FieldCondition(key=key, match=MatchValue(value=value))

    def get(self, model_class: Type[T], limit: int = 10, **kwargs) -> list[T]:
        if not issubclass(model_class, QdrantBaseModel):
            raise TypeError(f"model_class must be a subclass of QdrantBaseModel, got {model_class}")

        collection_name = getattr(model_class, "__collection__", None)
        if not collection_name:
            raise ValueError(f"Model class {model_class.__name__} has no __collection__ defined.")

        payload_fields = model_class.get_payload_fields()
        vector_fields = model_class.get_vector_fields()

        must_conditions = []
        vectors_to_search = {}

        for key, value in kwargs.items():
            if key in payload_fields:
                field_def = payload_fields[key]
                if isinstance(field_def, QdrantIntField):
                    must_conditions.append(self._parse_int_field(key, value))
                elif isinstance(field_def, QdrantStringField):
                    must_conditions.append(self._parse_string_field(key, value))
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            elif key in vector_fields:
                vectors_to_search[key] = value
            else:
                raise ValueError(f"Field '{key}' is not defined in model {model_class.__name__}")

        query_filter = Filter(must=must_conditions) if must_conditions else None

        if not vectors_to_search:
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            return [model_class.from_record(record) for record in records]
        elif len(vectors_to_search) == 1:
            vector_name, vector_value = next(iter(vectors_to_search.items()))
            response = self.client.query_points(
                collection_name=collection_name,
                query=vector_value,
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            return [model_class.from_scored_point(point) for point in response.points]
        else:
            raise NotImplementedError("Searching with multiple vectors simultaneously is not supported yet.")
