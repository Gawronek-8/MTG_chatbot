from typing import Type, TypeVar, Any, Iterable, Iterator
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, PointStruct, PointIdsList

from src.utils.qdrant.base import QdrantBaseModel
from src.utils.qdrant.fields import QdrantBaseField, QdrantIntField, QdrantStringField


class QdrantSession[T: QdrantBaseModel]:
    """
    A wrapper around QdrantClient providing ORM-like access to collections based on QdrantBaseModel subclasses.
    """
    def __init__(self, client: QdrantClient):
        self._client = client

    def _parse_int_field(self, key: str, value: int) -> FieldCondition:
        if isinstance(value, tuple) and len(value) == 2:
            return FieldCondition(key=key, range=Range(gte=value[0], lte=value[1]))
        return FieldCondition(key=key, match=MatchValue(value=value))

    def _parse_string_field(self, key: str, value: str) -> FieldCondition:
        if isinstance(value, (tuple, list)):
            return FieldCondition(key=key, match=MatchAny(any=list(value)))
        return FieldCondition(key=key, match=MatchValue(value=value))

    @staticmethod
    def _collection_name(model_class: Type[T]) -> str:
        if not issubclass(model_class, QdrantBaseModel):
            raise TypeError(f"model_class must be a subclass of QdrantBaseModel, got {model_class}")

        collection_name = getattr(model_class, "__collection__", None)
        if not collection_name:
            raise ValueError(f"Model class {model_class.__name__} has no __collection__ defined.")
        return collection_name

    def collection_exists(self, model_class: Type[T]) -> bool:
        """Returns whether the model's collection exists in Qdrant."""
        return self._client.collection_exists(self._collection_name(model_class))

    def _build_query(self, model_class: Type[T], kwargs: dict[str, Any]) -> tuple[Filter | None, dict[str, Any]]:
        """Splits the get() kwargs into a payload filter and the vectors to search with."""
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
        return query_filter, vectors_to_search

    @staticmethod
    def _selectors(model_class: Type[T], fields: list[str] | None) -> tuple[bool | list[str], bool | list[str]]:
        """
        Builds the with_payload / with_vectors arguments for the requested fields.

        None returns everything. The id is always returned by Qdrant, so "id" alone fetches
        neither payload nor vectors. Fields that aren't fetched are left as None.
        """
        if fields is None:
            return True, True

        payload_fields = model_class.get_payload_fields()
        vector_fields = model_class.get_vector_fields()

        unknown = [f for f in fields if not isinstance(getattr(model_class, f, None), QdrantBaseField)]
        if unknown:
            raise ValueError(f"Fields {unknown} are not defined in model {model_class.__name__}")

        with_payload = [f for f in fields if f in payload_fields] or False
        with_vectors = [f for f in fields if f in vector_fields] or False
        return with_payload, with_vectors

    def get(self, model_class: Type[T], limit: int = 10, fields: list[str] | None = None, **kwargs) -> list[T]:
        collection_name = self._collection_name(model_class)
        query_filter, vectors_to_search = self._build_query(model_class, kwargs)
        with_payload, with_vectors = self._selectors(model_class, fields)

        if not vectors_to_search:
            records, _ = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            return [model_class.from_record(record) for record in records]
        elif len(vectors_to_search) == 1:
            vector_name, vector_value = next(iter(vectors_to_search.items()))
            response = self._client.query_points(
                collection_name=collection_name,
                query=vector_value,
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            return [model_class.from_scored_point(point) for point in response.points]
        else:
            raise NotImplementedError("Searching with multiple vectors simultaneously is not supported yet.")

    def get_batches(
        self,
        model_class: Type[T],
        batch_size: int = 500,
        fields: list[str] | None = None,
        **kwargs,
    ) -> Iterator[list[T]]:
        """
        Pages through the whole collection (optionally filtered by payload kwargs), yielding one list per page.
        Vector search is not supported here, since ranked results aren't meant to be paged through exhaustively.
        """
        collection_name = self._collection_name(model_class)
        query_filter, vectors_to_search = self._build_query(model_class, kwargs)
        if vectors_to_search:
            raise ValueError("get_batches() does not support vector search, use get() instead.")
        with_payload, with_vectors = self._selectors(model_class, fields)

        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            if records:
                yield [model_class.from_record(record) for record in records]
            if offset is None:
                break

    def refresh_collection(self, model_class: Type[T], ids_to_delete: Iterable[str], *, batch_size: int = 500) -> None:
        """Deletes the points with the given ids from the model's collection."""
        collection_name = self._collection_name(model_class)
        ids = list(ids_to_delete)

        for i in range(0, len(ids), batch_size):
            self._client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids[i:i + batch_size])
            )


    def put(self, model_instances: T | Iterable[T], *, batch_size = 500, create_missing: bool = False) -> None:

        def _batch_insert(collection: str, points: list[PointStruct]):
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=collection,
                    points=batch
                )


        if not isinstance(model_instances, Iterable):
            model_instances = [model_instances]

        if not all(list(issubclass(instance.__class__, QdrantBaseModel) for instance in model_instances)):
            raise TypeError("All instances must be of a subclass of QdrantBaseModel.")

        if any(map(lambda x: not getattr(x, "__collection__", None) , model_instances)):
            raise ValueError("All model instances must have a __collection__ attribute defined.")

        COLLECTIONS = defaultdict(list)

        for instance in model_instances:
            model_class = instance.__class__
            collection_name = model_class.__collection__
            COLLECTIONS[collection_name].append(instance)

        for collection_name, instances in COLLECTIONS.items():
            if self._client.collection_exists(collection_name):
                continue

            if not create_missing:
                raise RuntimeError(
                    f"Collection '{collection_name}' does not exist in Qdrant. "
                    "Create it first (QdrantBaseModel.create_all) or pass create_missing=True."
                )

            instances[0].__class__.create(self._client)

        for collection_name, instances in COLLECTIONS.items():
            points: list[PointStruct] = [instance.to_point() for instance in instances]
            _batch_insert(collection_name, points)
