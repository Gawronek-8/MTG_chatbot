import uuid

def _validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError as e:
        raise ValueError(f"'{value}' is not a valid UUID.") from e