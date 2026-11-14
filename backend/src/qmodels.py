from qdrant_client.models import Distance
from src.utils.qdrant import QdrantBaseModel, QdrantStringField, QdrantVectorField

class Rule(QdrantBaseModel):
    """
    Rule model representing a single rule to be stored in Qdrant.
    """
    __collection__ = "rules"

    rule_number = QdrantStringField(index=True)
    rule_text = QdrantStringField(index=False)
    vector = QdrantVectorField(size=768, distance=Distance.COSINE)

    def __repr__(self):
        return f"<Rule(rule_number='{self.rule_number}')>"
