from typing import Optional, List
from qdrant_client.models import Distance
from src.utils.qdrant import QdrantBaseModel, QdrantIntField, QdrantStringField, QdrantVectorField

class Rule(QdrantBaseModel):
    """
    Rule model representing a single rule to be stored in Qdrant.
    """
    __collection__ = "rules"
    
    rule_number = QdrantIntField(index=True)
    rule_slug = QdrantStringField(index=True)
    vector = QdrantVectorField(size=1536, distance=Distance.COSINE)
    
    def __init__(self, rule_number: int, rule_slug: str, vector: Optional[List[float]] = None):
        self.rule_number = rule_number
        self.rule_slug = rule_slug
        
        if vector is None:
            self.vector = [0.0] * self.get_vector_fields()["vector"].size
        else:
            self.vector = vector

    def __repr__(self):
        return f"<Rule(rule_number={self.rule_number}, rule_slug='{self.rule_slug}')>"
