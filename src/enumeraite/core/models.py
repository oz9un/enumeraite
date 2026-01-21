"""Data models for Enumeraite core functionality."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

class GenerationResult(BaseModel):
    """Result of AI path generation including paths and confidence scores."""
    paths: List[str] = Field(..., description="Generated API paths")
    confidence_scores: List[float] = Field(..., description="Confidence scores 0.0-1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider metadata")

    @model_validator(mode='after')
    def validate_lengths_match(self):
        """Ensure paths and confidence scores have same length."""
        if len(self.paths) != len(self.confidence_scores):
            raise ValueError("Paths and confidence scores must have same length")
        return self

class ValidationResult(BaseModel):
    """Result of HTTP path validation."""
    path: str = Field(..., description="The path that was tested")
    status_code: Optional[int] = Field(None, description="HTTP status code received")
    exists: bool = Field(False, description="Whether the path appears to exist")
    method: str = Field("GET", description="HTTP method used for testing")
    response_time: Optional[float] = Field(None, description="Response time in seconds")