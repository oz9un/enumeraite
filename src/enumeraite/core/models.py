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

class PatternComponent(BaseModel):
    """A decomposed component of a subdomain pattern."""
    value: str = Field(..., description="The actual text component")
    type: str = Field(..., description="The type/category of this component")
    description: str = Field(..., description="Human-readable explanation of this component")
    alternatives: List[str] = Field(default_factory=list, description="Possible alternatives for this component")

class PatternAnalysisResult(BaseModel):
    """Result of AI-powered pattern analysis for a single subdomain."""
    original_subdomain: str = Field(..., description="The original subdomain being analyzed")
    decomposition: List[PatternComponent] = Field(..., description="Structured breakdown of the subdomain")
    reasoning: str = Field(..., description="AI's narrative explanation of the pattern recognition")
    pattern_template: str = Field(..., description="Abstract template showing the pattern structure")
    generated_variants: List[str] = Field(..., description="Generated subdomain variants based on the pattern")
    confidence_score: float = Field(..., description="Overall confidence in the pattern analysis (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")

class PathComponent(BaseModel):
    """A decomposed component of an API path."""
    value: str = Field(..., description="The actual path component")
    type: str = Field(..., description="The type/category of this component (prefix, version, resource, action)")
    description: str = Field(..., description="Human-readable explanation of this component")
    position: int = Field(..., description="Position in the path (0-indexed)")

class PathFunctionAnalysis(BaseModel):
    """Result of AI-powered path function analysis."""
    original_path: str = Field(..., description="The original API path being analyzed")
    function_context: str = Field(..., description="The requested functionality context")
    path_breakdown: List[PathComponent] = Field(..., description="Structured breakdown of the path")
    function_analysis: str = Field(..., description="AI's analysis of the requested function and how to implement it")
    reasoning: str = Field(..., description="AI's reasoning for the generated paths")
    generated_paths: List[str] = Field(..., description="Generated API paths for the requested functionality")
    confidence_score: float = Field(..., description="Overall confidence in the analysis (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")