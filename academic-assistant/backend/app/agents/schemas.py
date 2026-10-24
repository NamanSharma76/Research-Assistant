from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    intent: Literal[
        "paper_question",
        "summarize",
        "notes",
        "outline",
        "rephrase_previous",
        "off_topic",
        "needs_clarification",
    ] = Field(description="Detected routing intent.")
    standalone_query: str = Field(description="De-contextualized standalone query resolved from history.")
    section: Optional[str] = Field(default=None, description="Section name if intent is summarize and a specific section was requested.")
    num_slides: Optional[int] = Field(default=None, description="Number of presentation slides if requested.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    reason: str = Field(description="Brief reason for this routing decision.")


class NoteField(BaseModel):
    text: str = Field(description="Content for this note field.")
    sources: List[str] = Field(default_factory=list, description="Citations and chunk sources, e.g. [Section, p. X].")


class Term(BaseModel):
    term: str = Field(description="Key technical concept or term name.")
    definition: str = Field(description="Concise definition grounded in the paper.")


class Notes(BaseModel):
    problem: NoteField = Field(description="Problem statement, motivation, and challenges addressed.")
    method: NoteField = Field(description="Proposed model, architecture, or methodological approach.")
    dataset_and_setup: NoteField = Field(description="Datasets, benchmarks, hardware, and experimental setup.")
    key_results: NoteField = Field(description="Key experimental results, metrics, and comparisons.")
    limitations: NoteField = Field(description="Limitations, trade-offs, and future work discussed.")
    key_terms: List[Term] = Field(default_factory=list, description="List of key concepts and technical terms.")
    revision_questions: List[str] = Field(default_factory=list, description="Questions to test conceptual understanding.")


class Slide(BaseModel):
    slide_number: int = Field(description="Sequential slide number starting from 1.")
    title: str = Field(description="Slide heading.")
    bullets: List[str] = Field(default_factory=list, description="Bullet points summarizing the slide contents.")
    source_fields: List[str] = Field(
        default_factory=list,
        description="List of notes field names this slide draws on: problem, method, dataset_and_setup, key_results, limitations."
    )
    source: str = Field(default="", description="Derived citation string [Section, p. X] built in code.")


class Outline(BaseModel):
    title: str = Field(description="Presentation title.")
    slides: List[Slide] = Field(default_factory=list, description="List of consecutively numbered slides.")
