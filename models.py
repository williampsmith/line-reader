"""Core data models for the audition rehearsal app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class LineType(str, Enum):
    SCENE_HEADING = "scene_heading"
    ACTION = "action"
    CHARACTER = "character"
    PARENTHETICAL = "parenthetical"
    DIALOGUE = "dialogue"
    TRANSITION = "transition"


@dataclass
class ParsedLine:
    type: LineType
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    character: str | None = None
    modifier: str | None = None
    confidence: float = 1.0


@dataclass
class Scene:
    index: int
    number: str | None
    heading: str
    start_line: int
    end_line: int
    characters: set[str] = field(default_factory=set)


@dataclass
class Script:
    lines: list[ParsedLine]
    scenes: list[Scene]
    characters: set[str]


@dataclass
class VoiceAssignment:
    user_character: str
    voice_for_character: dict[str, str]


@dataclass
class PracticeQueueItem:
    role: Literal["user", "ai"]
    character: str
    text: str
    voice_id: str | None
    source_line_index: int
