"""OCR and heuristic screenplay parsing."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Iterable

from models import LineType, ParsedLine, Scene, Script


DPI = 200
MIN_OCR_CONFIDENCE = 0.5
HEADER_FOOTER_MARGIN_RATIO = 0.05
DEFAULT_RASTERIZE_THREADS = max(2, min((os.cpu_count() or 2), 6))

LEFT_MARGIN_MAX_IN = 1.35
CHARACTER_X_MIN_IN = 3.0
CHARACTER_X_MAX_IN = 4.6
PARENTHETICAL_X_MIN_IN = 2.5
PARENTHETICAL_X_MAX_IN = 3.6
DIALOGUE_X_MIN_IN = 1.5
DIALOGUE_X_MAX_IN = 2.8
RIGHT_ALIGNED_X_MIN_IN = 5.8

SCENE_HEADING_PREFIXES = (
    "INT.",
    "EXT.",
    "INT/EXT",
    "I/E",
    "INT./EXT.",
    "EXT./INT.",
)
KNOWN_TRANSITIONS = {
    "FADE IN:",
    "FADE OUT.",
    "CUT TO:",
    "SMASH CUT:",
    "MATCH CUT:",
    "DISSOLVE TO:",
    "BACK TO:",
}


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]
    page: int
    page_width: int
    page_height: int


class ParsingError(ValueError):
    """Raised when OCR output cannot produce a usable script."""


def rasterize(
    pdf_bytes: bytes,
    dpi: int = DPI,
    thread_count: int | None = None,
):
    """Rasterize PDF bytes into PIL images with pdf2image, using Poppler threads."""

    from pdf2image import convert_from_bytes

    threads = thread_count if thread_count is not None else DEFAULT_RASTERIZE_THREADS
    return convert_from_bytes(pdf_bytes, dpi=dpi, thread_count=threads)


def ocr_pages(images: Iterable[object]) -> list[list[OCRLine]]:
    """Run Apple Vision OCR over page images and normalize results."""

    pages: list[list[OCRLine]] = []
    for page_number, image in enumerate(images, start=1):
        width, height = image.size
        page_lines: list[OCRLine] = []
        for result in _recognize_image(image):
            text, confidence, bbox = _unpack_ocr_result(result)
            page_lines.append(
                OCRLine(
                    text=str(text).strip(),
                    confidence=float(confidence),
                    bbox=_normalize_ocr_bbox(bbox, width, height),
                    page=page_number,
                    page_width=width,
                    page_height=height,
                )
            )
        pages.append(page_lines)
    return pages


def _recognize_image(image: object):
    try:
        from ocrmac import OCR
    except ImportError:
        OCR = None

    if OCR is not None:
        return OCR(image).recognize()

    from ocrmac.ocrmac import text_from_image

    return text_from_image(
        image,
        recognition_level="accurate",
        confidence_threshold=0.0,
        detail=True,
    )


def _unpack_ocr_result(result):
    if len(result) < 3:
        raise ParsingError("OCR returned an unexpected result shape.")
    return result[0], result[1], result[2]


def _normalize_ocr_bbox(
    bbox: Iterable[float],
    page_width: int,
    page_height: int,
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bbox)
    if len(values) != 4:
        raise ParsingError("OCR returned an unexpected bounding box shape.")

    x, y, third, fourth = values
    if all(0.0 <= value <= 1.0 for value in values):
        # ocrmac.text_from_image returns Vision coordinates: x, y, width, height
        # with origin at the lower-left. The parser expects PIL-style pixels.
        x0 = x * page_width
        y1 = (1 - y) * page_height
        x1 = x0 + third * page_width
        y0 = y1 - fourth * page_height
        return tuple(round(value, 6) for value in (x0, y0, x1, y1))

    return values


def prefilter_ocr_lines(lines: Iterable[OCRLine]) -> list[OCRLine]:
    filtered = []
    for line in lines:
        if not line.text.strip():
            continue
        if line.confidence < MIN_OCR_CONFIDENCE:
            continue
        y0, y1 = line.bbox[1], line.bbox[3]
        if y0 < line.page_height * HEADER_FOOTER_MARGIN_RATIO:
            continue
        if y1 > line.page_height * (1 - HEADER_FOOTER_MARGIN_RATIO):
            continue
        filtered.append(line)
    return filtered


def classify_lines(ocr_pages: list[list[OCRLine]]) -> Script:
    filtered_pages = [prefilter_ocr_lines(page) for page in ocr_pages]
    flat_lines = [
        line
        for page in filtered_pages
        for line in sorted(page, key=lambda item: (item.bbox[1], item.bbox[0]))
    ]

    parsed: list[ParsedLine] = []
    current_character: str | None = None
    for index, line in enumerate(flat_lines):
        next_line = _next_non_empty(flat_lines, index + 1)
        line_type, character, modifier = _classify_line(
            line=line,
            next_line=next_line,
            current_character=current_character,
        )
        parsed_line = ParsedLine(
            type=line_type,
            text=line.text.strip(),
            page=line.page,
            bbox=line.bbox,
            character=character,
            modifier=modifier,
            confidence=line.confidence,
        )

        if line_type == LineType.CHARACTER:
            parsed_line = _normalize_character_line(parsed_line)
            current_character = parsed_line.character
        elif line_type in {LineType.SCENE_HEADING, LineType.ACTION, LineType.TRANSITION}:
            current_character = None

        parsed.append(parsed_line)

    return _build_script(_merge_lines(parsed))


def validate_parse_quality(script: Script, min_dialogue_lines: int = 5) -> None:
    dialogue_count = sum(1 for line in script.lines if line.type == LineType.DIALOGUE)
    if dialogue_count < min_dialogue_lines:
        raise ParsingError(
            "Parsing produced very little dialogue. The PDF may be too low quality for OCR."
        )


def apply_character_renames(script: Script, renames: dict[str, str]) -> Script:
    cleaned = {source.strip(): target.strip() for source, target in renames.items() if target.strip()}
    lines = []
    for line in script.lines:
        character = cleaned.get(line.character or "", line.character)
        text = cleaned.get(line.text, line.text) if line.type == LineType.CHARACTER else line.text
        lines.append(replace(line, text=text, character=character))
    return _build_script(lines)


def reclassify_line(script: Script, line_index: int, new_type: LineType | None) -> Script:
    lines = list(script.lines)
    if new_type is None:
        del lines[line_index]
        return _build_script(lines)

    line = lines[line_index]
    character = line.character if new_type in _CHARACTER_CARRYING_TYPES else None
    modifier = line.modifier if new_type == LineType.CHARACTER else None
    lines[line_index] = replace(line, type=new_type, character=character, modifier=modifier)
    return _build_script(lines)


def _classify_line(
    line: OCRLine,
    next_line: OCRLine | None,
    current_character: str | None,
) -> tuple[LineType, str | None, str | None]:
    text = line.text.strip()
    text_upper = text.upper()
    x_in = line.bbox[0] / DPI

    if _is_scene_heading(text_upper, x_in):
        return LineType.SCENE_HEADING, None, None
    if _is_transition(text_upper, x_in):
        return LineType.TRANSITION, None, None
    if _is_character_cue(text, x_in, next_line):
        character, modifier = _split_character_modifier(text)
        return LineType.CHARACTER, character, modifier
    if _is_parenthetical(text, x_in):
        return LineType.PARENTHETICAL, current_character, None
    if current_character and DIALOGUE_X_MIN_IN <= x_in <= DIALOGUE_X_MAX_IN:
        return LineType.DIALOGUE, current_character, None
    return LineType.ACTION, None, None


def _is_scene_heading(text_upper: str, x_in: float) -> bool:
    return x_in <= LEFT_MARGIN_MAX_IN and text_upper.startswith(SCENE_HEADING_PREFIXES)


def _is_transition(text_upper: str, x_in: float) -> bool:
    return (text_upper in KNOWN_TRANSITIONS or text_upper.endswith("TO:")) and (
        x_in >= RIGHT_ALIGNED_X_MIN_IN
    )


def _is_character_cue(text: str, x_in: float, next_line: OCRLine | None) -> bool:
    if not (CHARACTER_X_MIN_IN <= x_in <= CHARACTER_X_MAX_IN):
        return False
    if not _is_all_caps_script_token(text):
        return False
    if next_line is None:
        return False
    next_text = next_line.text.strip()
    next_x_in = next_line.bbox[0] / DPI
    return _is_parenthetical(next_text, next_x_in) or (
        DIALOGUE_X_MIN_IN <= next_x_in <= DIALOGUE_X_MAX_IN
    )


def _is_parenthetical(text: str, x_in: float) -> bool:
    return (
        PARENTHETICAL_X_MIN_IN <= x_in <= PARENTHETICAL_X_MAX_IN
        and text.startswith("(")
        and text.endswith(")")
    )


def _is_all_caps_script_token(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and text.upper() == text


def _split_character_modifier(text: str) -> tuple[str, str | None]:
    match = re.match(r"^(?P<character>.*?)\s*\((?P<modifier>[^)]+)\)$", text.strip())
    if match:
        return _normalize_character_name(match.group("character")), match.group("modifier").strip()
    return _normalize_character_name(text), None


def _normalize_character_name(text: str) -> str:
    cleaned = text.strip().replace("[", "").replace("]", "")
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9.']+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_character_line(line: ParsedLine) -> ParsedLine:
    character = line.character or line.text
    return replace(line, text=character, character=character)


def _next_non_empty(lines: list[OCRLine], start: int) -> OCRLine | None:
    for line in lines[start:]:
        if line.text.strip():
            return line
    return None


def _merge_lines(lines: list[ParsedLine]) -> list[ParsedLine]:
    merged: list[ParsedLine] = []
    for line in lines:
        if (
            merged
            and line.type in {LineType.ACTION, LineType.DIALOGUE}
            and merged[-1].type == line.type
            and merged[-1].character == line.character
            and merged[-1].page == line.page
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                text=f"{previous.text} {line.text}".strip(),
                bbox=_union_bbox(previous.bbox, line.bbox),
                confidence=min(previous.confidence, line.confidence),
            )
        else:
            merged.append(line)
    return merged


def _union_bbox(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


_CHARACTER_CARRYING_TYPES = {
    LineType.CHARACTER,
    LineType.PARENTHETICAL,
    LineType.DIALOGUE,
}


def _build_script(lines: list[ParsedLine]) -> Script:
    normalized_lines = _propagate_dialogue_characters(lines)
    scenes: list[Scene] = []
    scene_starts = [
        index for index, line in enumerate(normalized_lines) if line.type == LineType.SCENE_HEADING
    ]
    if not scene_starts and normalized_lines:
        scene_starts = [0]

    for scene_index, start_line in enumerate(scene_starts):
        end_line = scene_starts[scene_index + 1] if scene_index + 1 < len(scene_starts) else len(normalized_lines)
        heading = (
            normalized_lines[start_line].text
            if normalized_lines[start_line].type == LineType.SCENE_HEADING
            else f"Scene {scene_index + 1}"
        )
        characters = {
            line.character
            for line in normalized_lines[start_line:end_line]
            if line.type in _CHARACTER_CARRYING_TYPES and line.character
        }
        scenes.append(
            Scene(
                index=scene_index,
                number=None,
                heading=heading,
                start_line=start_line,
                end_line=end_line,
                characters=set(characters),
            )
        )

    characters = {
        line.character
        for line in normalized_lines
        if line.type in _CHARACTER_CARRYING_TYPES and line.character
    }
    return Script(lines=normalized_lines, scenes=scenes, characters=set(characters))


def _propagate_dialogue_characters(lines: list[ParsedLine]) -> list[ParsedLine]:
    propagated: list[ParsedLine] = []
    current_character: str | None = None
    for line in lines:
        if line.type == LineType.CHARACTER:
            current_character = line.character or line.text
            propagated.append(replace(line, text=current_character, character=current_character))
        elif line.type in {LineType.PARENTHETICAL, LineType.DIALOGUE}:
            propagated.append(replace(line, character=line.character or current_character))
        else:
            if line.type in {LineType.SCENE_HEADING, LineType.ACTION, LineType.TRANSITION}:
                current_character = None
            propagated.append(line)
    return propagated
