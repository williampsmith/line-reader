"""Gemini-powered screenplay parser."""

from __future__ import annotations

import io
import importlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field, ValidationError
from PIL import Image

from models import LineType, ParsedLine, Script
from parser import OCRLine, _build_script, _normalize_character_name

MAX_GEMINI_IMAGE_EDGE_PX = 1400
DEFAULT_GEMINI_MAX_WORKERS = 4


class GeminiParseError(RuntimeError):
    """Raised when Gemini cannot return a valid screenplay parse."""


class GeminiParsedLine(BaseModel):
    type: LineType
    text: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    character: str | None = None
    modifier: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class GeminiParsedScript(BaseModel):
    lines: list[GeminiParsedLine]


class LocalPart:
    """Small test double compatible with the fields used by GenAI parts."""

    def __init__(self, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type


class GeminiScriptParser:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        timeout_ms: int = 45_000,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_ms = timeout_ms
        self.client = client

    @classmethod
    def from_api_key_file(
        cls,
        api_key_path: str | Path,
        *,
        model: str = "gemini-2.5-flash-lite",
        timeout_ms: int = 45_000,
    ) -> "GeminiScriptParser":
        path = Path(api_key_path).expanduser()
        if not path.exists():
            raise GeminiParseError(f"Gemini API key file not found at {path}.")
        api_key = path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise GeminiParseError(f"Gemini API key file at {path} is empty.")
        return cls(api_key=api_key, model=model, timeout_ms=timeout_ms)

    def parse(
        self,
        images: Iterable[object],
        ocr_pages: list[list[OCRLine]],
    ) -> Script:
        lines: list[ParsedLine] = []
        for page_index, image in enumerate(images):
            page_ocr = ocr_pages[page_index] if page_index < len(ocr_pages) else []
            page_script = self.parse_page(image, page_ocr, page_index + 1)
            lines.extend(page_script.lines)
        return _build_script(lines)

    def parse_page(
        self,
        image: object,
        ocr_lines: list[OCRLine],
        page_number: int,
        *,
        use_image: bool = True,
    ) -> Script:
        client = self._client()
        config = self._generate_config()
        if not use_image:
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=self._ocr_only_contents([ocr_lines], page_number),
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                raise GeminiParseError(
                    f"Gemini OCR-only request failed on page {page_number}: {exc}"
                ) from exc
            return self._script_from_response(response.text)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=self._contents([image], [ocr_lines], first_page_number=page_number),
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - SDK transports expose varied exceptions.
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=self._ocr_only_contents([ocr_lines], page_number),
                    config=config,
                )
            except Exception as retry_exc:  # noqa: BLE001
                raise GeminiParseError(
                    "Gemini request failed on page "
                    f"{page_number}: image parse failed with {exc}; "
                    f"OCR-only retry failed with {retry_exc}"
                ) from retry_exc
        return self._script_from_response(response.text)

    def _client(self):
        if self.client is not None:
            return self.client

        genai = importlib.import_module("google.genai")
        try:
            from google.genai import types
            http_options = types.HttpOptions(timeout=self.timeout_ms)
        except ImportError:
            http_options = None

        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self.client

    def _generate_config(self):
        try:
            from google.genai import types
        except ImportError:
            return None

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=GeminiParsedScript.model_json_schema(),
        )

    def _contents(
        self,
        images: Iterable[object],
        ocr_pages: list[list[OCRLine]],
        *,
        first_page_number: int = 1,
    ) -> list[Any]:
        contents: list[Any] = [
            _parser_prompt(),
            "OCR hints:\n" + _ocr_hints(ocr_pages),
        ]
        for index, image in enumerate(images, start=first_page_number):
            contents.append(f"Page {index} image:")
            contents.append(_image_part(image))
        return contents

    def _ocr_only_contents(
        self,
        ocr_pages: list[list[OCRLine]],
        page_number: int,
    ) -> list[Any]:
        return [
            _parser_prompt(),
            (
                f"Page {page_number} image parsing timed out. "
                "Use these OCR text and layout hints to produce the same JSON schema."
            ),
            "OCR hints:\n" + _ocr_hints(ocr_pages),
        ]

    def _script_from_response(self, text: str) -> Script:
        try:
            data = GeminiParsedScript.model_validate_json(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise GeminiParseError("Gemini did not return valid JSON screenplay data.") from exc

        parsed_lines = [_to_parsed_line(line) for line in data.lines if line.text.strip()]
        return _build_script(parsed_lines)


def combine_page_scripts(scripts: Iterable[Script]) -> Script:
    lines: list[ParsedLine] = []
    for script in scripts:
        lines.extend(script.lines)
    return _build_script(lines)


ProgressCallback = Callable[[int, int], None]
PageParserCallable = Callable[[Any, Any, list[OCRLine], int], Script]


def parse_pages_in_parallel(
    parser: Any,
    images: Sequence[Any],
    ocr_per_page: Sequence[list[OCRLine]],
    *,
    max_workers: int = DEFAULT_GEMINI_MAX_WORKERS,
    parser_callable: PageParserCallable | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[Script]:
    """Parse pages concurrently while preserving original page ordering.

    The optional ``parser_callable`` lets tests inject a fake page parser. It
    receives ``(parser, image, page_ocr, page_number)``. When omitted, falls
    back to ``parser.parse_page(image, page_ocr, page_number)``.
    """

    if not images:
        if on_progress is not None:
            on_progress(0, 0)
        return []

    total = len(images)
    if parser_callable is None:
        def parser_callable(parser_obj, image, page_ocr, page_number):
            return parser_obj.parse_page(image, page_ocr, page_number)

    results: list[Script | None] = [None] * total
    completed = 0
    workers = max(1, min(max_workers, total))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for index, image in enumerate(images):
            page_ocr = ocr_per_page[index] if index < len(ocr_per_page) else []
            future = executor.submit(parser_callable, parser, image, page_ocr, index + 1)
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

    if any(result is None for result in results):
        raise GeminiParseError("Gemini parse returned no result for one or more pages.")

    return [result for result in results if result is not None]


def _to_parsed_line(line: GeminiParsedLine) -> ParsedLine:
    character = _normalize_character_name(line.character) if line.character else None
    text = line.text.strip()
    if line.type == LineType.CHARACTER:
        character = _normalize_character_name(character or text)
        text = character
    return ParsedLine(
        type=line.type,
        text=text,
        page=line.page,
        bbox=tuple(line.bbox),
        character=character,
        modifier=line.modifier,
        confidence=line.confidence,
    )


def _image_part(image: object):
    data = _image_png_bytes(image)
    try:
        from google.genai import types
    except ImportError:
        return LocalPart(data=data, mime_type="image/png")
    return types.Part.from_bytes(data=data, mime_type="image/png")


def _image_png_bytes(image: object) -> bytes:
    if hasattr(image, "copy") and hasattr(image, "thumbnail") and hasattr(image, "size"):
        image = image.copy()
        image.thumbnail(
            (MAX_GEMINI_IMAGE_EDGE_PX, MAX_GEMINI_IMAGE_EDGE_PX),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _ocr_hints(ocr_pages: list[list[OCRLine]]) -> str:
    rows = []
    for page in ocr_pages:
        for line in page:
            rows.append(
                json.dumps(
                    {
                        "page": line.page,
                        "text": line.text,
                        "confidence": line.confidence,
                        "bbox": line.bbox,
                    }
                )
            )
    return "\n".join(rows)


def _parser_prompt() -> str:
    return """You are parsing audition sides into screenplay structure.
Return only JSON matching the provided schema.
Classify every visible script line as one of: scene_heading, action, character,
parenthetical, dialogue, transition.
Set character on character cues, parentheticals, and dialogue.
Preserve source order. Strip OCR artifacts around character names such as
leading commas and square brackets. Do not invent dialogue."""
