import json
import io

import pytest
from PIL import Image

from ai_parser import GeminiParseError, GeminiScriptParser, _image_png_bytes
from models import LineType
from parser import OCRLine


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


class FailingModels:
    def generate_content(self, **kwargs):
        raise RuntimeError("504 DEADLINE_EXCEEDED")


class FailingClient:
    models = FailingModels()


class FailsOnceModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return FakeResponse(self.response_text)


class FailsOnceClient:
    def __init__(self, response_text):
        self.models = FailsOnceModels(response_text)


class FakeImage:
    size = (1000, 2000)

    def save(self, buffer, format):
        buffer.write(b"png-bytes")


def ocr_line(text, page=1):
    return OCRLine(
        text=text,
        confidence=0.9,
        bbox=(100.0, 100.0, 300.0, 140.0),
        page=page,
        page_width=1000,
        page_height=2000,
    )


def test_gemini_response_maps_to_script_dataclasses():
    response = {
        "lines": [
            {
                "type": "scene_heading",
                "text": "INT. KITCHEN - DAY",
                "page": 1,
                "bbox": [100, 100, 500, 130],
                "character": None,
                "confidence": 0.95,
            },
            {
                "type": "character",
                "text": "SARAH",
                "page": 1,
                "bbox": [420, 200, 550, 230],
                "character": "SARAH",
                "confidence": 0.94,
            },
            {
                "type": "dialogue",
                "text": "Are you ready?",
                "page": 1,
                "bbox": [260, 240, 700, 290],
                "character": "SARAH",
                "confidence": 0.93,
            },
            {
                "type": "character",
                "text": "[JOHN]",
                "page": 1,
                "bbox": [420, 330, 550, 360],
                "character": "[JOHN]",
                "confidence": 0.94,
            },
            {
                "type": "dialogue",
                "text": "I was born ready.",
                "page": 1,
                "bbox": [260, 370, 700, 420],
                "character": "[JOHN]",
                "confidence": 0.93,
            },
        ]
    }
    client = FakeClient(json.dumps(response))
    parser = GeminiScriptParser(api_key="test-key", client=client)

    script = parser.parse([FakeImage()], [[ocr_line("INT. KITCHEN - DAY")]])

    assert [line.type for line in script.lines] == [
        LineType.SCENE_HEADING,
        LineType.CHARACTER,
        LineType.DIALOGUE,
        LineType.CHARACTER,
        LineType.DIALOGUE,
    ]
    assert script.characters == {"SARAH", "JOHN"}
    assert script.scenes[0].heading == "INT. KITCHEN - DAY"
    assert script.scenes[0].characters == {"SARAH", "JOHN"}
    assert client.models.calls
    assert client.models.calls[0]["model"] == "gemini-2.5-flash-lite"


def test_gemini_parser_sends_images_and_ocr_hints():
    client = FakeClient(json.dumps({"lines": []}))
    parser = GeminiScriptParser(api_key="test-key", client=client)

    parser.parse([FakeImage()], [[ocr_line("SARAH")]])

    contents = client.models.calls[0]["contents"]
    assert any("OCR hints" in part for part in contents if isinstance(part, str))
    assert any(
        (
            hasattr(part, "mime_type")
            and part.mime_type == "image/png"
        )
        or (
            getattr(part, "inline_data", None) is not None
            and part.inline_data.mime_type == "image/png"
        )
        for part in contents
    )


def test_gemini_parser_calls_model_once_per_page():
    response = {
        "lines": [
            {
                "type": "scene_heading",
                "text": "INT. ROOM - DAY",
                "page": 1,
                "bbox": [0, 0, 1, 1],
            }
        ]
    }
    client = FakeClient(json.dumps(response))
    parser = GeminiScriptParser(api_key="test-key", client=client)

    parser.parse(
        [FakeImage(), FakeImage()],
        [[ocr_line("Page one", page=1)], [ocr_line("Page two", page=2)]],
    )

    assert len(client.models.calls) == 2
    assert "Page 1 image:" in client.models.calls[0]["contents"]
    assert "Page 2 image:" in client.models.calls[1]["contents"]


def test_gemini_parser_sets_request_timeout_on_real_client(monkeypatch):
    created = {}

    class FakeGenai:
        class types:
            class HttpOptions:
                def __init__(self, timeout):
                    self.timeout = timeout

        class Client:
            def __init__(self, **kwargs):
                created.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "google.genai", FakeGenai)
    parser = GeminiScriptParser(api_key="test-key", timeout_ms=12345)

    parser._client()

    assert created["http_options"].timeout == 12345


def test_gemini_sdk_errors_are_wrapped_as_parse_errors():
    parser = GeminiScriptParser(api_key="test-key", client=FailingClient())

    with pytest.raises(GeminiParseError, match="page 3.*504 DEADLINE_EXCEEDED"):
        parser.parse_page(FakeImage(), [], 3)


def test_gemini_image_timeout_retries_with_ocr_only():
    response = {
        "lines": [
            {
                "type": "character",
                "text": "SARAH",
                "page": 1,
                "bbox": [100, 100, 200, 120],
                "character": "SARAH",
            },
            {
                "type": "dialogue",
                "text": "We have to go.",
                "page": 1,
                "bbox": [100, 130, 400, 170],
                "character": "SARAH",
            },
        ]
    }
    client = FailsOnceClient(json.dumps(response))
    parser = GeminiScriptParser(api_key="test-key", client=client)

    script = parser.parse_page(FakeImage(), [ocr_line("SARAH")], 1)

    assert len(client.models.calls) == 2
    first_contents = client.models.calls[0]["contents"]
    second_contents = client.models.calls[1]["contents"]
    assert any(getattr(part, "inline_data", None) is not None for part in first_contents)
    assert not any(getattr(part, "inline_data", None) is not None for part in second_contents)
    assert script.characters == {"SARAH"}


def test_large_images_are_downscaled_before_upload():
    image = Image.new("RGB", (2800, 2200), "white")

    data = _image_png_bytes(image)
    resized = Image.open(io.BytesIO(data))

    assert max(resized.size) <= 1400


def test_invalid_gemini_json_raises_parse_error():
    parser = GeminiScriptParser(api_key="test-key", client=FakeClient("not json"))

    with pytest.raises(GeminiParseError, match="valid JSON"):
        parser.parse([FakeImage()], [[]])


def test_parse_page_with_use_image_false_skips_image_attempt():
    response = {
        "lines": [
            {
                "type": "dialogue",
                "text": "Hello.",
                "page": 1,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "character": "SARAH",
            }
        ]
    }
    client = FakeClient(json.dumps(response))
    parser = GeminiScriptParser(api_key="test-key", client=client)

    parser.parse_page(FakeImage(), [ocr_line("SARAH")], 1, use_image=False)

    assert len(client.models.calls) == 1
    contents = client.models.calls[0]["contents"]
    assert not any(getattr(part, "inline_data", None) is not None for part in contents)
    assert any(isinstance(part, str) and "OCR hints" in part for part in contents)


def test_parse_pages_in_parallel_returns_results_in_page_order():
    import threading
    import time

    from ai_parser import parse_pages_in_parallel
    from models import LineType, ParsedLine, Script

    parse_lock = threading.Lock()
    in_flight = {"now": 0, "max": 0}

    def fake_parse(parser, image, page_ocr, page_number):
        with parse_lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        time.sleep(0.05)
        with parse_lock:
            in_flight["now"] -= 1
        return Script(
            lines=[
                ParsedLine(
                    type=LineType.SCENE_HEADING,
                    text=f"PAGE {page_number}",
                    page=page_number,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                )
            ],
            scenes=[],
            characters=set(),
        )

    progress_events: list[tuple[int, int]] = []
    parser = object()
    images = [FakeImage() for _ in range(4)]
    ocr_per_page = [[ocr_line(f"hint {idx}", page=idx + 1)] for idx in range(4)]

    def on_progress(completed: int, total: int) -> None:
        progress_events.append((completed, total))

    page_scripts = parse_pages_in_parallel(
        parser,
        images,
        ocr_per_page,
        max_workers=4,
        parser_callable=fake_parse,
        on_progress=on_progress,
    )

    assert [page.lines[0].text for page in page_scripts] == [
        "PAGE 1",
        "PAGE 2",
        "PAGE 3",
        "PAGE 4",
    ]
    assert in_flight["max"] >= 2
    assert progress_events[-1] == (4, 4)
    assert progress_events == sorted(progress_events)


def test_parse_pages_in_parallel_propagates_errors():
    from ai_parser import parse_pages_in_parallel

    def fake_parse(parser, image, page_ocr, page_number):
        if page_number == 2:
            raise GeminiParseError("page 2 boom")
        from models import LineType, ParsedLine, Script

        return Script(
            lines=[ParsedLine(LineType.ACTION, "ok", page_number, (0, 0, 1, 1))],
            scenes=[],
            characters=set(),
        )

    with pytest.raises(GeminiParseError, match="page 2 boom"):
        parse_pages_in_parallel(
            object(),
            [FakeImage(), FakeImage(), FakeImage()],
            [[], [], []],
            max_workers=3,
            parser_callable=fake_parse,
            on_progress=lambda completed, total: None,
        )
