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
