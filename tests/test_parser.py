import sys
import types

import pytest

from models import LineType
from parser import (
    OCRLine,
    ParsingError,
    apply_character_renames,
    classify_lines,
    ocr_pages,
    prefilter_ocr_lines,
    reclassify_line,
    validate_parse_quality,
)


def ocr(text, x0, y0, x1=None, y1=None, confidence=0.95):
    return OCRLine(
        text=text,
        confidence=confidence,
        bbox=(x0, y0, x1 or x0 + 300, y1 or y0 + 40),
        page=1,
        page_width=1700,
        page_height=2200,
    )


def test_rasterize_passes_thread_count_to_pdf2image(monkeypatch):
    captured = {}

    def fake_convert(pdf_bytes, dpi, thread_count=None):
        captured["thread_count"] = thread_count
        captured["dpi"] = dpi
        return ["page-1", "page-2"]

    fake_module = types.ModuleType("pdf2image")
    fake_module.convert_from_bytes = fake_convert
    monkeypatch.setitem(sys.modules, "pdf2image", fake_module)

    from parser import rasterize

    pages = rasterize(b"pdf-bytes", dpi=200, thread_count=4)

    assert pages == ["page-1", "page-2"]
    assert captured["thread_count"] == 4
    assert captured["dpi"] == 200


def test_ocr_pages_uses_current_ocrmac_text_from_image_api(monkeypatch):
    package = types.ModuleType("ocrmac")
    package.__path__ = []
    implementation = types.ModuleType("ocrmac.ocrmac")
    implementation.text_from_image = lambda image, **kwargs: [
        ("Hello", 0.91, [0.1, 0.8, 0.2, 0.1])
    ]
    monkeypatch.setitem(sys.modules, "ocrmac", package)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", implementation)

    image = types.SimpleNamespace(size=(1000, 2000))

    pages = ocr_pages([image])

    assert pages == [
        [
            OCRLine(
                text="Hello",
                confidence=0.91,
                bbox=(100.0, 200.0, 300.0, 400.0),
                page=1,
                page_width=1000,
                page_height=2000,
            )
        ]
    ]


def test_prefilter_discards_headers_footers_and_low_confidence():
    lines = [
        ocr("Header", 120, 60),
        ocr("Usable action", 180, 300),
        ocr("Footer", 120, 2140),
        ocr("Smudged", 180, 400, confidence=0.49),
    ]

    filtered = prefilter_ocr_lines(lines)

    assert [line.text for line in filtered] == ["Usable action"]


def test_classify_lines_builds_scenes_characters_and_merged_dialogue():
    page = [
        ocr("INT. KITCHEN - DAY", 180, 150),
        ocr("John stares at the door.", 180, 230),
        ocr("JOHN (V.O.)", 700, 310),
        ocr("(quietly)", 560, 370),
        ocr("I thought you left.", 360, 430),
        ocr("I heard the car.", 360, 490),
        ocr("SARAH", 700, 570),
        ocr("I came back.", 360, 630),
        ocr("CUT TO:", 1320, 710),
        ocr("EXT. PORCH - NIGHT", 180, 790),
        ocr("SARAH", 700, 870),
        ocr("We should go.", 360, 930),
    ]

    script = classify_lines([page])

    assert [line.type for line in script.lines] == [
        LineType.SCENE_HEADING,
        LineType.ACTION,
        LineType.CHARACTER,
        LineType.PARENTHETICAL,
        LineType.DIALOGUE,
        LineType.CHARACTER,
        LineType.DIALOGUE,
        LineType.TRANSITION,
        LineType.SCENE_HEADING,
        LineType.CHARACTER,
        LineType.DIALOGUE,
    ]
    assert script.lines[2].text == "JOHN"
    assert script.lines[2].modifier == "V.O."
    assert script.lines[4].text == "I thought you left. I heard the car."
    assert script.lines[4].character == "JOHN"
    assert script.characters == {"JOHN", "SARAH"}
    assert [scene.heading for scene in script.scenes] == [
        "INT. KITCHEN - DAY",
        "EXT. PORCH - NIGHT",
    ]
    assert script.scenes[0].characters == {"JOHN", "SARAH"}
    assert script.scenes[1].characters == {"SARAH"}


def test_character_cues_strip_bracket_and_comma_ocr_noise():
    script = classify_lines(
        [
            [
                ocr("INT. OFFICE - DAY", 180, 150),
                ocr("WENDY]", 700, 230),
                ocr("First line.", 360, 290),
                ocr("[WENDY", 700, 370),
                ocr("Second line.", 360, 430),
                ocr("[WENDY]", 700, 510),
                ocr("Third line.", 360, 570),
                ocr(",JENNIFER", 700, 650),
                ocr("Another line.", 360, 710),
                ocr("D.J.", 700, 790),
                ocr("Music cue.", 360, 850),
            ]
        ]
    )

    assert script.characters == {"WENDY", "JENNIFER", "D.J."}
    assert script.scenes[0].characters == {"WENDY", "JENNIFER", "D.J."}


def test_apply_character_renames_merges_ocr_variants_everywhere():
    script = classify_lines(
        [
            [
                ocr("INT. KITCHEN - DAY", 180, 150),
                ocr("5ARAH", 700, 230),
                ocr("It is me.", 360, 290),
                ocr("SARAH", 700, 370),
                ocr("Still me.", 360, 430),
            ]
        ]
    )

    updated = apply_character_renames(script, {"5ARAH": "SARAH"})

    assert updated.characters == {"SARAH"}
    assert {line.character for line in updated.lines if line.character} == {"SARAH"}
    assert updated.scenes[0].characters == {"SARAH"}


def test_reclassify_line_can_delete_or_change_line_type():
    script = classify_lines(
        [
            [
                ocr("INT. KITCHEN - DAY", 180, 150),
                ocr("JOHN", 700, 230),
                ocr("Hello.", 360, 290),
            ]
        ]
    )

    changed = reclassify_line(script, 1, LineType.ACTION)
    deleted = reclassify_line(changed, 1, None)

    assert changed.lines[1].type == LineType.ACTION
    assert changed.lines[1].character is None
    assert [line.text for line in deleted.lines] == ["INT. KITCHEN - DAY", "Hello."]


def test_validate_parse_quality_rejects_too_little_dialogue():
    script = classify_lines(
        [
            [
                ocr("INT. KITCHEN - DAY", 180, 150),
                ocr("JOHN", 700, 230),
                ocr("Hello.", 360, 290),
            ]
        ]
    )

    with pytest.raises(ParsingError, match="very little dialogue"):
        validate_parse_quality(script, min_dialogue_lines=5)
