import types
from dataclasses import replace

import pandas as pd

import app
from app import (
    _practice_outputs,
    _state,
    _tab_switch_js,
    commit_casting,
    prepare_casting,
    process_pdf,
    start_practice,
    update_dialogue_pacing,
)
from ai_parser import GeminiParseError
from models import LineType, ParsedLine, PracticeQueueItem, Scene, Script, VoiceAssignment
from practice import PracticeSession


def script_with_characters():
    lines = [
        ParsedLine(LineType.SCENE_HEADING, "INT. KITCHEN - DAY", 1, (0, 0, 1, 1)),
        ParsedLine(LineType.DIALOGUE, "Hi.", 1, (0, 0, 1, 1), character="JOHN"),
        ParsedLine(LineType.DIALOGUE, "Hello.", 1, (0, 0, 1, 1), character="SARAH"),
    ]
    return Script(
        lines=lines,
        scenes=[
            Scene(
                index=0,
                number=None,
                heading="INT. KITCHEN - DAY",
                start_line=0,
                end_line=len(lines),
                characters={"JOHN", "SARAH"},
            )
        ],
        characters={"JOHN", "SARAH"},
    )


def test_prepare_casting_requires_explicit_user_selection():
    state = _state()
    state.script = script_with_characters()

    user_update, status, tab_update = prepare_casting(state)

    assert user_update["value"] is None
    assert "Choose your role" in status
    assert tab_update.startswith("cast:")


def test_commit_casting_uses_voice_overrides_on_state():
    state = _state()
    state.script = script_with_characters()
    state.voice_overrides = {"SARAH": "en-US-Chirp3-HD-Aoede"}

    updated_state, scene_update, status, tab_update = commit_casting(state, "JOHN")

    assert updated_state.assignment.user_character == "JOHN"
    assert updated_state.assignment.voice_for_character == {
        "SARAH": "en-US-Chirp3-HD-Aoede"
    }
    assert scene_update["value"].startswith("1. INT. KITCHEN")
    assert "Casting saved" in status
    assert tab_update.startswith("scenes:")


def test_commit_casting_rejects_missing_ai_voice_assignment():
    state = _state()
    state.script = script_with_characters()
    state.voice_overrides = {}

    updated_state, scene_update, status, tab_update = commit_casting(state, "JOHN")

    assert updated_state.assignment is None
    assert scene_update["value"] is None
    assert "Choose a voice for SARAH" in status
    assert tab_update.startswith("cast:")


def test_set_voice_override_updates_state():
    from app import set_voice_override

    state = _state()
    state.voice_overrides = {"SARAH": "voice-a"}

    state = set_voice_override(state, "SARAH", "voice-b")
    assert state.voice_overrides["SARAH"] == "voice-b"

    state = set_voice_override(state, "SARAH", "")
    assert "SARAH" not in state.voice_overrides


def test_update_default_voices_seeds_overrides_for_other_characters():
    from app import update_default_voices

    state = _state()
    state.script = script_with_characters()

    state, character_update, voice_update, status = update_default_voices(state, "JOHN")

    assert "SARAH" in state.voice_overrides
    assert state.voice_overrides["SARAH"]
    assert "JOHN" not in state.voice_overrides
    assert character_update["choices"] == ["SARAH"]
    assert character_update["value"] == "SARAH"
    assert voice_update["value"] == state.voice_overrides["SARAH"]
    assert "SARAH" in status


def test_update_voice_override_updates_summary():
    from app import update_default_voices, update_voice_override

    state = _state()
    state.script = script_with_characters()
    state, _, _, _ = update_default_voices(state, "JOHN")

    state, summary = update_voice_override(
        state,
        "JOHN",
        "SARAH",
        "en-US-Chirp3-HD-Kore",
    )

    assert state.voice_overrides["SARAH"] == "en-US-Chirp3-HD-Kore"
    assert "SARAH" in summary
    assert "en-US-Chirp3-HD-Kore" in summary


def test_process_pdf_streams_status_and_selects_review_tab(monkeypatch, tmp_path):
    pdf = tmp_path / "side.pdf"
    pdf.write_bytes(b"%PDF")
    script = script_with_characters()

    monkeypatch.setattr(app, "rasterize", lambda pdf_bytes: ["page-1", "page-2"])
    monkeypatch.setattr(app, "ocr_pages", lambda images: [["ocr-1"], ["ocr-2"]])
    monkeypatch.setattr(app, "_gemini_parser_from_state", lambda state: object())
    monkeypatch.setattr(
        app,
        "_parse_gemini_page",
        lambda parser, image, ocr, page_number, **kwargs: script,
    )
    monkeypatch.setattr(app, "classify_lines", lambda ocr: (_ for _ in ()).throw(AssertionError("local fallback should not run")))
    monkeypatch.setattr(app, "validate_parse_quality", lambda parsed: None)

    outputs = list(process_pdf(types.SimpleNamespace(name=str(pdf)), _state()))

    statuses = [o[6] for o in outputs]
    assert "Rasterizing pages" in statuses[0]
    assert "Running OCR" in statuses[1]
    assert any("0 of 2 pages parsed" in s for s in statuses)
    assert any("Parsing screenplay with Gemini" in s for s in statuses)
    assert len(outputs[-1][0].script.lines) == 6
    assert outputs[-1][-1].startswith("review:")


def test_process_pdf_falls_back_to_local_parser_when_gemini_fails(monkeypatch, tmp_path):
    pdf = tmp_path / "side.pdf"
    pdf.write_bytes(b"%PDF")
    script = script_with_characters()
    state = _state()
    state.config = replace(
        state.config,
        parser=replace(state.config.parser, fallback_to_local=True),
    )

    monkeypatch.setattr(app, "rasterize", lambda pdf_bytes: ["page-image"])
    monkeypatch.setattr(app, "ocr_pages", lambda images: [["ocr-line"]])
    monkeypatch.setattr(app, "_gemini_parser_from_state", lambda state: object())
    monkeypatch.setattr(
        app,
        "_parse_gemini_page",
        lambda parser, image, ocr, page_number, **kwargs: (_ for _ in ()).throw(GeminiParseError("quota")),
    )
    monkeypatch.setattr(app, "classify_lines", lambda ocr: script)
    monkeypatch.setattr(app, "validate_parse_quality", lambda parsed: None)

    outputs = list(process_pdf(types.SimpleNamespace(name=str(pdf)), state))

    assert any("Gemini parsing failed: quota" in output[6] for output in outputs)
    assert any("Falling back to local parser" in output[6] for output in outputs)
    assert outputs[-1][0].script is script


def test_process_pdf_does_not_fall_back_when_disabled(monkeypatch, tmp_path):
    pdf = tmp_path / "side.pdf"
    pdf.write_bytes(b"%PDF")
    state = _state()

    monkeypatch.setattr(app, "rasterize", lambda pdf_bytes: ["page-image"])
    monkeypatch.setattr(app, "ocr_pages", lambda images: [["ocr-line"]])
    monkeypatch.setattr(app, "_gemini_parser_from_state", lambda state: object())
    monkeypatch.setattr(
        app,
        "_parse_gemini_page",
        lambda parser, image, ocr, page_number, **kwargs: (_ for _ in ()).throw(GeminiParseError("quota")),
    )
    monkeypatch.setattr(
        app,
        "classify_lines",
        lambda ocr: (_ for _ in ()).throw(AssertionError("local fallback should not run")),
    )

    outputs = list(process_pdf(types.SimpleNamespace(name=str(pdf)), state))

    assert any("Could not process PDF: quota" in output[6] for output in outputs)
    assert not any("Falling back to local parser" in output[6] for output in outputs)


class FakeTTS:
    def __init__(self):
        self.speaking_rate = 1.0
        self.synthesized = []

    def synthesize(self, text, voice_id, speaking_rate=1.0):
        self.synthesized.append((text, voice_id, speaking_rate))
        return b"audio"

    def prefetch_next_ai(self, queue, current_index, speaking_rate=1.0):
        return None


def test_practice_outputs_show_current_line_and_next_two_only():
    state = _state()
    state.session = PracticeSession(
        queue=[
            PracticeQueueItem("ai", "SARAH", "Are you ready?", "voice-sarah", 1),
            PracticeQueueItem("user", "JOHN", "I was born ready.", None, 2),
            PracticeQueueItem("ai", "MARCUS", "Then let's move.", "voice-marcus", 3),
            PracticeQueueItem("user", "JOHN", "Right behind you.", None, 4),
        ],
        tts_client=FakeTTS(),
        audio_player=state.pending_audio.append,
    )
    state.session.start()

    first = _practice_outputs(state)
    second = _practice_outputs(state)

    assert "Are you ready?" not in first[1]
    assert "Line 1 of 4" in first[1]
    assert "CURRENTLY SPEAKING" in first[2]
    assert 'class="line-card current"' in first[2]
    assert 'class="line-text"' in first[2]
    assert 'class="line-speaker">SARAH<' in first[2]
    assert "YOUR LINE" in first[2]
    assert 'class="line-card upcoming user-line"' in first[2]
    assert "I was born ready." in first[2]
    assert "Then let&#x27;s move." in first[2]
    assert "Right behind you." not in first[2]
    assert "# Are you ready?" not in first[2]
    assert "## I was born ready." not in first[2]
    assert first[3].endswith(".mp3")
    assert second[3]["__type__"] == "update"


def test_start_practice_selects_practice_tab():
    state = _state()
    state.script = script_with_characters()
    state.assignment = VoiceAssignment(
        user_character="JOHN",
        voice_for_character={"SARAH": "voice-sarah"},
    )
    state.tts_client = FakeTTS()

    outputs = start_practice(state, "1. INT. KITCHEN - DAY (p.1) - JOHN, SARAH", 800, 0.9)

    assert outputs[-2].startswith("practice:")
    assert state.session.speaking_rate == 0.9
    assert state.tts_client.synthesized == []
    state.session.manual_done()
    assert state.tts_client.synthesized[0] == ("Hello.", "voice-sarah", 0.9)


def test_update_parser_mode_toggles_gemini_use_image():
    from app import update_parser_mode

    state = _state()
    assert state.config.parser.gemini_use_image is True

    state, status = update_parser_mode(state, "Text-only (faster, ~2-5s per page)")

    assert state.config.parser.gemini_use_image is False
    assert "Text-only" in status

    state, status = update_parser_mode(state, "Vision (richer layout, slower, ~10-30s per page)")

    assert state.config.parser.gemini_use_image is True
    assert "Vision" in status


def test_update_dialogue_pacing_updates_session_and_tts_client():
    state = _state()
    state.tts_client = FakeTTS()
    state.session = PracticeSession(
        queue=[PracticeQueueItem("ai", "SARAH", "Are you ready?", "voice-sarah", 1)],
        tts_client=state.tts_client,
        audio_player=lambda audio: None,
    )

    updated_state, status = update_dialogue_pacing(state, 1.2)

    assert updated_state.dialogue_pacing == 1.2
    assert updated_state.tts_client.speaking_rate == 1.2
    assert updated_state.session.speaking_rate == 1.2
    assert "1.20x" in status


def test_tab_switch_js_clicks_expected_tab_labels():
    js = _tab_switch_js()

    assert "Upload" in js
    assert "Review" in js
    assert "Cast" in js
    assert "Scene" in js
    assert "Rehearse" in js
    assert "tab.click()" in js


def test_apply_review_table_renames_reclassifies_and_deletes():
    from app import apply_review_table

    state = _state()
    state.script = Script(
        lines=[
            ParsedLine(LineType.SCENE_HEADING, "INT. KITCHEN - DAY", 1, (0, 0, 1, 1)),
            ParsedLine(LineType.CHARACTER, "5ARAH", 1, (0, 0, 1, 1), character="5ARAH"),
            ParsedLine(LineType.DIALOGUE, "Hello there.", 1, (0, 0, 1, 1), character="5ARAH"),
            ParsedLine(LineType.ACTION, "OOC noise", 1, (0, 0, 1, 1)),
        ],
        scenes=[],
        characters={"5ARAH"},
    )

    rows = [
        ["1", "scene_heading", "", "INT. KITCHEN - DAY", "1", ""],
        ["2", "character", "SARAH", "SARAH", "1", ""],
        ["3", "dialogue", "SARAH", "Hello there!", "1", ""],
        ["4", "action", "", "OOC noise", "1", "x"],
    ]

    updated_state, characters_table, scenes_table, grid, summary, status = apply_review_table(
        state, rows
    )

    line_texts = [line.text for line in updated_state.script.lines]
    assert "Hello there!" in line_texts
    assert "OOC noise" not in line_texts
    assert updated_state.script.characters == {"SARAH"}
    assert "Removed 1 line" in status
    assert ">SARAH<" in summary or "SARAH" in str(characters_table)


def test_practice_status_uses_human_readable_copy():
    from app import _practice_status

    state = _state()
    state.session = PracticeSession(
        queue=[
            PracticeQueueItem("ai", "SARAH", "Are you ready?", "voice-sarah", 1),
            PracticeQueueItem("user", "JOHN", "I was born ready.", None, 2),
        ],
        tts_client=FakeTTS(),
        audio_player=state.pending_audio.append,
    )
    state.session.start()

    status = _practice_status(state)

    assert "ai_turn" not in status
    assert "VAD" not in status
    assert "Line" in status
    assert "AI is speaking" in status
