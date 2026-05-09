from pathlib import Path

from config import AppConfig, DEFAULT_CONFIG_PATH, load_config
from models import LineType, ParsedLine, PracticeQueueItem, Scene, Script, VoiceAssignment


def test_models_capture_script_and_casting_state():
    line = ParsedLine(
        type=LineType.DIALOGUE,
        text="Where were you?",
        page=2,
        bbox=(72.0, 120.0, 320.0, 145.0),
        character="SARAH",
        confidence=0.93,
    )
    scene = Scene(
        index=0,
        number=None,
        heading="INT. KITCHEN - DAY",
        start_line=0,
        end_line=1,
        characters={"SARAH"},
    )
    script = Script(lines=[line], scenes=[scene], characters={"SARAH"})
    assignment = VoiceAssignment(
        user_character="JOHN",
        voice_for_character={"SARAH": "en-US-Chirp3-HD-Aoede"},
    )
    queue_item = PracticeQueueItem(
        role="ai",
        character="SARAH",
        text="Where were you?",
        voice_id="en-US-Chirp3-HD-Aoede",
        source_line_index=0,
    )

    assert script.scenes[0].characters == {"SARAH"}
    assert assignment.voice_for_character["SARAH"].endswith("Aoede")
    assert queue_item.role == "ai"


def test_load_config_uses_defaults_when_file_is_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.toml"
    monkeypatch.setattr("config.DEFAULT_CONFIG_PATH", missing_path)

    config = load_config()

    assert config.gcp.credentials_path == Path("~/.config/audition-app/gcp-key.json").expanduser()
    assert config.vad.silence_threshold_ms == 800
    assert config.vad.min_speech_duration_ms == 250
    assert config.parser.mode == "gemini"
    assert config.parser.gemini_api_key_path == Path(
        "~/.config/audition-app/gemini-api-key.txt"
    ).expanduser()
    assert config.parser.gemini_model == "gemini-2.5-flash-lite"
    assert config.parser.gemini_timeout_ms == 45000
    assert config.parser.gemini_use_image is True
    assert config.parser.fallback_to_local is False
    assert config.tts.provider == "google"
    assert config.tts.elevenlabs_api_key_path == Path(
        "~/.config/audition-app/elevenlabs-api-key.txt"
    ).expanduser()
    assert config.tts.elevenlabs_model == "eleven_multilingual_v2"
    assert config.ui.port == 7860
    assert config.ui.auto_open_browser is True
    assert DEFAULT_CONFIG_PATH.name == "config.toml"


def test_load_config_overrides_nested_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[gcp]
credentials_path = "~/custom/key.json"

[vad]
silence_threshold_ms = 1200

[parser]
mode = "local"
gemini_api_key_path = "~/custom/gemini-key.txt"
gemini_model = "gemini-2.0-flash"
gemini_timeout_ms = 12000
fallback_to_local = false

[tts]
provider = "elevenlabs"
elevenlabs_api_key_path = "~/custom/elevenlabs-key.txt"
elevenlabs_model = "eleven_turbo_v2_5"

[ui]
port = 9000
auto_open_browser = false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.gcp.credentials_path == Path("~/custom/key.json").expanduser()
    assert config.vad.silence_threshold_ms == 1200
    assert config.vad.min_speech_duration_ms == 250
    assert config.parser.mode == "local"
    assert config.parser.gemini_api_key_path == Path("~/custom/gemini-key.txt").expanduser()
    assert config.parser.gemini_model == "gemini-2.0-flash"
    assert config.parser.gemini_timeout_ms == 12000
    assert config.parser.fallback_to_local is False
    assert config.tts.provider == "elevenlabs"
    assert config.tts.elevenlabs_api_key_path == Path("~/custom/elevenlabs-key.txt").expanduser()
    assert config.tts.elevenlabs_model == "eleven_turbo_v2_5"
    assert config.ui.port == 9000
    assert config.ui.auto_open_browser is False
