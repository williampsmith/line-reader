import pytest

from models import PracticeQueueItem
from tts import (
    CHIRP_3_HD_VOICES,
    TTSAuthenticationError,
    TTSClient,
    TtsSynthesisError,
    default_voice_assignment,
    validate_hardcoded_voices,
)


def test_default_voice_assignment_excludes_user_and_round_robins():
    assignment = default_voice_assignment(["JOHN", "SARAH", "MARCUS"], "JOHN")

    assert assignment.user_character == "JOHN"
    assert "JOHN" not in assignment.voice_for_character
    assert assignment.voice_for_character["SARAH"] == CHIRP_3_HD_VOICES[0]
    assert assignment.voice_for_character["MARCUS"] == CHIRP_3_HD_VOICES[1]


def test_synthesize_caches_by_text_and_voice():
    calls = []

    def synthesizer(text, voice_id):
        calls.append((text, voice_id))
        return f"{voice_id}:{text}".encode()

    client = TTSClient(synthesizer=synthesizer)

    assert client.synthesize("Hello", "en-US-Chirp3-HD-Aoede") == b"en-US-Chirp3-HD-Aoede:Hello"
    assert client.synthesize("Hello", "en-US-Chirp3-HD-Aoede") == b"en-US-Chirp3-HD-Aoede:Hello"
    assert calls == [("Hello", "en-US-Chirp3-HD-Aoede")]


def test_synthesize_cache_varies_by_speaking_rate():
    calls = []

    def synthesizer(text, voice_id, speaking_rate):
        calls.append((text, voice_id, speaking_rate))
        return f"{voice_id}:{speaking_rate}:{text}".encode()

    client = TTSClient(synthesizer=synthesizer)

    assert client.synthesize("Hello", "voice", speaking_rate=0.85) == b"voice:0.85:Hello"
    assert client.synthesize("Hello", "voice", speaking_rate=0.85) == b"voice:0.85:Hello"
    assert client.synthesize("Hello", "voice", speaking_rate=1.15) == b"voice:1.15:Hello"
    assert calls == [
        ("Hello", "voice", 0.85),
        ("Hello", "voice", 1.15),
    ]


def test_prefetch_next_ai_line_skips_user_lines():
    calls = []

    def synthesizer(text, voice_id):
        calls.append((text, voice_id))
        return b"audio"

    queue = [
        PracticeQueueItem("ai", "SARAH", "First.", "voice-a", 0),
        PracticeQueueItem("user", "JOHN", "My line.", None, 1),
        PracticeQueueItem("ai", "MARCUS", "Second.", "voice-b", 2),
    ]
    client = TTSClient(synthesizer=synthesizer)

    future = client.prefetch_next_ai(queue, current_index=0, speaking_rate=1.2)

    assert future is not None
    assert future.result(timeout=1) == b"audio"
    assert calls == [("Second.", "voice-b")]


def test_prefetch_never_synthesizes_user_items():
    calls = []
    queue = [
        PracticeQueueItem("ai", "SARAH", "First.", "voice-a", 0),
        PracticeQueueItem("user", "JOHN", "My line.", None, 1),
    ]
    client = TTSClient(synthesizer=lambda text, voice_id: calls.append((text, voice_id)) or b"audio")

    assert client.prefetch_next_ai(queue, current_index=0) is None
    assert calls == []


def test_validate_hardcoded_voices_reports_missing_names():
    reported = {"en-US-Chirp3-HD-Aoede", "en-GB-Chirp3-HD-Zephyr"}

    missing = validate_hardcoded_voices(reported)

    assert "en-US-Chirp3-HD-Charon" in missing
    assert "en-US-Chirp3-HD-Aoede" not in missing


def test_auth_errors_are_not_retried_as_generic_failures():
    class Forbidden(Exception):
        code = 403

    client = TTSClient(synthesizer=lambda text, voice_id: (_ for _ in ()).throw(Forbidden()))

    with pytest.raises(TTSAuthenticationError, match="Google Cloud Text-to-Speech credentials"):
        client.synthesize("Hello", "en-US-Chirp3-HD-Aoede")


def test_network_errors_are_wrapped_for_retry_or_skip():
    client = TTSClient(synthesizer=lambda text, voice_id: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(TtsSynthesisError, match="boom"):
        client.synthesize("Hello", "en-US-Chirp3-HD-Aoede")
