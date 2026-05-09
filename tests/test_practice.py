import pytest

from models import LineType, ParsedLine, Scene, Script, VoiceAssignment
from practice import PracticeSession, SessionState, build_practice_queue
from tts import TtsSynthesisError


def line(line_type, text, character=None):
    return ParsedLine(
        type=line_type,
        text=text,
        page=1,
        bbox=(0.0, 0.0, 1.0, 1.0),
        character=character,
    )


def sample_script():
    lines = [
        line(LineType.SCENE_HEADING, "INT. KITCHEN - DAY"),
        line(LineType.ACTION, "The room is quiet."),
        line(LineType.CHARACTER, "SARAH", "SARAH"),
        line(LineType.DIALOGUE, "Are you ready?", "SARAH"),
        line(LineType.PARENTHETICAL, "(beat)", "JOHN"),
        line(LineType.CHARACTER, "JOHN", "JOHN"),
        line(LineType.DIALOGUE, "I was born ready.", "JOHN"),
        line(LineType.CHARACTER, "MARCUS", "MARCUS"),
        line(LineType.DIALOGUE, "Then let's move.", "MARCUS"),
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
                characters={"SARAH", "JOHN", "MARCUS"},
            )
        ],
        characters={"SARAH", "JOHN", "MARCUS"},
    )


def assignment():
    return VoiceAssignment(
        user_character="JOHN",
        voice_for_character={"SARAH": "voice-sarah", "MARCUS": "voice-marcus"},
    )


class FakeTTS:
    def __init__(self):
        self.synthesized = []
        self.prefetched = []
        self.fail = False

    def synthesize(self, text, voice_id, speaking_rate=1.0):
        self.synthesized.append((text, voice_id, speaking_rate))
        if self.fail:
            raise TtsSynthesisError("network")
        return f"{voice_id}:{text}".encode()

    def prefetch_next_ai(self, queue, current_index, speaking_rate=1.0):
        self.prefetched.append((current_index, speaking_rate))
        return None


def test_build_practice_queue_skips_non_spoken_lines_and_user_has_no_voice():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)

    assert [(item.role, item.character, item.text, item.voice_id) for item in queue] == [
        ("ai", "SARAH", "Are you ready?", "voice-sarah"),
        ("user", "JOHN", "I was born ready.", None),
        ("ai", "MARCUS", "Then let's move.", "voice-marcus"),
    ]


def test_build_practice_queue_rejects_voice_for_user_character():
    bad_assignment = VoiceAssignment(
        user_character="JOHN",
        voice_for_character={"JOHN": "voice-john", "SARAH": "voice-sarah"},
    )

    with pytest.raises(ValueError, match="user character"):
        build_practice_queue(sample_script(), bad_assignment, start_scene_index=0)


def test_user_turn_is_silent_until_vad_speech_end():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)[1:]
    tts = FakeTTS()
    played = []
    session = PracticeSession(queue, tts_client=tts, audio_player=played.append)

    session.start()

    assert session.state == SessionState.USER_TURN
    assert tts.synthesized == []
    assert played == []

    session.handle_vad_event("speech_end")

    assert tts.synthesized == [("Then let's move.", "voice-marcus", 1.0)]
    assert played == [b"voice-marcus:Then let's move."]
    assert session.state == SessionState.AI_TURN

    session.handle_audio_complete()

    assert session.state == SessionState.DONE


def test_ai_turn_synthesizes_prefetches_and_waits_for_playback_complete():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)
    tts = FakeTTS()
    played = []
    session = PracticeSession(queue, tts_client=tts, audio_player=played.append)

    session.start()

    assert tts.synthesized == [("Are you ready?", "voice-sarah", 1.0)]
    assert tts.prefetched == [(0, 1.0)]
    assert played == [b"voice-sarah:Are you ready?"]
    assert session.state == SessionState.AI_TURN
    assert session.current_item.character == "SARAH"

    session.handle_audio_complete()

    assert session.state == SessionState.USER_TURN
    assert session.current_item.character == "JOHN"


def test_tts_failure_pauses_session_for_retry_or_skip():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)
    tts = FakeTTS()
    tts.fail = True
    session = PracticeSession(queue, tts_client=tts, audio_player=lambda audio: None)

    session.start()

    assert session.state == SessionState.ERROR
    assert session.paused is True
    assert "network" in session.error_message

    session.skip_forward()

    assert session.state == SessionState.USER_TURN
    assert session.paused is False


def test_manual_controls_pause_resume_skip_back_and_restart():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)
    tts = FakeTTS()
    played = []
    session = PracticeSession(queue, tts_client=tts, audio_player=played.append)

    session.start()
    session.pause()
    session.handle_audio_complete()
    assert session.state == SessionState.PAUSED
    assert session.index == 0

    session.resume()
    session.handle_audio_complete()
    session.handle_vad_event("speech_end")
    assert session.state == SessionState.AI_TURN
    assert session.index == 2

    session.skip_forward()
    assert session.state == SessionState.DONE

    session.skip_back()
    assert session.state == SessionState.AI_TURN
    assert tts.synthesized[-1] == ("Then let's move.", "voice-marcus", 1.0)
    session.handle_audio_complete()
    assert session.state == SessionState.DONE

    session.restart()
    assert session.index == 0
    assert session.state == SessionState.AI_TURN


def test_ai_turn_uses_adjustable_speaking_rate_for_synthesis_and_prefetch():
    queue = build_practice_queue(sample_script(), assignment(), start_scene_index=0)
    tts = FakeTTS()
    session = PracticeSession(
        queue,
        tts_client=tts,
        audio_player=lambda audio: None,
        speaking_rate=0.85,
    )

    session.start()

    assert tts.synthesized == [("Are you ready?", "voice-sarah", 0.85)]
    assert tts.prefetched == [(0, 0.85)]

    session.set_speaking_rate(1.2)
    session.handle_audio_complete()
    session.handle_vad_event("speech_end")

    assert tts.synthesized[-1] == ("Then let's move.", "voice-marcus", 1.2)
