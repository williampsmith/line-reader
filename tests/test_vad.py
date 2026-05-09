from vad import VADEvent, VadTurnDetector


def test_detector_emits_start_after_minimum_speech_and_end_after_silence():
    detector = VadTurnDetector(silence_threshold_ms=800, min_speech_duration_ms=250)

    assert detector.process_frame(is_speech=True, frame_duration_ms=100, mic_level=0.2) == []
    assert detector.process_frame(is_speech=True, frame_duration_ms=150, mic_level=0.3) == [
        VADEvent.SPEECH_START
    ]
    assert detector.process_frame(is_speech=False, frame_duration_ms=500, mic_level=0.0) == []
    assert detector.process_frame(is_speech=False, frame_duration_ms=300, mic_level=0.0) == [
        VADEvent.SPEECH_END
    ]


def test_detector_ignores_short_speech_bursts():
    detector = VadTurnDetector(silence_threshold_ms=800, min_speech_duration_ms=250)

    assert detector.process_frame(is_speech=True, frame_duration_ms=100, mic_level=0.8) == []
    assert detector.process_frame(is_speech=False, frame_duration_ms=900, mic_level=0.0) == []
    assert detector.in_speech is False
    assert detector.mic_level == 0.0
