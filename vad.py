"""Local voice activity detection for hands-free turn taking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Event
from typing import Iterator


class VADEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VadTurnDetector:
    silence_threshold_ms: int = 800
    min_speech_duration_ms: int = 250
    in_speech: bool = False
    mic_level: float = 0.0

    def __post_init__(self) -> None:
        self._speech_ms = 0
        self._silence_ms = 0

    def process_frame(
        self,
        *,
        is_speech: bool,
        frame_duration_ms: int,
        mic_level: float,
    ) -> list[VADEvent]:
        self.mic_level = mic_level
        events: list[VADEvent] = []

        if is_speech:
            self._speech_ms += frame_duration_ms
            self._silence_ms = 0
            if not self.in_speech and self._speech_ms >= self.min_speech_duration_ms:
                self.in_speech = True
                events.append(VADEvent.SPEECH_START)
            return events

        if self.in_speech:
            self._silence_ms += frame_duration_ms
            if self._silence_ms >= self.silence_threshold_ms:
                self.in_speech = False
                self._speech_ms = 0
                self._silence_ms = 0
                events.append(VADEvent.SPEECH_END)
        else:
            self._speech_ms = 0
            self._silence_ms = 0
        return events


def detect_turn_events(
    *,
    silence_threshold_ms: int = 800,
    min_speech_duration_ms: int = 250,
    sample_rate: int = 16_000,
    block_ms: int = 32,
    stop_event: Event | None = None,
) -> Iterator[VADEvent]:
    """Yield speech start/end events from the default microphone.

    The app uses this generator only while the user is on turn. Audio stays in
    memory and is never written to disk or sent over the network.
    """

    import numpy as np
    import sounddevice as sd
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    detector = VadTurnDetector(
        silence_threshold_ms=silence_threshold_ms,
        min_speech_duration_ms=min_speech_duration_ms,
    )
    block_size = int(sample_rate * (block_ms / 1000))

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
    ) as stream:
        while stop_event is None or not stop_event.is_set():
            audio, _overflowed = stream.read(block_size)
            if stop_event is not None and stop_event.is_set():
                break
            mono = np.asarray(audio, dtype=np.float32).reshape(-1)
            mic_level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            with torch.no_grad():
                speech_probability = float(model(torch.from_numpy(mono), sample_rate).item())
            for event in detector.process_frame(
                is_speech=speech_probability >= 0.5,
                frame_duration_ms=block_ms,
                mic_level=mic_level,
            ):
                yield event
