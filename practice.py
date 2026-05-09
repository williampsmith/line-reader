"""Practice queue and turn-taking state machine."""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Protocol

from models import LineType, PracticeQueueItem, Script, VoiceAssignment
from tts import TtsSynthesisError


class SessionState(str, Enum):
    INIT = "init"
    AI_TURN = "ai_turn"
    USER_TURN = "user_turn"
    PAUSED = "paused"
    ERROR = "error"
    DONE = "done"


class TTSLike(Protocol):
    def synthesize(
        self,
        text: str,
        voice_id: str,
        speaking_rate: float = 1.0,
    ) -> bytes:
        ...

    def prefetch_next_ai(
        self,
        queue: list[PracticeQueueItem],
        current_index: int,
        speaking_rate: float = 1.0,
    ):
        ...


AudioPlayer = Callable[[bytes], None]


def build_practice_queue(
    script: Script,
    assignment: VoiceAssignment,
    start_scene_index: int,
) -> list[PracticeQueueItem]:
    if assignment.user_character in assignment.voice_for_character:
        raise ValueError("The user character must not be assigned an AI voice.")

    if not script.scenes:
        return []
    start_scene = script.scenes[start_scene_index]
    lines = script.lines[start_scene.start_line :]

    queue: list[PracticeQueueItem] = []
    for source_index, line in enumerate(lines, start=start_scene.start_line):
        if line.type != LineType.DIALOGUE or not line.character:
            continue
        role = "user" if line.character == assignment.user_character else "ai"
        voice_id = None if role == "user" else assignment.voice_for_character.get(line.character)
        if role == "ai" and voice_id is None:
            raise ValueError(f"No voice assigned for AI character {line.character!r}.")
        queue.append(
            PracticeQueueItem(
                role=role,
                character=line.character,
                text=_strip_parentheticals(line.text),
                voice_id=voice_id,
                source_line_index=source_index,
            )
        )
    return queue


class PracticeSession:
    def __init__(
        self,
        queue: list[PracticeQueueItem],
        tts_client: TTSLike,
        audio_player: AudioPlayer,
        speaking_rate: float = 1.0,
    ) -> None:
        self.queue = queue
        self.tts_client = tts_client
        self.audio_player = audio_player
        self.speaking_rate = speaking_rate
        self.index = 0
        self.state = SessionState.INIT
        self.paused = False
        self.error_message = ""
        self._state_before_pause: SessionState | None = None

    @property
    def current_item(self) -> PracticeQueueItem | None:
        if 0 <= self.index < len(self.queue):
            return self.queue[self.index]
        return None

    @property
    def progress(self) -> tuple[int, int]:
        return min(self.index + 1, len(self.queue)), len(self.queue)

    def start(self) -> None:
        self.index = 0
        self.paused = False
        self.error_message = ""
        self._enter_current_item()

    def pause(self) -> None:
        if self.state in {SessionState.DONE, SessionState.ERROR}:
            return
        self._state_before_pause = self.state
        self.state = SessionState.PAUSED
        self.paused = True

    def resume(self) -> None:
        if self.state != SessionState.PAUSED:
            return
        self.paused = False
        self.state = self._state_before_pause or SessionState.INIT
        if self.state == SessionState.INIT:
            self._enter_current_item()

    def handle_audio_complete(self) -> None:
        if self.paused or self.state != SessionState.AI_TURN:
            return
        self.index += 1
        self._enter_current_item()

    def handle_vad_event(self, event: str) -> None:
        if self.paused or self.state != SessionState.USER_TURN:
            return
        if event.lower() in {"speech_end", "end", "vad_speech_end"}:
            self.index += 1
            self._enter_current_item()

    def manual_done(self) -> None:
        self.handle_vad_event("speech_end")

    def set_speaking_rate(self, speaking_rate: float) -> None:
        self.speaking_rate = speaking_rate

    def skip_forward(self) -> None:
        self.paused = False
        self.error_message = ""
        if self.index < len(self.queue):
            self.index += 1
        self._enter_current_item()

    def skip_back(self) -> None:
        self.paused = False
        self.error_message = ""
        if self.index >= len(self.queue):
            self.index = max(0, len(self.queue) - 1)
        else:
            self.index = max(0, self.index - 1)
        self._enter_current_item()

    def restart(self) -> None:
        self.start()

    def retry(self) -> None:
        self.paused = False
        self.error_message = ""
        self._enter_current_item()

    def _enter_current_item(self) -> None:
        if self.paused:
            return
        if self.index < len(self.queue):
            item = self.queue[self.index]
            if item.role == "user":
                self.state = SessionState.USER_TURN
                return
            self._play_ai_item(item)
            return
        self.state = SessionState.DONE

    def _play_ai_item(self, item: PracticeQueueItem) -> None:
        if item.voice_id is None:
            raise ValueError("AI queue item is missing a voice.")
        self.state = SessionState.AI_TURN
        try:
            audio = self.tts_client.synthesize(
                item.text,
                item.voice_id,
                speaking_rate=self.speaking_rate,
            )
            self.tts_client.prefetch_next_ai(
                self.queue,
                self.index,
                speaking_rate=self.speaking_rate,
            )
            self.audio_player(audio)
        except TtsSynthesisError as exc:
            self.error_message = str(exc)
            self.paused = True
            self.state = SessionState.ERROR


def _strip_parentheticals(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
