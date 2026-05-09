"""Google Cloud Chirp 3 HD text-to-speech integration."""

from __future__ import annotations

import os
import inspect
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from models import PracticeQueueItem, VoiceAssignment


CHIRP_3_HD_VOICES = [
    "en-US-Chirp3-HD-Aoede",
    "en-US-Chirp3-HD-Charon",
    "en-US-Chirp3-HD-Fenrir",
    "en-US-Chirp3-HD-Kore",
    "en-US-Chirp3-HD-Leda",
    "en-US-Chirp3-HD-Orus",
    "en-US-Chirp3-HD-Puck",
    "en-US-Chirp3-HD-Zephyr",
    "en-GB-Chirp3-HD-Aoede",
    "en-GB-Chirp3-HD-Charon",
    "en-GB-Chirp3-HD-Fenrir",
    "en-GB-Chirp3-HD-Kore",
    "en-GB-Chirp3-HD-Leda",
    "en-GB-Chirp3-HD-Orus",
    "en-GB-Chirp3-HD-Puck",
    "en-GB-Chirp3-HD-Zephyr",
]

ELEVENLABS_VOICES = [
    ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
    ("Drew", "29vD33N1CtxCmqQRPOHJ"),
    ("Clyde", "2EiwWnXFnvU5JabPnvG"),
    ("Paul", "5Q0t7uMcjvnagumLfvZi"),
    ("Domi", "AZnzlk1XvdvUeBnXmlld"),
    ("Dave", "CYw3kZ02Hs0563khs1Fj"),
    ("Fin", "D38z5RcWu1voky8WS1ja"),
    ("Sarah", "EXAVITQu4vr4xnSDxMaL"),
    ("Antoni", "ErXwobaYiN019PkySvjV"),
    ("Thomas", "GBv7mTt0atIp3Br8iCZE"),
    ("Charlie", "IKne3meq5aSn9XLyUdCD"),
    ("George", "JBFqnCBsd6RMkjVDRZzb"),
]

TTS_PROVIDERS = ["google", "elevenlabs"]

DEFAULT_PREVIEW_TEXT = "Hello, this is a test."


class TtsSynthesisError(RuntimeError):
    """A retryable or skippable TTS synthesis failure."""


class TTSAuthenticationError(TtsSynthesisError):
    """Google Cloud credentials are missing or not authorized."""


Synthesizer = Callable[..., bytes]


def default_voice_assignment(
    characters: Iterable[str],
    user_character: str,
    voices: list[str] | None = None,
) -> VoiceAssignment:
    available = voices or CHIRP_3_HD_VOICES
    voice_for_character: dict[str, str] = {}
    voice_index = 0
    seen: set[str] = set()
    for character in characters:
        if character in seen:
            continue
        seen.add(character)
        if character == user_character:
            continue
        voice_for_character[character] = available[voice_index % len(available)]
        voice_index += 1
    return VoiceAssignment(user_character=user_character, voice_for_character=voice_for_character)


def validate_hardcoded_voices(reported_voice_names: Iterable[str]) -> list[str]:
    reported = set(reported_voice_names)
    return [voice for voice in CHIRP_3_HD_VOICES if voice not in reported]


class TTSClient:
    provider = "google"

    def __init__(
        self,
        synthesizer: Synthesizer | None = None,
        credentials_path: str | Path | None = None,
        executor: ThreadPoolExecutor | None = None,
        speaking_rate: float = 1.0,
    ) -> None:
        self._synthesizer = synthesizer
        self._credentials_path = Path(credentials_path).expanduser() if credentials_path else None
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self.speaking_rate = speaking_rate
        self._cache: dict[tuple[str, str, float], bytes] = {}

    @property
    def cache(self) -> dict[tuple[str, str, float], bytes]:
        return self._cache

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speaking_rate: float | None = None,
    ) -> bytes:
        rate = _normalize_speaking_rate(
            self.speaking_rate if speaking_rate is None else speaking_rate
        )
        key = (text, voice_id, rate)
        if key in self._cache:
            return self._cache[key]

        try:
            audio = self._call_synthesizer(text, voice_id, rate)
        except Exception as exc:  # noqa: BLE001 - SDK exceptions vary by transport.
            if _is_auth_error(exc):
                raise TTSAuthenticationError(
                    "Google Cloud Text-to-Speech credentials are missing or unauthorized. "
                    "Check ~/.config/audition-app/gcp-key.json and the README setup steps."
                ) from exc
            raise TtsSynthesisError(
                "Unable to synthesize this line. "
                f"Google Cloud reported: {exc}"
            ) from exc

        self._cache[key] = audio
        return audio

    def prefetch(
        self,
        item: PracticeQueueItem,
        speaking_rate: float | None = None,
    ) -> Future[bytes] | None:
        if item.role != "ai" or item.voice_id is None:
            return None
        return self._executor.submit(
            self.synthesize,
            item.text,
            item.voice_id,
            speaking_rate,
        )

    def prefetch_next_ai(
        self,
        queue: list[PracticeQueueItem],
        current_index: int,
        speaking_rate: float | None = None,
    ) -> Future[bytes] | None:
        for item in queue[current_index + 1 :]:
            if item.role == "ai":
                return self.prefetch(item, speaking_rate=speaking_rate)
        return None

    def preview(self, voice_id: str, sample_text: str = DEFAULT_PREVIEW_TEXT) -> bytes:
        return self.synthesize(sample_text, voice_id)

    def list_live_chirp_voices(self, language_code: str = "en-US") -> list[str]:
        client = self._google_client()
        response = client.list_voices(language_code=language_code)
        return [voice.name for voice in response.voices if "Chirp3-HD" in voice.name]

    def warn_if_catalog_changed(self) -> list[str]:
        live_names = set(self.list_live_chirp_voices("en-US")) | set(
            self.list_live_chirp_voices("en-GB")
        )
        missing = validate_hardcoded_voices(live_names)
        if missing:
            print(
                "Warning: Google Cloud Text-to-Speech did not report these configured "
                f"Chirp 3 HD voices: {', '.join(missing)}"
            )
        return missing

    def _call_synthesizer(
        self,
        text: str,
        voice_id: str,
        speaking_rate: float,
    ) -> bytes:
        if self._synthesizer is not None:
            if _accepts_speaking_rate(self._synthesizer):
                return self._synthesizer(text, voice_id, speaking_rate)
            return self._synthesizer(text, voice_id)
        return self._google_synthesize(text, voice_id, speaking_rate)

    def _google_synthesize(self, text: str, voice_id: str, speaking_rate: float) -> bytes:
        client = self._google_client()
        from google.cloud import texttospeech

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="-".join(voice_id.split("-")[:2]),
            name=voice_id,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return bytes(response.audio_content)

    def _google_client(self):
        if self._credentials_path:
            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS", str(self._credentials_path)
            )
        from google.cloud import texttospeech

        return texttospeech.TextToSpeechClient()


class ElevenLabsTTSClient:
    provider = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = "eleven_multilingual_v2",
        client: object | None = None,
        executor: ThreadPoolExecutor | None = None,
        speaking_rate: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.client = client
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self.speaking_rate = speaking_rate
        self._cache: dict[tuple[str, str, float, str], bytes] = {}

    @classmethod
    def from_api_key_file(
        cls,
        api_key_path: str | Path,
        *,
        model_id: str = "eleven_multilingual_v2",
    ) -> "ElevenLabsTTSClient":
        path = Path(api_key_path).expanduser()
        if not path.exists():
            raise TTSAuthenticationError(f"ElevenLabs API key file not found at {path}.")
        api_key = path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise TTSAuthenticationError(f"ElevenLabs API key file at {path} is empty.")
        return cls(api_key=api_key, model_id=model_id)

    @property
    def cache(self) -> dict[tuple[str, str, float, str], bytes]:
        return self._cache

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speaking_rate: float | None = None,
    ) -> bytes:
        rate = _normalize_speaking_rate(
            self.speaking_rate if speaking_rate is None else speaking_rate
        )
        key = (text, voice_id, rate, self.model_id)
        if key in self._cache:
            return self._cache[key]

        try:
            audio = self._elevenlabs_synthesize(text, voice_id)
        except Exception as exc:  # noqa: BLE001
            if _is_auth_error(exc):
                raise TTSAuthenticationError(
                    "ElevenLabs credentials are missing or unauthorized. "
                    "Check ~/.config/audition-app/elevenlabs-api-key.txt."
                ) from exc
            raise TtsSynthesisError(
                "Unable to synthesize this line. "
                f"ElevenLabs reported: {exc}"
            ) from exc

        self._cache[key] = audio
        return audio

    def prefetch(
        self,
        item: PracticeQueueItem,
        speaking_rate: float | None = None,
    ) -> Future[bytes] | None:
        if item.role != "ai" or item.voice_id is None:
            return None
        return self._executor.submit(
            self.synthesize,
            item.text,
            item.voice_id,
            speaking_rate,
        )

    def prefetch_next_ai(
        self,
        queue: list[PracticeQueueItem],
        current_index: int,
        speaking_rate: float | None = None,
    ) -> Future[bytes] | None:
        for item in queue[current_index + 1 :]:
            if item.role == "ai":
                return self.prefetch(item, speaking_rate=speaking_rate)
        return None

    def preview(self, voice_id: str, sample_text: str = DEFAULT_PREVIEW_TEXT) -> bytes:
        return self.synthesize(sample_text, voice_id)

    def _elevenlabs_synthesize(self, text: str, voice_id: str) -> bytes:
        client = self._elevenlabs_client()
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        if isinstance(audio, bytes):
            return audio
        return b"".join(audio)

    def _elevenlabs_client(self):
        if self.client is not None:
            return self.client
        from elevenlabs import ElevenLabs

        self.client = ElevenLabs(api_key=self.api_key)
        return self.client


def create_tts_client(config, provider: str | None = None):
    selected_provider = (provider or config.tts.provider).lower()
    if selected_provider == "elevenlabs":
        return ElevenLabsTTSClient.from_api_key_file(
            config.tts.elevenlabs_api_key_path,
            model_id=config.tts.elevenlabs_model,
        )
    return TTSClient(credentials_path=config.gcp.credentials_path)


def _is_auth_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if callable(code):
        code = code()
    status_code = getattr(exc, "status_code", None)
    if callable(status_code):
        status_code = status_code()
    candidates = {code, status_code}
    names = {str(candidate).upper() for candidate in candidates if candidate is not None}
    return 401 in candidates or 403 in candidates or any(
        token in names for token in {"UNAUTHENTICATED", "PERMISSION_DENIED", "401", "403"}
    )


def _normalize_speaking_rate(rate: float) -> float:
    return round(max(0.5, min(1.5, float(rate))), 2)


def _accepts_speaking_rate(synthesizer: Synthesizer) -> bool:
    signature = inspect.signature(synthesizer)
    parameters = signature.parameters.values()
    return any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters) or (
        len(signature.parameters) >= 3
    )
