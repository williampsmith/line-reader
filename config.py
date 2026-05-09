"""Configuration loading for local audition rehearsal sessions."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("~/.config/audition-app/config.toml").expanduser()
DEFAULT_CREDENTIALS_PATH = Path("~/.config/audition-app/gcp-key.json").expanduser()
DEFAULT_GEMINI_API_KEY_PATH = Path(
    "~/.config/audition-app/gemini-api-key.txt"
).expanduser()


@dataclass(frozen=True)
class GcpConfig:
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH


@dataclass(frozen=True)
class VadConfig:
    silence_threshold_ms: int = 800
    min_speech_duration_ms: int = 250
    sample_rate: int = 16_000


@dataclass(frozen=True)
class UiConfig:
    port: int = 7860
    auto_open_browser: bool = True


@dataclass(frozen=True)
class ParserConfig:
    mode: str = "gemini"
    gemini_api_key_path: Path = DEFAULT_GEMINI_API_KEY_PATH
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_timeout_ms: int = 45_000
    gemini_use_image: bool = True
    fallback_to_local: bool = False


@dataclass(frozen=True)
class AppConfig:
    gcp: GcpConfig = field(default_factory=GcpConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load app config from TOML, falling back to safe local defaults."""

    config_path = _expand_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as config_file:
        raw: dict[str, Any] = tomllib.load(config_file)

    gcp_raw = raw.get("gcp", {})
    vad_raw = raw.get("vad", {})
    parser_raw = raw.get("parser", {})
    ui_raw = raw.get("ui", {})

    return AppConfig(
        gcp=GcpConfig(
            credentials_path=_expand_path(
                gcp_raw.get("credentials_path", DEFAULT_CREDENTIALS_PATH)
            )
        ),
        vad=VadConfig(
            silence_threshold_ms=int(vad_raw.get("silence_threshold_ms", 800)),
            min_speech_duration_ms=int(vad_raw.get("min_speech_duration_ms", 250)),
            sample_rate=int(vad_raw.get("sample_rate", 16_000)),
        ),
        parser=ParserConfig(
            mode=str(parser_raw.get("mode", "gemini")),
            gemini_api_key_path=_expand_path(
                parser_raw.get("gemini_api_key_path", DEFAULT_GEMINI_API_KEY_PATH)
            ),
            gemini_model=str(parser_raw.get("gemini_model", "gemini-2.5-flash-lite")),
            gemini_timeout_ms=int(parser_raw.get("gemini_timeout_ms", 45_000)),
            gemini_use_image=bool(parser_raw.get("gemini_use_image", True)),
            fallback_to_local=bool(parser_raw.get("fallback_to_local", False)),
        ),
        ui=UiConfig(
            port=int(ui_raw.get("port", 7860)),
            auto_open_browser=bool(ui_raw.get("auto_open_browser", True)),
        ),
    )
