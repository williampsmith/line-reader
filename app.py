"""Gradio entry point for the audition rehearsal app."""

from __future__ import annotations

import tempfile
import threading
import time
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

from ai_parser import GeminiParseError, GeminiScriptParser, combine_page_scripts
from config import AppConfig, load_config
from models import LineType, PracticeQueueItem, Script, VoiceAssignment
from parser import (
    ParsingError,
    apply_character_renames,
    classify_lines,
    ocr_pages,
    rasterize,
    reclassify_line,
    validate_parse_quality,
)
from practice import PracticeSession, SessionState, build_practice_queue
from tts import CHIRP_3_HD_VOICES, TTSClient, default_voice_assignment
from vad import VADEvent, detect_turn_events


PRIVACY_NOTICE = (
    "PDFs and microphone audio are processed locally on your Mac. "
    "When AI parsing is enabled, script page images and OCR text are sent to Gemini "
    "for parsing. AI character dialogue text is sent to Google Cloud for voice synthesis."
)


PROMPTER_CSS = """
.current-line {
  border: 2px solid var(--color-accent, #f59e0b);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
  background: rgba(245, 158, 11, 0.14);
}

.upcoming-line {
  border-left: 4px solid var(--border-color-primary, #555);
  padding: 12px 18px;
  margin-bottom: 10px;
  opacity: 0.78;
}

.line-meta {
  font-size: 0.95rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.78;
  margin-bottom: 0.45rem;
}

.line-speaker {
  font-size: 1.15rem;
  margin-bottom: 0.45rem;
}

.line-text {
  font-size: clamp(1.55rem, 2.8vw, 2.35rem);
  line-height: 1.22;
  font-weight: 650;
}
"""


@dataclass
class PracticeUiState:
    config: AppConfig = field(default_factory=load_config)
    script: Script | None = None
    assignment: VoiceAssignment | None = None
    tts_client: TTSClient | None = None
    session: PracticeSession | None = None
    queue_scene_index: int = 0
    pending_audio: list[bytes] = field(default_factory=list)
    dialogue_pacing: float = 1.0
    vad_thread: threading.Thread | None = None
    vad_stop: threading.Event = field(default_factory=threading.Event)
    vad_active: bool = False
    mic_level: float = 0.0
    last_status: str = "Upload a PDF to begin."

    def __deepcopy__(self, memo):
        # This is a single-user local app. Gradio checks that initial state can
        # be deep-copied, but runtime state intentionally includes threads and
        # clients that should be shared across callbacks for this session.
        return self


def _state() -> PracticeUiState:
    return PracticeUiState()


def _ensure_state(state: PracticeUiState | None) -> PracticeUiState:
    return state if isinstance(state, PracticeUiState) else _state()


def _nav_target(tab_id: str) -> str:
    return f"{tab_id}:{time.monotonic_ns()}"


def _tab_switch_js() -> str:
    return """
(targetValue) => {
  const raw = String(targetValue || "");
  const target = raw.split(":")[0];
  if (!target) return raw;

  const labels = {
    upload: "1. Upload",
    review: "2. Review parsed script",
    cast: "3. Cast voices",
    scenes: "4. Scene picker",
    practice: "5. Practice",
  };
  const expected = labels[target];

  setTimeout(() => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"], button'));
    const tab = tabs.find((element) => {
      const text = (element.textContent || "").trim();
      return text === expected || text.endsWith(expected);
    });
    if (tab) {
      tab.click();
      tab.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }, 0);

  return raw;
}
"""


def process_pdf(file: Any, state: PracticeUiState):
    state = _ensure_state(state)
    if file is None:
        yield state, [], [], "", "Choose a PDF first.", _nav_target("upload")
        return

    try:
        yield state, [], [], "", "Rasterizing pages...", ""
        pdf_path = Path(file.name if hasattr(file, "name") else file)
        pdf_bytes = pdf_path.read_bytes()
        images = rasterize(pdf_bytes)
        yield state, [], [], "", (
            f"Running OCR on {len(images)} page{'s' if len(images) != 1 else ''}..."
        ), ""
        ocr = ocr_pages(images)
        if state.config.parser.mode == "gemini":
            try:
                parser = _gemini_parser_from_state(state)
                page_scripts = []
                total_pages = len(images)
                for page_index, image in enumerate(images, start=1):
                    page_ocr = ocr[page_index - 1] if page_index - 1 < len(ocr) else []
                    yield (
                        state,
                        [],
                        [],
                        "",
                        f"Parsing screenplay with Gemini (page {page_index} of {total_pages})...",
                        "",
                    )
                    page_scripts.append(
                        _parse_gemini_page(parser, image, page_ocr, page_index)
                    )
                script = combine_page_scripts(page_scripts)
            except GeminiParseError as exc:
                if not state.config.parser.fallback_to_local:
                    raise
                yield (
                    state,
                    [],
                    [],
                    "",
                    f"Gemini parsing failed: {exc}",
                    "",
                )
                yield state, [], [], "", "Falling back to local parser...", ""
                script = classify_lines(ocr)
        else:
            yield state, [], [], "", "Parsing screenplay locally...", ""
            script = classify_lines(ocr)
        validate_parse_quality(script)
    except ParsingError as exc:
        state.last_status = str(exc)
        yield state, [], [], "", str(exc), _nav_target("upload")
        return
    except Exception as exc:  # noqa: BLE001 - OCR/PDF libraries expose varied errors.
        state.last_status = f"Could not process PDF: {exc}"
        yield state, [], [], "", state.last_status, _nav_target("upload")
        return

    state.script = script
    state.last_status = (
        f"Parsed {len(script.lines)} lines, {len(script.scenes)} scenes, "
        f"and {len(script.characters)} characters."
    )
    yield (
        state,
        _character_table(script),
        _scene_table(script),
        _script_markdown(script),
        state.last_status,
        _nav_target("review"),
    )


def _gemini_parser_from_state(state: PracticeUiState) -> GeminiScriptParser:
    return GeminiScriptParser.from_api_key_file(
        state.config.parser.gemini_api_key_path,
        model=state.config.parser.gemini_model,
        timeout_ms=state.config.parser.gemini_timeout_ms,
    )


def _parse_gemini_page(
    parser: GeminiScriptParser,
    image: object,
    page_ocr: list[Any],
    page_number: int,
) -> Script:
    return parser.parse_page(image, page_ocr, page_number)


def _parse_with_gemini(
    state: PracticeUiState,
    images: Iterable[object],
    ocr: list[list[Any]],
) -> Script:
    parser = _gemini_parser_from_state(state)
    return parser.parse(images, ocr)


def apply_review_changes(
    state: PracticeUiState,
    rename_map_text: str,
    line_number: float | int | None,
    new_type: str,
):
    state = _ensure_state(state)
    if state.script is None:
        return state, [], [], "", "Upload and parse a script first."

    script = state.script
    renames = _parse_renames(rename_map_text)
    if renames:
        script = apply_character_renames(script, renames)

    if line_number:
        index = int(line_number) - 1
        if 0 <= index < len(script.lines):
            line_type = None if new_type == "[delete]" else LineType(new_type)
            script = reclassify_line(script, index, line_type)

    state.script = script
    state.last_status = "Review edits applied."
    return (
        state,
        _character_table(script),
        _scene_table(script),
        _script_markdown(script),
        state.last_status,
    )


def prepare_casting(state: PracticeUiState):
    state = _ensure_state(state)
    if state.script is None:
        return (
            gr.update(choices=[], value=None),
            [],
            "Parse and review a script first.",
            _nav_target("review"),
        )
    characters = sorted(state.script.characters)
    return gr.update(choices=characters, value=None), [], "Choose your role.", _nav_target("cast")


def update_default_voices(state: PracticeUiState, user_character: str):
    state = _ensure_state(state)
    if state.script is None or not user_character:
        return []
    return _voice_table(default_voice_assignment(sorted(state.script.characters), user_character))


def preview_voice(state: PracticeUiState, voice_id: str):
    state = _ensure_state(state)
    if not voice_id:
        return None, "Choose a voice to preview."
    client = _tts_client(state)
    try:
        audio = client.preview(voice_id)
    except Exception as exc:  # noqa: BLE001 - shown inline for user action.
        return None, str(exc)
    return _audio_file(audio), f"Previewed {voice_id}."


def commit_casting(state: PracticeUiState, user_character: str, voice_rows):
    state = _ensure_state(state)
    if state.script is None:
        return state, gr.update(choices=[], value=None), "Parse a script first.", _nav_target("review")
    if not user_character:
        return (
            state,
            gr.update(choices=[], value=None),
            "Choose the character you are playing.",
            _nav_target("cast"),
        )

    voice_for_character = {}
    for row in _iter_voice_rows(voice_rows):
        if len(row) < 2:
            continue
        character, voice_id = str(row[0]).strip(), str(row[1]).strip()
        if character and character != user_character and voice_id:
            voice_for_character[character] = voice_id

    missing = sorted((state.script.characters - {user_character}) - set(voice_for_character))
    if missing:
        state.assignment = None
        return (
            state,
            gr.update(choices=[], value=None),
            f"Choose a voice for {', '.join(missing)} before continuing.",
            _nav_target("cast"),
        )

    state.assignment = VoiceAssignment(
        user_character=user_character,
        voice_for_character=voice_for_character,
    )
    choices = _scene_choices(state.script)
    return (
        state,
        gr.update(choices=choices, value=choices[0] if choices else None),
        "Casting saved. Pick a scene to start from.",
        _nav_target("scenes"),
    )


def start_practice(
    state: PracticeUiState,
    scene_choice: str,
    silence_threshold_ms: int,
    dialogue_pacing: float = 1.0,
):
    state = _ensure_state(state)
    if state.script is None or state.assignment is None:
        return (
            state,
            "",
            "",
            gr.update(),
            "Finish review and casting first.",
            0,
            _nav_target("cast"),
        )
    scene_index = _scene_index_from_choice(scene_choice)
    state.queue_scene_index = scene_index
    try:
        queue = build_practice_queue(state.script, state.assignment, scene_index)
    except ValueError as exc:
        return state, "", "", gr.update(), str(exc), 0, _nav_target("scenes")
    state.pending_audio.clear()
    state.dialogue_pacing = float(dialogue_pacing)
    tts_client = _tts_client(state)
    tts_client.speaking_rate = state.dialogue_pacing
    state.session = PracticeSession(
        queue=queue,
        tts_client=tts_client,
        audio_player=state.pending_audio.append,
        speaking_rate=state.dialogue_pacing,
    )
    state.config = AppConfig(
        gcp=state.config.gcp,
        vad=type(state.config.vad)(
            silence_threshold_ms=int(silence_threshold_ms),
            min_speech_duration_ms=state.config.vad.min_speech_duration_ms,
            sample_rate=state.config.vad.sample_rate,
        ),
        parser=state.config.parser,
        ui=state.config.ui,
    )
    state.session.start()
    _start_vad_if_needed(state)
    return (*_practice_outputs(state), _nav_target("practice"))


def update_dialogue_pacing(state: PracticeUiState, dialogue_pacing: float):
    state = _ensure_state(state)
    state.dialogue_pacing = float(dialogue_pacing)
    if state.tts_client is not None:
        state.tts_client.speaking_rate = state.dialogue_pacing
    if state.session is not None:
        state.session.set_speaking_rate(state.dialogue_pacing)
    return state, f"AI dialogue pace: {state.dialogue_pacing:.2f}x"


def poll_practice(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def audio_complete(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        state.session.handle_audio_complete()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def pause_practice(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        state.session.pause()
        _stop_vad(state)
    return _practice_outputs(state)


def resume_practice(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        state.session.resume()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def skip_forward(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        _stop_vad(state)
        state.session.skip_forward()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def skip_back(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        _stop_vad(state)
        state.session.skip_back()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def manual_done(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        _stop_vad(state)
        state.session.manual_done()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def restart_practice(state: PracticeUiState):
    state = _ensure_state(state)
    if state.session:
        _stop_vad(state)
        state.session.restart()
        _start_vad_if_needed(state)
    return _practice_outputs(state)


def _tts_client(state: PracticeUiState) -> TTSClient:
    if state.tts_client is None:
        state.tts_client = TTSClient(credentials_path=state.config.gcp.credentials_path)
    return state.tts_client


def _start_vad_if_needed(state: PracticeUiState) -> None:
    session = state.session
    if session is None or session.state != SessionState.USER_TURN:
        return
    if state.pending_audio:
        return
    if state.vad_thread and state.vad_thread.is_alive():
        return
    state.vad_stop = threading.Event()
    state.vad_active = True

    def worker() -> None:
        try:
            for event in detect_turn_events(
                silence_threshold_ms=state.config.vad.silence_threshold_ms,
                min_speech_duration_ms=state.config.vad.min_speech_duration_ms,
                sample_rate=state.config.vad.sample_rate,
                stop_event=state.vad_stop,
            ):
                if state.vad_stop.is_set():
                    return
                if event == VADEvent.SPEECH_END and state.session:
                    state.session.handle_vad_event("speech_end")
                    return
        finally:
            state.vad_active = False

    state.vad_thread = threading.Thread(target=worker, daemon=True)
    state.vad_thread.start()


def _stop_vad(state: PracticeUiState) -> None:
    state.vad_stop.set()
    state.vad_active = False
    if state.vad_thread and state.vad_thread.is_alive():
        state.vad_thread.join(timeout=0.2)


def _practice_outputs(state: PracticeUiState):
    session = state.session
    if session is None:
        return (
            state,
            "Start a scene to practice.",
            "",
            gr.update(),
            state.last_status,
            state.mic_level,
        )
    item = session.current_item
    line_md = _practice_header_markdown(session)
    scene_md = _practice_scene_markdown(session)
    audio = _audio_file(state.pending_audio.pop(0)) if state.pending_audio else gr.update()
    status = _practice_status(state)
    return state, line_md, scene_md, audio, status, state.mic_level


def _practice_status(state: PracticeUiState) -> str:
    session = state.session
    if session is None:
        return state.last_status
    if session.state == SessionState.ERROR:
        return f"TTS error: {session.error_message}"
    if session.state == SessionState.DONE:
        return "Scene complete."
    current, total = session.progress
    vad = "VAD listening" if state.vad_active else "VAD idle"
    return f"{session.state.value} - line {current} of {total} - {vad}"


def _practice_header_markdown(session: PracticeSession) -> str:
    if session.current_item is None:
        return "## Scene complete"
    current, total = session.progress
    state_label = session.state.value.replace("_", " ")
    return f"### Line {current} of {total}\n\n`{state_label}`"


def _practice_scene_markdown(session: PracticeSession) -> str:
    if not session.queue:
        return ""

    blocks = []
    start = min(session.index, len(session.queue))
    visible_items = session.queue[start : start + 3]
    for offset, item in enumerate(visible_items):
        absolute_index = start + offset
        is_current = offset == 0 and session.state != SessionState.DONE
        blocks.append(_practice_prompt_block(item, absolute_index, is_current))
    return "\n\n".join(blocks)


def _practice_prompt_block(
    item: PracticeQueueItem,
    index: int,
    is_current: bool,
) -> str:
    speaker_label = "YOUR LINE" if item.role == "user" else "AI LINE"
    current_label = "CURRENTLY SPEAKING" if is_current else f"NEXT {index + 1}"
    css_class = "current-line" if is_current else "upcoming-line"
    character = html.escape(item.character)
    text = html.escape(item.text)
    return (
        f'<div class="{css_class}">\n\n'
        f'<div class="line-meta">{current_label} - {speaker_label}</div>\n'
        f'<div class="line-speaker"><strong>{character}</strong></div>\n'
        f'<div class="line-text">{text}</div>\n'
        "</div>"
    )


def _character_table(script: Script) -> list[list[Any]]:
    rows = []
    for character in sorted(script.characters):
        count = sum(
            1
            for line in script.lines
            if line.type == LineType.DIALOGUE and line.character == character
        )
        rows.append([character, count])
    return rows


def _scene_table(script: Script) -> list[list[Any]]:
    rows = []
    for scene in script.scenes:
        page = script.lines[scene.start_line].page if script.lines else ""
        rows.append(
            [
                scene.index + 1,
                scene.heading,
                page,
                ", ".join(sorted(scene.characters)),
            ]
        )
    return rows


def _script_markdown(script: Script) -> str:
    lines = []
    for index, line in enumerate(script.lines, start=1):
        character = f" - {line.character}" if line.character else ""
        lines.append(
            f"`{index:03d}` **{line.type.value}{character}**: {line.text}"
        )
    return "\n\n".join(lines)


def _voice_table(assignment: VoiceAssignment) -> list[list[str]]:
    return [
        [character, voice_id]
        for character, voice_id in assignment.voice_for_character.items()
    ]


def _iter_voice_rows(voice_rows) -> list[list[Any]]:
    if voice_rows is None:
        return []
    if hasattr(voice_rows, "values"):
        return voice_rows.values.tolist()
    return list(voice_rows)


def _scene_choices(script: Script) -> list[str]:
    return [
        f"{scene.index + 1}. {scene.heading} (p.{script.lines[scene.start_line].page}) - "
        f"{', '.join(sorted(scene.characters))}"
        for scene in script.scenes
    ]


def _scene_index_from_choice(choice: str) -> int:
    try:
        return int(choice.split(".", 1)[0]) - 1
    except Exception:
        return 0


def _parse_renames(rename_map_text: str) -> dict[str, str]:
    renames = {}
    for raw_line in (rename_map_text or "").splitlines():
        if "=" not in raw_line:
            continue
        source, target = raw_line.split("=", 1)
        if source.strip() and target.strip():
            renames[source.strip()] = target.strip()
    return renames


def _audio_file(audio: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        temp.write(audio)
        return temp.name


def build_app():
    with gr.Blocks(title="Audition Rehearsal") as demo:
        state = gr.State(None)
        nav_target = gr.Textbox(value="", visible=False, elem_id="workflow-nav-target")
        gr.HTML(f"<style>{PROMPTER_CSS}</style>")
        gr.Markdown("# Audition Rehearsal")
        gr.Markdown(PRIVACY_NOTICE)

        with gr.Tabs(selected="upload") as workflow_tabs:
            with gr.Tab("1. Upload", id="upload"):
                upload = gr.File(label="Upload your script", file_types=[".pdf"])
                upload_button = gr.Button("Re-parse uploaded PDF")
                upload_status = gr.Markdown("Upload a PDF to begin.")

            with gr.Tab("2. Review parsed script", id="review"):
                with gr.Row():
                    with gr.Column(scale=1):
                        characters = gr.Dataframe(
                            headers=["Character", "Dialogue lines"],
                            label="Detected characters",
                            interactive=False,
                        )
                        scenes = gr.Dataframe(
                            headers=["#", "Heading", "Page", "Characters"],
                            label="Detected scenes",
                            interactive=False,
                        )
                        rename_map = gr.Textbox(
                            label="Rename or merge characters",
                            placeholder="5ARAH=SARAH\nSARA H=SARAH",
                            lines=4,
                        )
                        line_number = gr.Number(label="Line number to reclassify", precision=0)
                        new_type = gr.Dropdown(
                            choices=[line_type.value for line_type in LineType] + ["[delete]"],
                            value=LineType.ACTION.value,
                            label="New type",
                        )
                        apply_review = gr.Button("Apply review edits")
                        continue_casting = gr.Button("Continue to Casting")
                    script_view = gr.Markdown(label="Parsed script")

            with gr.Tab("3. Cast voices", id="cast"):
                user_character = gr.Radio(label="Which character are you playing?")
                voice_rows = gr.Dataframe(
                    headers=["Character", "Voice"],
                    datatype=["str", "str"],
                    type="array",
                    label="AI voice assignments",
                    interactive=True,
                )
                preview_voice_id = gr.Dropdown(
                    choices=CHIRP_3_HD_VOICES,
                    label="Preview voice",
                    value=CHIRP_3_HD_VOICES[0],
                )
                preview_button = gr.Button("Preview selected voice")
                preview_audio = gr.Audio(label="Voice preview", autoplay=True)
                casting_status = gr.Markdown()
                continue_scenes = gr.Button("Continue to Scene Selection")

            with gr.Tab("4. Scene picker", id="scenes"):
                scene_choice = gr.Radio(label="Pick a scene")
                start_scene = gr.Button("Start Practicing")
                scene_status = gr.Markdown()

            with gr.Tab("5. Practice", id="practice"):
                progress = gr.Markdown("Start a scene to practice.")
                scene_script = gr.Markdown(label="Scene text")
                practice_audio = gr.Audio(label="AI line audio", autoplay=True)
                practice_status = gr.Markdown()
                mic_level = gr.Number(label="Mic input level", interactive=False)
                practice_timer = gr.Timer(value=0.5, active=True)
                dialogue_pacing = gr.Slider(
                    minimum=0.75,
                    maximum=1.25,
                    value=1.0,
                    step=0.05,
                    label="AI dialogue pace",
                    info="Lower is slower, higher is faster. Applies to upcoming AI lines.",
                )
                silence_threshold = gr.Slider(
                    minimum=500,
                    maximum=1200,
                    value=800,
                    step=50,
                    label="VAD silence threshold (ms)",
                )
                with gr.Row():
                    pause_button = gr.Button("Pause")
                    resume_button = gr.Button("Resume")
                    audio_complete_button = gr.Button("AI audio finished")
                    skip_back_button = gr.Button("Skip back")
                    skip_forward_button = gr.Button("Skip forward")
                    done_button = gr.Button("I'm done - advance")
                    restart_button = gr.Button("Restart scene")
                    refresh_button = gr.Button("Refresh")

        upload_button.click(
            process_pdf,
            inputs=[upload, state],
            outputs=[state, characters, scenes, script_view, upload_status, nav_target],
        )
        upload.upload(
            process_pdf,
            inputs=[upload, state],
            outputs=[state, characters, scenes, script_view, upload_status, nav_target],
        )
        apply_review.click(
            apply_review_changes,
            inputs=[state, rename_map, line_number, new_type],
            outputs=[state, characters, scenes, script_view, upload_status],
        )
        continue_casting.click(
            prepare_casting,
            inputs=[state],
            outputs=[user_character, voice_rows, casting_status, nav_target],
        )
        user_character.change(
            update_default_voices,
            inputs=[state, user_character],
            outputs=[voice_rows],
        )
        preview_button.click(
            preview_voice,
            inputs=[state, preview_voice_id],
            outputs=[preview_audio, casting_status],
        )
        continue_scenes.click(
            commit_casting,
            inputs=[state, user_character, voice_rows],
            outputs=[state, scene_choice, scene_status, nav_target],
        )
        start_scene.click(
            start_practice,
            inputs=[state, scene_choice, silence_threshold, dialogue_pacing],
            outputs=[
                state,
                progress,
                scene_script,
                practice_audio,
                practice_status,
                mic_level,
                nav_target,
            ],
        )
        dialogue_pacing.change(
            update_dialogue_pacing,
            inputs=[state, dialogue_pacing],
            outputs=[state, practice_status],
        )
        nav_target.change(
            fn=None,
            inputs=nav_target,
            outputs=None,
            js=_tab_switch_js(),
            show_progress="hidden",
        )
        pause_button.click(
            pause_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        resume_button.click(
            resume_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        audio_complete_button.click(
            audio_complete,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        skip_back_button.click(
            skip_back,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        skip_forward_button.click(
            skip_forward,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        done_button.click(
            manual_done,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        restart_button.click(
            restart_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        refresh_button.click(
            poll_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        practice_timer.tick(
            poll_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
        practice_audio.stop(
            audio_complete,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status, mic_level],
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    config = load_config()
    demo.launch(server_name="127.0.0.1", server_port=config.ui.port, inbrowser=config.ui.auto_open_browser)
