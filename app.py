"""Gradio entry point for the audition rehearsal app."""

from __future__ import annotations

import html
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

from ai_parser import (
    DEFAULT_GEMINI_MAX_WORKERS,
    GeminiParseError,
    GeminiScriptParser,
    combine_page_scripts,
    parse_pages_in_parallel,
)
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
    "PDFs and microphone audio stay on your Mac. AI character lines and "
    "screenplay parsing are routed through Google Gemini and Cloud TTS."
)


TAB_LABELS = {
    "upload": "Upload",
    "review": "Review",
    "cast": "Cast",
    "scenes": "Scene",
    "practice": "Rehearse",
}

MAX_CAST_ROWS = 20


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

:root,
.dark,
html.dark,
body.dark {
  color-scheme: light !important;
  --bg-page: #f4f1ea;
  --bg-surface: #ffffff;
  --bg-surface-soft: #faf8f3;
  --ink-primary: #18120c;
  --ink-secondary: #44403c;
  --ink-tertiary: #78716c;
  --line-soft: #e7e2d6;
  --line-strong: #c8c2b4;
  --accent: #b04404;
  --accent-strong: #8a3503;
  --accent-soft: rgba(176, 68, 4, 0.07);

  --color-background-primary: var(--bg-page);
  --color-background-secondary: var(--bg-surface);
  --background-fill-primary: var(--bg-page);
  --background-fill-secondary: var(--bg-surface);
  --block-background-fill: var(--bg-surface);
  --panel-background-fill: var(--bg-surface);
  --input-background-fill: var(--bg-surface);
  --table-background-fill: var(--bg-surface);
  --table-row-focus-background-fill: var(--bg-surface-soft);
  --table-even-background-fill: var(--bg-surface-soft);
  --table-odd-background-fill: var(--bg-surface);
  --neutral-50: #faf8f3;
  --neutral-100: #f4f1ea;
  --neutral-200: #e7e2d6;
  --neutral-300: #c8c2b4;
  --neutral-400: #a8a29e;
  --neutral-500: #78716c;
  --neutral-600: #44403c;
  --neutral-700: #2c2724;
  --neutral-800: #1c1815;
  --neutral-900: #18120c;
  --neutral-950: #0c0907;
  --body-background-fill: var(--bg-page);
  --body-text-color: var(--ink-primary);
  --body-text-color-subdued: var(--ink-secondary);
  --block-label-text-color: var(--ink-primary);
  --block-title-text-color: var(--ink-primary);
  --block-info-text-color: var(--ink-tertiary);
  --link-text-color: var(--accent);
  --border-color-primary: var(--line-soft);
  --border-color-accent: var(--accent);
  --color-accent: var(--accent);
  --color-accent-soft: var(--accent-soft);
  --color-accent-strong: var(--accent-strong);
  --primary-50: #fff4ed;
  --primary-100: #ffe1cc;
  --primary-200: #ffc299;
  --primary-300: #ff9a5c;
  --primary-400: #f87224;
  --primary-500: #d95604;
  --primary-600: #b04404;
  --primary-700: #8a3503;
  --primary-800: #6f2b03;
  --primary-900: #4a1c01;
  --button-primary-background-fill: var(--accent);
  --button-primary-background-fill-hover: var(--accent-strong);
  --button-primary-text-color: #ffffff;
  --button-primary-border-color: var(--accent);
  --button-secondary-background-fill: var(--bg-surface);
  --button-secondary-background-fill-hover: var(--bg-surface-soft);
  --button-secondary-text-color: var(--ink-primary);
  --button-secondary-border-color: var(--line-strong);
  --shadow-drop: 0 1px 0 rgba(24, 18, 12, 0.04);
  --shadow-drop-lg: 0 18px 48px -32px rgba(24, 18, 12, 0.22);
}

html, body, .gradio-container, .dark .gradio-container,
html.dark body, html.dark .gradio-container {
  background: var(--bg-page) !important;
  color: var(--ink-primary) !important;
  font-family: 'Geist', system-ui, -apple-system, "Segoe UI", sans-serif !important;
  font-feature-settings: "ss01", "ss02", "cv11";
  letter-spacing: -0.006em;
}

.dark .block, .dark .form, .dark .panel, .dark .gr-box,
.dark .gr-form, .dark .gr-panel, .dark .gr-block,
.dark [class*="background"] {
  background: var(--bg-surface) !important;
  color: var(--ink-primary) !important;
}

.gradio-container {
  max-width: min(96vw, 1500px) !important;
  margin: 0 auto !important;
  padding: 32px 36px 64px !important;
}

p, .gradio-container p, .gradio-container .prose p {
  color: var(--ink-secondary) !important;
}

strong, .gradio-container strong { color: var(--ink-primary) !important; }
a, .gradio-container a { color: var(--accent) !important; text-decoration: none; }
a:hover { color: var(--accent-strong) !important; text-decoration: underline; }

footer.svelte-mpyp5e, footer { display: none !important; }

.app-shell-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 32px;
  padding: 12px 0 22px;
  border-bottom: 1px solid var(--line-soft);
  margin-bottom: 14px;
}

.app-shell-header h1 {
  font-size: 1.55rem;
  font-weight: 600;
  letter-spacing: -0.022em;
  color: var(--ink-primary);
  margin: 0;
  white-space: nowrap;
}

.app-shell-header .subtitle {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
  text-align: right;
  max-width: 38ch;
  line-height: 1.5;
}

.tab-nav {
  border: none !important;
  background: transparent !important;
  border-bottom: 1px solid var(--line-soft) !important;
  margin-bottom: 28px !important;
  padding: 0 !important;
  gap: 4px !important;
}

.tab-nav button[role="tab"] {
  border: none !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  font-family: 'Geist Mono', ui-monospace, monospace !important;
  font-weight: 500 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--ink-tertiary) !important;
  padding: 12px 18px !important;
  margin: 0 !important;
  transition: color 0.18s ease, border-color 0.18s ease !important;
}

.tab-nav button[role="tab"]:hover {
  color: var(--ink-primary) !important;
}

.tab-nav button[role="tab"][aria-selected="true"] {
  color: var(--ink-primary) !important;
  border-bottom-color: var(--accent) !important;
}

button.primary, button.lg.primary, .gr-button-primary,
.gradio-container button[class*="primary"] {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid var(--accent) !important;
  border-radius: 999px !important;
  font-weight: 500 !important;
  letter-spacing: -0.004em !important;
  padding: 10px 20px !important;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.18) inset, 0 6px 18px -10px rgba(176, 68, 4, 0.55) !important;
  transition: background 0.18s ease, transform 0.08s ease !important;
}

button.primary:hover, .gr-button-primary:hover,
.gradio-container button[class*="primary"]:hover {
  background: var(--accent-strong) !important;
}

button.primary:active, .gr-button-primary:active,
.gradio-container button[class*="primary"]:active {
  transform: translateY(1px);
}

button.secondary, .gr-button-secondary,
.gradio-container button[class*="secondary"] {
  background: var(--bg-surface) !important;
  color: var(--ink-primary) !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 999px !important;
  font-weight: 500 !important;
  padding: 9px 18px !important;
  transition: background 0.18s ease, transform 0.08s ease, border-color 0.18s ease !important;
}

button.secondary:hover, .gr-button-secondary:hover,
.gradio-container button[class*="secondary"]:hover {
  background: var(--bg-surface-soft) !important;
  border-color: var(--ink-tertiary) !important;
}

button.secondary:active { transform: translateY(1px); }

input, textarea, select, .gr-input, .gr-textbox, .gr-dropdown,
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: var(--bg-surface) !important;
  color: var(--ink-primary) !important;
  border-radius: 10px !important;
  border: 1px solid var(--line-strong) !important;
}

input::placeholder, textarea::placeholder { color: var(--ink-tertiary) !important; }

input:focus, textarea:focus, select:focus {
  outline: none !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

.gradio-container .file-preview, .gradio-container .file,
.gradio-container [data-testid="file"], .gradio-container .file-preview-holder {
  background: var(--bg-surface) !important;
  color: var(--ink-primary) !important;
  border: 1px dashed var(--line-strong) !important;
  border-radius: 14px !important;
}

.gradio-container .file-preview > *,
.gradio-container [data-testid="file"] * {
  color: var(--ink-primary) !important;
  background: transparent !important;
}

.gradio-container .file-preview button,
.gradio-container [data-testid="file"] button {
  background: var(--bg-surface) !important;
  border: 1px solid var(--line-strong) !important;
  color: var(--ink-primary) !important;
  border-radius: 999px !important;
  padding: 6px 14px !important;
}

label, .label-wrap, .gradio-container label, .gradio-container .gr-label {
  font-family: 'Geist Mono', ui-monospace, monospace !important;
  color: var(--ink-tertiary) !important;
  font-weight: 500 !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
}

.gradio-container .gr-form, .gradio-container .gr-box, .gradio-container .gr-block {
  background: transparent !important;
  border-radius: 14px !important;
}

.gradio-container .progress-bar { background: var(--accent) !important; }

.gradio-container table {
  border-collapse: collapse;
  font-family: 'Geist', system-ui, sans-serif;
  background: var(--bg-surface) !important;
}

.gradio-container th {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.7rem !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase;
  color: var(--ink-tertiary) !important;
  background: var(--bg-surface-soft) !important;
  font-weight: 500 !important;
}

.gradio-container td {
  color: var(--ink-primary) !important;
  font-size: 0.92rem !important;
}

.gradio-container td, .gradio-container th {
  border-bottom: 1px solid var(--line-soft) !important;
  padding: 11px 14px !important;
}

.gradio-container details, .gradio-container .accordion {
  background: var(--bg-surface) !important;
  border: 1px solid var(--line-soft) !important;
  border-radius: 12px !important;
}

.gradio-container details > summary, .gradio-container .accordion-header {
  font-family: 'Geist Mono', ui-monospace, monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--ink-secondary) !important;
}

.section-lead {
  display: grid;
  grid-template-columns: auto 1fr;
  column-gap: 24px;
  row-gap: 8px;
  align-items: baseline;
  margin: 8px 0 28px;
  padding-top: 12px;
  border-top: 1px solid var(--line-soft);
}

.section-lead .eyebrow {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
  white-space: nowrap;
  padding-top: 6px;
}

.section-lead .eyebrow .symbol {
  color: var(--accent);
  margin-right: 6px;
  font-weight: 600;
}

.section-lead h2 {
  margin: 0;
  font-size: clamp(1.6rem, 2.4vw, 2rem);
  font-weight: 600;
  letter-spacing: -0.022em;
  color: var(--ink-primary);
  line-height: 1.15;
}

.section-lead p {
  grid-column: 2 / 3;
  margin: 0;
  color: var(--ink-secondary);
  font-size: 0.98rem;
  line-height: 1.55;
  max-width: 60ch;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 14px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent-strong);
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.74rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: 1px solid rgba(176, 68, 4, 0.18);
}

.status-pill .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse-dot 1.6s ease-in-out infinite;
}

.status-pill.muted {
  background: var(--bg-surface-soft);
  color: var(--ink-tertiary);
  border-color: var(--line-soft);
}

.status-pill.muted .dot {
  background: var(--ink-tertiary);
  animation: none;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.55; transform: scale(0.85); }
}

.stage-shell { display: grid; gap: 14px; }

.line-card {
  background: var(--bg-surface);
  border: 1px solid var(--line-soft);
  border-radius: 14px;
  padding: 22px 26px;
  position: relative;
  transition: border-color 0.18s ease, transform 0.18s ease, opacity 0.18s ease;
}

.line-card.current {
  border-color: var(--accent);
  background: linear-gradient(180deg, var(--accent-soft) 0%, rgba(176, 68, 4, 0.0) 100%);
  box-shadow: 0 0 0 4px rgba(176, 68, 4, 0.06), 0 18px 36px -28px rgba(176, 68, 4, 0.32);
}

.line-card.user-line { border-left: 3px solid var(--accent); }
.line-card.user-line .line-meta-badge {
  background: var(--accent);
  color: #ffffff;
}

.line-card.upcoming { opacity: 0.7; }

.line-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
}

.line-meta-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(24, 18, 12, 0.06);
  color: var(--ink-primary);
  letter-spacing: 0.14em;
  font-weight: 500;
}

.line-card.current .line-meta { color: var(--accent-strong); }

.line-speaker {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.88rem;
  letter-spacing: 0.06em;
  color: var(--ink-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
  font-weight: 500;
}

.line-card.current .line-speaker { color: var(--ink-primary); }

.line-text {
  font-size: clamp(1.45rem, 2vw, 1.85rem);
  line-height: 1.32;
  font-weight: 550;
  letter-spacing: -0.014em;
  color: var(--ink-primary);
  max-width: 60ch;
}

.controls-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 8px;
  padding: 10px 4px;
  border-top: 1px solid var(--line-soft);
}

.controls-bar > * { flex: 0 0 auto; }

.settings-panel {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 20px 22px;
  background: var(--bg-surface);
  border: 1px solid var(--line-soft);
  border-radius: 14px;
}

.settings-panel .label {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
  display: block;
  margin-bottom: -4px;
}

.assignment-list {
  background: var(--bg-surface);
  border: 1px solid var(--line-soft);
  border-radius: 14px;
  overflow: hidden;
  margin-top: 12px;
}

.assignment-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line-soft);
}

.assignment-row:last-child { border-bottom: none; }

.assignment-row span {
  font-weight: 550;
  color: var(--ink-primary);
}

.assignment-row code {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  color: var(--ink-tertiary);
  background: var(--bg-surface-soft);
  padding: 3px 7px;
  border-radius: 999px;
}

.app-footer {
  margin-top: 56px;
  padding-top: 20px;
  border-top: 1px solid var(--line-soft);
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

.app-footer .symbol { color: var(--accent); }

.script-readout {
  max-height: 540px;
  overflow-y: auto;
  padding: 18px 20px;
  background: var(--bg-surface);
  border: 1px solid var(--line-soft);
  border-radius: 14px;
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.84rem;
  line-height: 1.65;
  color: var(--ink-primary);
}

.review-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0;
  border: 1px solid var(--line-soft);
  border-radius: 14px;
  background: var(--bg-surface);
  margin: 0 0 22px;
  overflow: hidden;
}

.review-summary .metric {
  padding: 18px 22px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-right: 1px solid var(--line-soft);
}

.review-summary .metric:last-child { border-right: none; }

.review-summary .value {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 1.65rem;
  font-weight: 500;
  color: var(--ink-primary);
  letter-spacing: -0.01em;
}

.review-summary.muted .value { color: var(--ink-tertiary); }

.review-summary .key {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
}

.review-row {
  gap: 24px !important;
  align-items: stretch !important;
  width: 100% !important;
}

.review-pdf-col, .review-grid-col { min-width: 0 !important; }

#pdf-preview {
  background: var(--bg-surface) !important;
  border: 1px solid var(--line-soft) !important;
  border-radius: 14px !important;
  padding: 10px !important;
  overflow: hidden !important;
  height: 80vh !important;
  max-height: 880px !important;
  min-height: 540px !important;
}

#pdf-preview .grid-wrap, #pdf-preview .grid-container {
  background: transparent !important;
  height: 100% !important;
  max-height: none !important;
  overflow-y: auto !important;
}

#pdf-preview img {
  background: #ffffff !important;
  border-radius: 8px !important;
  width: 100% !important;
  height: auto !important;
}

#review-grid {
  background: var(--bg-surface) !important;
  border: 1px solid var(--line-soft) !important;
  border-radius: 14px !important;
  padding: 4px !important;
  height: 80vh !important;
  max-height: 880px !important;
  min-height: 540px !important;
  overflow-y: auto !important;
  width: 100% !important;
}

#review-grid table {
  font-family: 'Geist Mono', ui-monospace, monospace !important;
  width: 100% !important;
  table-layout: fixed !important;
  border-collapse: collapse !important;
}

#review-grid th {
  background: var(--bg-surface-soft) !important;
  position: sticky;
  top: 0;
  z-index: 2;
}

#review-grid td, #review-grid th {
  vertical-align: top !important;
  word-wrap: break-word !important;
  overflow-wrap: anywhere !important;
  white-space: normal !important;
  padding: 8px 10px !important;
}

#review-grid th:nth-child(1), #review-grid td:nth-child(1) { width: 6% !important; }
#review-grid th:nth-child(2), #review-grid td:nth-child(2) { width: 18% !important; }
#review-grid th:nth-child(3), #review-grid td:nth-child(3) { width: 16% !important; }
#review-grid th:nth-child(4), #review-grid td:nth-child(4) { width: 46% !important; }
#review-grid th:nth-child(5), #review-grid td:nth-child(5) { width: 6% !important; }
#review-grid th:nth-child(6), #review-grid td:nth-child(6) { width: 8% !important; }

#review-grid td input, #review-grid td textarea {
  font-family: 'Geist Mono', ui-monospace, monospace !important;
  font-size: 0.84rem !important;
  background: transparent !important;
  border: none !important;
  width: 100% !important;
  white-space: normal !important;
  word-wrap: break-word !important;
  overflow-wrap: anywhere !important;
  resize: vertical !important;
  min-height: 1.6em !important;
}

#review-grid td input:focus, #review-grid td textarea:focus {
  background: var(--accent-soft) !important;
  outline: 1px solid var(--accent) !important;
  border-radius: 6px !important;
}

#review-grid td:nth-child(4) textarea { line-height: 1.4 !important; }

.script-readout::-webkit-scrollbar { width: 8px; }
.script-readout::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 999px; }

.gradio-container .markdown, .gradio-container .prose,
.gradio-container .gr-markdown {
  color: var(--ink-secondary);
}

.gradio-container .markdown p, .gradio-container .prose p { color: var(--ink-secondary) !important; }
.gradio-container .markdown strong, .gradio-container .prose strong { color: var(--ink-primary) !important; }

.gradio-container .gr-radio label, .gradio-container .gr-radio-group label {
  font-family: 'Geist', system-ui, sans-serif !important;
  text-transform: none !important;
  letter-spacing: -0.005em !important;
  font-size: 0.95rem !important;
  color: var(--ink-primary) !important;
  font-weight: 450 !important;
}

.gradio-container input[type="radio"] {
  accent-color: var(--accent) !important;
  transform: scale(1.12);
}

.gradio-container input[type="radio"]:checked {
  accent-color: var(--accent) !important;
}

.gradio-container label:has(input[type="radio"]:checked) {
  background: var(--accent-soft) !important;
  border-color: var(--accent) !important;
  color: var(--ink-primary) !important;
  font-weight: 600 !important;
}

.voice-table-head {
  display: grid;
  grid-template-columns: minmax(160px, 0.8fr) minmax(260px, 1.2fr);
  gap: 16px;
  padding: 12px 0 8px;
  border-bottom: 1px solid var(--line-soft);
  margin-top: 16px;
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
}

.voice-row {
  display: grid !important;
  grid-template-columns: minmax(160px, 0.8fr) minmax(260px, 1.2fr);
  gap: 16px !important;
  align-items: center !important;
  padding: 10px 0 !important;
  border-bottom: 1px solid var(--line-soft);
}

.voice-row-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.voice-row-label strong {
  font-size: 0.96rem;
  color: var(--ink-primary);
}

.voice-row-label span {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-tertiary);
}

.gradio-container .gr-slider input[type="range"] {
  accent-color: var(--accent) !important;
}
"""


FORCE_LIGHT_THEME_HEAD = """
<script>
(function forceLightTheme() {
  try {
    var apply = function () {
      if (document.documentElement) {
        document.documentElement.classList.remove('dark');
        document.documentElement.setAttribute('data-theme', 'light');
        document.documentElement.style.colorScheme = 'light';
      }
      if (document.body) {
        document.body.classList.remove('dark');
      }
    };
    apply();
    var observer = new MutationObserver(apply);
    if (document.documentElement) {
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    }
    document.addEventListener('DOMContentLoaded', function () {
      apply();
      if (document.body) {
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
      }
    });
  } catch (err) { /* no-op */ }
})();
</script>
"""


_LIGHT_TOKENS = {
    "body_background_fill": "#f4f1ea",
    "body_text_color": "#18120c",
    "body_text_color_subdued": "#44403c",
    "background_fill_primary": "#f4f1ea",
    "background_fill_secondary": "#ffffff",
    "block_background_fill": "#ffffff",
    "block_border_color": "#e7e2d6",
    "block_label_background_fill": "#ffffff",
    "block_label_text_color": "#18120c",
    "block_title_background_fill": "#ffffff",
    "block_title_text_color": "#18120c",
    "block_info_text_color": "#78716c",
    "block_shadow": "0 1px 0 rgba(24, 18, 12, 0.04)",
    "panel_background_fill": "#ffffff",
    "panel_border_color": "#e7e2d6",
    "border_color_primary": "#e7e2d6",
    "border_color_accent": "#b04404",
    "border_color_accent_subdued": "rgba(176, 68, 4, 0.35)",
    "color_accent_soft": "rgba(176, 68, 4, 0.07)",
    "input_background_fill": "#ffffff",
    "input_background_fill_focus": "#ffffff",
    "input_border_color": "#c8c2b4",
    "input_border_color_focus": "#b04404",
    "input_placeholder_color": "#78716c",
    "code_background_fill": "#faf8f3",
    "table_background_fill": "#ffffff",
    "table_border_color": "#e7e2d6",
    "table_even_background_fill": "#faf8f3",
    "table_odd_background_fill": "#ffffff",
    "table_row_focus_background_fill": "#faf8f3",
    "table_text_color": "#18120c",
    "link_text_color": "#b04404",
    "link_text_color_active": "#8a3503",
    "link_text_color_hover": "#8a3503",
    "link_text_color_visited": "#b04404",
    "shadow_drop": "0 1px 0 rgba(24, 18, 12, 0.04)",
    "shadow_drop_lg": "0 18px 48px -32px rgba(24, 18, 12, 0.22)",
    "shadow_inset": "inset 0 1px 0 rgba(255, 255, 255, 0.18)",
    "shadow_spread": "3px",
    "button_primary_background_fill": "#b04404",
    "button_primary_background_fill_hover": "#8a3503",
    "button_primary_text_color": "#ffffff",
    "button_primary_border_color": "#b04404",
    "button_secondary_background_fill": "#ffffff",
    "button_secondary_background_fill_hover": "#faf8f3",
    "button_secondary_text_color": "#18120c",
    "button_secondary_border_color": "#c8c2b4",
    "button_cancel_background_fill": "#ffffff",
    "button_cancel_background_fill_hover": "#faf8f3",
    "button_cancel_text_color": "#18120c",
    "button_cancel_border_color": "#c8c2b4",
    "checkbox_background_color": "#ffffff",
    "checkbox_background_color_focus": "#ffffff",
    "checkbox_background_color_hover": "#faf8f3",
    "checkbox_background_color_selected": "#b04404",
    "checkbox_border_color": "#c8c2b4",
    "checkbox_border_color_focus": "#b04404",
    "checkbox_border_color_hover": "#b04404",
    "checkbox_border_color_selected": "#b04404",
    "checkbox_label_background_fill": "#ffffff",
    "checkbox_label_background_fill_hover": "#faf8f3",
    "checkbox_label_background_fill_selected": "#ffffff",
    "checkbox_label_text_color": "#18120c",
    "checkbox_label_text_color_selected": "#18120c",
    "slider_color": "#b04404",
    "stat_background_fill": "#ffffff",
    "accordion_text_color": "#18120c",
    "chatbot_code_background_color": "#faf8f3",
    "chatbot_text_size": "0.95rem",
    "color_accent": "#b04404",
    "neutral_50": "#faf8f3",
    "neutral_100": "#f4f1ea",
    "neutral_200": "#e7e2d6",
    "neutral_300": "#c8c2b4",
    "neutral_400": "#a8a29e",
    "neutral_500": "#78716c",
    "neutral_600": "#44403c",
    "neutral_700": "#2c2724",
    "neutral_800": "#1c1815",
    "neutral_900": "#18120c",
    "neutral_950": "#0c0907",
}


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
    last_status: str = "Drop a PDF to begin."
    pdf_images: list[str] = field(default_factory=list)
    voice_overrides: dict[str, str] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        # Single-user local app. Gradio expects the initial gr.State value to be
        # deep-copyable, but runtime state holds threads and clients we want
        # to share across callbacks for this session.
        return self


def _state() -> PracticeUiState:
    return PracticeUiState()


def _ensure_state(state: PracticeUiState | None) -> PracticeUiState:
    return state if isinstance(state, PracticeUiState) else _state()


def _nav_target(tab_id: str) -> str:
    return f"{tab_id}:{time.monotonic_ns()}"


def _tab_switch_js() -> str:
    label_lookup = ", ".join(f'{key}: "{value}"' for key, value in TAB_LABELS.items())
    return f"""
(targetValue) => {{
  const raw = String(targetValue || "");
  const target = raw.split(":")[0];
  if (!target) return raw;

  const labels = {{ {label_lookup} }};
  const expected = labels[target];
  if (!expected) return raw;

  const findAndClick = function () {{
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const tab = tabs.find(function (element) {{
      const text = (element.textContent || "").trim();
      return text === expected;
    }}) || Array.from(document.querySelectorAll('button')).find(function (element) {{
      const text = (element.textContent || "").trim();
      return text === expected;
    }});
    if (tab) {{
      tab.click();
      tab.scrollIntoView({{ block: "nearest", inline: "nearest" }});
      return true;
    }}
    return false;
  }};

  if (!findAndClick()) {{
    [60, 180, 360, 720].forEach(function (delay) {{
      setTimeout(findAndClick, delay);
    }});
  }}

  return raw;
}}
"""


def _switch_to_tab_js(tab_id: str) -> str:
    label = TAB_LABELS[tab_id]
    return f"""
() => {{
  const expected = "{label}";
  const findAndClick = function () {{
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const tab = tabs.find(function (element) {{
      return (element.textContent || "").trim() === expected;
    }}) || Array.from(document.querySelectorAll('button')).find(function (element) {{
      return (element.textContent || "").trim() === expected;
    }});
    if (tab) {{
      tab.click();
      tab.scrollIntoView({{ block: "nearest", inline: "nearest" }});
      return true;
    }}
    return false;
  }};
  if (!findAndClick()) {{
    [80, 220, 480, 960].forEach(function (delay) {{
      setTimeout(findAndClick, delay);
    }});
  }}
}}
"""


def process_pdf(file: Any, state: PracticeUiState):
    state = _ensure_state(state)
    if file is None:
        yield state, [], [], [], "", state.pdf_images, "Choose a PDF first.", _nav_target("upload")
        return

    try:
        yield state, [], [], [], "", state.pdf_images, "Rasterizing pages...", ""
        pdf_path = Path(file.name if hasattr(file, "name") else file)
        pdf_bytes = pdf_path.read_bytes()
        images = rasterize(pdf_bytes)
        state.pdf_images = _save_pdf_page_images(images)
        yield state, [], [], [], "", state.pdf_images, (
            f"Running OCR on {len(images)} page{'s' if len(images) != 1 else ''}..."
        ), ""
        ocr = ocr_pages(images)
        if state.config.parser.mode == "gemini":
            try:
                parser = _gemini_parser_from_state(state)
                total_pages = len(images)
                yield (
                    state,
                    [],
                    [],
                    [],
                    "",
                    state.pdf_images,
                    f"Parsing screenplay with Gemini (0 of {total_pages} pages parsed)...",
                    "",
                )
                progress_counter = {"completed": 0}
                progress_lock = threading.Lock()

                def _on_progress(completed: int, _total: int) -> None:
                    with progress_lock:
                        progress_counter["completed"] = completed

                worker_result: dict[str, Any] = {}
                use_image = bool(state.config.parser.gemini_use_image)

                def _worker() -> None:
                    try:
                        def _page_parser(parser_obj, image, page_ocr, page_number):
                            return _parse_gemini_page(
                                parser_obj,
                                image,
                                page_ocr,
                                page_number,
                                use_image=use_image,
                            )

                        worker_result["scripts"] = parse_pages_in_parallel(
                            parser,
                            list(images),
                            ocr,
                            max_workers=DEFAULT_GEMINI_MAX_WORKERS,
                            parser_callable=_page_parser,
                            on_progress=_on_progress,
                        )
                    except BaseException as exc:  # noqa: BLE001
                        worker_result["error"] = exc

                worker_thread = threading.Thread(target=_worker, daemon=True)
                worker_thread.start()

                import time as _time
                last_completed = 0
                last_message = ""
                stall_counter = 0
                while worker_thread.is_alive():
                    worker_thread.join(timeout=0.5)
                    with progress_lock:
                        completed_now = progress_counter["completed"]

                    if completed_now != last_completed:
                        last_completed = completed_now
                        stall_counter = 0
                        message = (
                            f"Parsing screenplay with Gemini ({completed_now} of "
                            f"{total_pages} pages parsed)..."
                        )
                    else:
                        stall_counter += 1
                        outstanding = total_pages - completed_now
                        if completed_now == 0:
                            message = (
                                f"Parsing screenplay with Gemini "
                                f"({total_pages} pages in flight)..."
                            )
                        else:
                            tail = "page" if outstanding == 1 else "pages"
                            message = (
                                f"Parsing screenplay with Gemini ({completed_now} of "
                                f"{total_pages} pages parsed, waiting on {outstanding} "
                                f"{tail})..."
                            )
                    if message != last_message and (stall_counter == 0 or stall_counter % 4 == 0):
                        last_message = message
                        yield (
                            state,
                            [],
                            [],
                            [],
                            "",
                            state.pdf_images,
                            message,
                            "",
                        )

                if "error" in worker_result:
                    raise worker_result["error"]
                script = combine_page_scripts(worker_result.get("scripts", []))
            except GeminiParseError as exc:
                if not state.config.parser.fallback_to_local:
                    raise
                yield (
                    state,
                    [],
                    [],
                    [],
                    "",
                    state.pdf_images,
                    f"Gemini parsing failed: {exc}",
                    "",
                )
                yield state, [], [], [], "", state.pdf_images, "Falling back to local parser...", ""
                script = classify_lines(ocr)
        else:
            yield state, [], [], [], "", state.pdf_images, "Parsing screenplay locally...", ""
            script = classify_lines(ocr)
        validate_parse_quality(script)
    except ParsingError as exc:
        state.last_status = str(exc)
        yield state, [], [], [], "", state.pdf_images, str(exc), _nav_target("upload")
        return
    except Exception as exc:  # noqa: BLE001 - OCR/PDF libraries expose varied errors.
        state.last_status = f"Could not process PDF: {exc}"
        yield state, [], [], [], "", state.pdf_images, state.last_status, _nav_target("upload")
        return

    state.script = script
    state.last_status = (
        f"Parsed {len(script.lines)} lines across {len(script.scenes)} "
        f"{'scene' if len(script.scenes) == 1 else 'scenes'}."
    )
    yield (
        state,
        _character_table(script),
        _scene_table(script),
        _review_grid(script),
        _review_summary(script),
        state.pdf_images,
        state.last_status,
        _nav_target("review"),
    )


def _save_pdf_page_images(images: Iterable[object]) -> list[str]:
    paths = []
    for index, image in enumerate(images, start=1):
        if not hasattr(image, "save"):
            paths.append(image if isinstance(image, str) else str(image))
            continue
        suffix = f"-page-{index:03d}.png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            image.save(temp, format="PNG")
            paths.append(temp.name)
    return paths


def _gemini_parser_from_state(state: PracticeUiState) -> GeminiScriptParser:
    # Use a smaller per-page timeout when running pages in parallel so a single
    # hung page falls into the OCR-only retry sooner instead of stalling the
    # whole batch.
    timeout_ms = min(state.config.parser.gemini_timeout_ms, 30_000)
    return GeminiScriptParser.from_api_key_file(
        state.config.parser.gemini_api_key_path,
        model=state.config.parser.gemini_model,
        timeout_ms=timeout_ms,
    )


def _parse_gemini_page(
    parser: GeminiScriptParser,
    image: object,
    page_ocr: list[Any],
    page_number: int,
    *,
    use_image: bool = True,
) -> Script:
    return parser.parse_page(image, page_ocr, page_number, use_image=use_image)


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
        _review_grid(script),
        _review_summary(script),
        state.last_status,
    )


_REVIEW_TYPES = [line_type.value for line_type in LineType]
_REVIEW_HEADERS = ["#", "Type", "Character", "Text", "Page", "Delete (type x)"]


def _review_grid(script: Script) -> list[list[Any]]:
    rows = []
    for index, line in enumerate(script.lines, start=1):
        rows.append(
            [
                str(index),
                line.type.value,
                line.character or "",
                line.text,
                str(line.page),
                "",
            ]
        )
    return rows


def _is_truthy_delete_marker(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"x", "yes", "y", "1", "true", "del", "delete"}


def _review_summary_empty() -> str:
    return (
        '<div class="review-summary muted">'
        '<div class="metric"><span class="value">--</span>'
        '<span class="key">scenes</span></div>'
        '<div class="metric"><span class="value">--</span>'
        '<span class="key">lines</span></div>'
        '<div class="metric"><span class="value">--</span>'
        '<span class="key">dialogue</span></div>'
        '<div class="metric"><span class="value">--</span>'
        '<span class="key">characters</span></div>'
        "</div>"
    )


def _review_summary(script: Script) -> str:
    if script is None:
        return _review_summary_empty()
    total_lines = len(script.lines)
    dialogue_lines = sum(1 for line in script.lines if line.type == LineType.DIALOGUE)
    return (
        '<div class="review-summary">'
        f'<div class="metric"><span class="value">{len(script.scenes)}</span>'
        '<span class="key">scenes</span></div>'
        f'<div class="metric"><span class="value">{total_lines}</span>'
        '<span class="key">lines</span></div>'
        f'<div class="metric"><span class="value">{dialogue_lines}</span>'
        '<span class="key">dialogue</span></div>'
        f'<div class="metric"><span class="value">{len(script.characters)}</span>'
        '<span class="key">characters</span></div>'
        "</div>"
    )


def apply_review_table(state: PracticeUiState, table_rows):
    state = _ensure_state(state)
    if state.script is None:
        return (
            state,
            [],
            [],
            [],
            "",
            "Upload and parse a script first.",
        )

    rows = _iter_voice_rows(table_rows)
    new_lines = []
    errors = []
    deletions = 0
    original = state.script.lines
    for row_index, row in enumerate(rows):
        if row_index >= len(original):
            continue
        if len(row) < 6:
            continue
        if _is_truthy_delete_marker(row[5]):
            deletions += 1
            continue
        type_value = str(row[1]).strip().lower()
        if type_value not in _REVIEW_TYPES:
            errors.append(f"Row {row_index + 1}: unknown type '{type_value}'.")
            new_lines.append(original[row_index])
            continue
        character_value = str(row[2]).strip() or None
        text_value = str(row[3]).strip()
        from dataclasses import replace as dc_replace

        new_lines.append(
            dc_replace(
                original[row_index],
                type=LineType(type_value),
                character=character_value,
                text=text_value,
            )
        )

    from parser import _build_script

    script = _build_script(new_lines)
    state.script = script
    summary = _review_summary(script)
    if errors:
        status = "Applied with warnings: " + " ".join(errors)
    elif deletions:
        status = f"Applied. Removed {deletions} line{'s' if deletions != 1 else ''}."
    else:
        status = "Applied. Edit any cell again to keep refining."
    state.last_status = status
    return (
        state,
        _character_table(script),
        _scene_table(script),
        _review_grid(script),
        summary,
        status,
    )


def prepare_casting(state: PracticeUiState):
    state = _ensure_state(state)
    if state.script is None:
        return (
            gr.update(choices=[], value=None),
            "Parse and review a script first.",
            _nav_target("review"),
        )
    characters = sorted(state.script.characters)
    return gr.update(choices=characters, value=None), "Choose your role.", _nav_target("cast")


def update_default_voices(state: PracticeUiState, user_character: str):
    state = _ensure_state(state)
    if state.script is None or not user_character:
        state.voice_overrides = {}
        return (
            state,
            gr.update(choices=[], value=None),
            gr.update(value=None),
            _voice_assignment_status(state, user_character),
        )
    defaults = default_voice_assignment(
        sorted(state.script.characters),
        user_character,
    )
    new_overrides: dict[str, str] = {}
    for character, default_voice in defaults.voice_for_character.items():
        new_overrides[character] = state.voice_overrides.get(character) or default_voice
    state.voice_overrides = new_overrides
    choices = sorted(new_overrides)
    selected_character = choices[0] if choices else None
    selected_voice = new_overrides.get(selected_character) if selected_character else None
    return (
        state,
        gr.update(choices=choices, value=selected_character),
        gr.update(value=selected_voice),
        _voice_assignment_status(state, user_character),
    )


def set_voice_override(state: PracticeUiState, character: str, voice_id: str) -> PracticeUiState:
    state = _ensure_state(state)
    if not character:
        return state
    if voice_id:
        state.voice_overrides[character] = voice_id
    else:
        state.voice_overrides.pop(character, None)
    return state


def update_voice_override(
    state: PracticeUiState,
    user_character: str,
    character: str,
    voice_id: str,
):
    state = set_voice_override(state, character, voice_id)
    return state, _voice_assignment_status(state, user_character)


def load_voice_for_character(state: PracticeUiState, character: str):
    state = _ensure_state(state)
    return gr.update(value=state.voice_overrides.get(character))


def update_cast_voice_rows(state: PracticeUiState, user_character: str):
    state = _ensure_state(state)
    if state.script is None or not user_character:
        state.voice_overrides = {}
        return (state, *_empty_cast_row_updates(), _voice_assignment_status(state, user_character))

    defaults = default_voice_assignment(sorted(state.script.characters), user_character)
    new_overrides: dict[str, str] = {}
    for character, default_voice in defaults.voice_for_character.items():
        new_overrides[character] = state.voice_overrides.get(character) or default_voice
    state.voice_overrides = new_overrides

    row_updates: list[Any] = []
    characters = sorted(new_overrides)
    for index in range(MAX_CAST_ROWS):
        if index < len(characters):
            character = characters[index]
            voice_id = new_overrides[character]
            row_updates.extend(
                [
                    gr.update(visible=True),
                    _voice_row_label(character),
                    gr.update(value=voice_id),
                    character,
                ]
            )
        else:
            row_updates.extend([gr.update(visible=False), "", gr.update(value=None), ""])
    return (state, *row_updates, _voice_assignment_status(state, user_character))


def _empty_cast_row_updates() -> list[Any]:
    row_updates: list[Any] = []
    for _ in range(MAX_CAST_ROWS):
        row_updates.extend([gr.update(visible=False), "", gr.update(value=None), ""])
    return row_updates


def _voice_row_label(character: str) -> str:
    return (
        '<div class="voice-row-label">'
        f"<strong>{html.escape(character)}</strong>"
        "<span>AI character</span>"
        "</div>"
    )


def _voice_assignment_status(
    state: PracticeUiState,
    user_character: str | None = None,
) -> str:
    state = _ensure_state(state)
    if not user_character:
        return (
            '<div class="status-pill muted"><span class="dot"></span>'
            "<span>Pick your character first</span></div>"
        )
    if not state.voice_overrides:
        return (
            '<div class="status-pill muted"><span class="dot"></span>'
            "<span>No AI characters to assign</span></div>"
        )
    rows = "".join(
        "<div class=\"assignment-row\">"
        f"<span>{html.escape(character)}</span>"
        f"<code>{html.escape(voice_id)}</code>"
        "</div>"
        for character, voice_id in sorted(state.voice_overrides.items())
    )
    return f'<div class="assignment-list">{rows}</div>'


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


def commit_casting(state: PracticeUiState, user_character: str):
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

    needed = set(state.script.characters) - {user_character}
    voice_for_character = {
        character: voice_id
        for character, voice_id in state.voice_overrides.items()
        if character in needed and voice_id
    }

    missing = sorted(needed - set(voice_for_character))
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
            _nav_target("cast"),
            gr.Timer(active=False),
        )
    scene_index = _scene_index_from_choice(scene_choice)
    state.queue_scene_index = scene_index
    try:
        queue = build_practice_queue(state.script, state.assignment, scene_index)
    except ValueError as exc:
        return state, "", "", gr.update(), str(exc), _nav_target("scenes"), gr.Timer(active=False)
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
    return (*_practice_outputs(state), _nav_target("practice"), gr.Timer(value=0.5, active=True))


def update_parser_mode(state: PracticeUiState, parser_choice: str):
    state = _ensure_state(state)
    use_image = parser_choice.startswith("Vision")
    state.config = AppConfig(
        gcp=state.config.gcp,
        vad=state.config.vad,
        parser=replace(state.config.parser, gemini_use_image=use_image),
        ui=state.config.ui,
    )
    label = "Vision" if use_image else "Text-only"
    return state, f"Parser mode: {label}."


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
            "Pick a scene to start rehearsing.",
            "",
            gr.update(),
            state.last_status,
        )
    line_md = _practice_header_markdown(session)
    scene_md = _practice_scene_markdown(session)
    audio = _audio_file(state.pending_audio.pop(0)) if state.pending_audio else gr.update()
    status = _practice_status(state)
    return state, line_md, scene_md, audio, status


def _practice_status(state: PracticeUiState) -> str:
    session = state.session
    if session is None:
        return state.last_status
    if session.state == SessionState.ERROR:
        return f"Synthesis error: {session.error_message}"
    if session.state == SessionState.DONE:
        return "Scene complete. Restart or pick a different scene."
    if session.paused:
        return "Paused."
    current, total = session.progress
    detail = {
        SessionState.AI_TURN: "AI is speaking",
        SessionState.USER_TURN: (
            "Listening for you to finish your line"
            if state.vad_active
            else "Your turn"
        ),
        SessionState.INIT: "Loading",
    }.get(session.state, session.state.value.replace("_", " "))
    return f"Line {current} of {total} - {detail}"


def _practice_header_markdown(session: PracticeSession) -> str:
    if session.current_item is None:
        return (
            '<div class="status-pill"><span class="dot"></span>'
            "<span>Scene complete</span></div>"
        )
    current, total = session.progress
    if session.state == SessionState.USER_TURN:
        detail = "Your turn"
    elif session.state == SessionState.AI_TURN:
        detail = "AI is speaking"
    elif session.state == SessionState.PAUSED:
        detail = "Paused"
    else:
        detail = "Loading"
    pill_class = "status-pill" if session.state != SessionState.PAUSED else "status-pill muted"
    pill = (
        f'<div class="{pill_class}"><span class="dot"></span>'
        f"<span>Line {current} of {total} &middot; {html.escape(detail)}</span></div>"
    )
    return pill


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
    return '<div class="stage-shell">\n' + "\n".join(blocks) + "\n</div>"


def _practice_prompt_block(
    item: PracticeQueueItem,
    index: int,
    is_current: bool,
) -> str:
    is_user = item.role == "user"
    speaker_label = "YOUR LINE" if is_user else "AI LINE"
    current_label = "CURRENTLY SPEAKING" if is_current else f"NEXT {index + 1}"
    classes = ["line-card"]
    if is_current:
        classes.append("current")
    else:
        classes.append("upcoming")
    if is_user:
        classes.append("user-line")
    character = html.escape(item.character)
    text = html.escape(item.text)
    return (
        f'<div class="{" ".join(classes)}">\n'
        f'<div class="line-meta">'
        f'<span class="line-meta-badge">{speaker_label}</span>'
        f"<span>{current_label}</span>"
        "</div>\n"
        f'<div class="line-speaker">{character}</div>\n'
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


def _after_parse_success(state: PracticeUiState) -> None:
    return None


def _section_lead(number: str, title: str, description: str) -> str:
    return (
        '<div class="section-lead">'
        f'<span class="eyebrow"><span class="symbol">&sect;</span>{html.escape(number)}</span>'
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(description)}</p>"
        "</div>"
    )


def _app_theme():
    import inspect as _inspect

    base = gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
    )
    valid_params = set(_inspect.signature(base.set).parameters)
    overrides: dict[str, str] = {}
    for token, value in _LIGHT_TOKENS.items():
        if token in valid_params:
            overrides[token] = value
        dark_token = f"{token}_dark"
        if dark_token in valid_params:
            overrides[dark_token] = value
    base.set(**overrides)
    return base


def build_app():
    with gr.Blocks(title="Audition Rehearsal", theme=_app_theme()) as demo:
        state = gr.State(None)
        nav_target = gr.Textbox(value="", visible=False, elem_id="workflow-nav-target")
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.HTML(
            '<div class="app-shell-header">'
            "<h1>Audition Rehearsal</h1>"
            '<span class="subtitle">'
            "Local-first &middot; Gemini parse &middot; Chirp 3 HD voice"
            "</span>"
            "</div>"
        )

        with gr.Tabs(selected="upload") as workflow_tabs:
            with gr.Tab(TAB_LABELS["upload"], id="upload"):
                gr.HTML(
                    _section_lead(
                        "Step 01",
                        "Upload your scanned side",
                        "Drop a PDF and we will rasterize, OCR with Apple Vision, and parse with Gemini.",
                    )
                )
                upload = gr.File(
                    label="Drop or browse for a PDF",
                    file_types=[".pdf"],
                    height=180,
                )
                parser_mode_choice = gr.Radio(
                    label="AI parser mode",
                    info=(
                        "Vision sends the page image plus OCR. Text-only sends only OCR "
                        "and is much faster, but loses visual layout cues."
                    ),
                    choices=[
                        "Vision (richer layout, slower, ~10-30s per page)",
                        "Text-only (faster, ~2-5s per page)",
                    ],
                    value="Vision (richer layout, slower, ~10-30s per page)",
                )
                upload_status = gr.Markdown("Drop a PDF to begin.")
                with gr.Accordion("Advanced", open=False):
                    upload_button = gr.Button(
                        "Re-parse the current PDF",
                        variant="secondary",
                    )

            with gr.Tab(TAB_LABELS["review"], id="review"):
                gr.HTML(
                    _section_lead(
                        "02",
                        "Review the parse",
                        "Compare the source PDF on the left against the parsed lines on "
                        "the right. Edit any cell. Type x in the Delete column to drop "
                        "a row. Hit Apply, then continue.",
                    )
                )
                review_summary = gr.HTML(_review_summary_empty())
                with gr.Row(equal_height=False, elem_classes="review-row"):
                    with gr.Column(scale=5, elem_classes="review-pdf-col"):
                        pdf_preview = gr.Gallery(
                            label="Source PDF",
                            show_label=False,
                            columns=1,
                            object_fit="contain",
                            height=720,
                            preview=True,
                            allow_preview=True,
                            elem_id="pdf-preview",
                        )
                    with gr.Column(scale=7, elem_classes="review-grid-col"):
                        review_grid = gr.Dataframe(
                            headers=_REVIEW_HEADERS,
                            interactive=True,
                            wrap=True,
                            label=" ",
                            elem_id="review-grid",
                        )
                with gr.Row():
                    apply_review = gr.Button(
                        "Apply edits",
                        variant="secondary",
                    )
                    continue_casting = gr.Button(
                        "Continue to casting",
                        variant="primary",
                    )
                with gr.Accordion("Diagnostics", open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            characters = gr.Dataframe(
                                headers=["Character", "Dialogue lines"],
                                label="Detected characters",
                                interactive=False,
                            )
                        with gr.Column(scale=1):
                            scenes = gr.Dataframe(
                                headers=["#", "Heading", "Page", "Characters"],
                                label="Detected scenes",
                                interactive=False,
                            )

            with gr.Tab(TAB_LABELS["cast"], id="cast"):
                gr.HTML(
                    _section_lead(
                        "Step 03",
                        "Cast the room",
                        "Pick the character you are reading and assign voices to the rest.",
                    )
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        user_character = gr.Radio(
                            label="You are playing",
                        )
                        gr.HTML(
                            '<div class="voice-table-head">'
                            "<span>Character</span><span>Voice</span>"
                            "</div>"
                        )
                        cast_voice_rows = []
                        for _ in range(MAX_CAST_ROWS):
                            with gr.Row(visible=False, elem_classes="voice-row") as voice_row:
                                character_label = gr.HTML()
                                voice_dropdown = gr.Dropdown(
                                    choices=CHIRP_3_HD_VOICES,
                                    label=" ",
                                    show_label=False,
                                )
                                character_holder = gr.Textbox(visible=False)
                            cast_voice_rows.append(
                                (voice_row, character_label, voice_dropdown, character_holder)
                            )
                        voice_assignment_status = gr.HTML(
                            _voice_assignment_status(_state(), None)
                        )
                    with gr.Column(scale=3):
                        with gr.Accordion("Voice preview", open=True):
                            preview_voice_id = gr.Dropdown(
                                choices=CHIRP_3_HD_VOICES,
                                label="Preview a voice",
                                value=CHIRP_3_HD_VOICES[0],
                            )
                            preview_button = gr.Button(
                                "Play preview",
                                variant="secondary",
                            )
                            preview_audio = gr.Audio(
                                label="Sample",
                                autoplay=True,
                                show_label=False,
                            )
                casting_status = gr.Markdown()
                continue_scenes = gr.Button(
                    "Continue to Scene Selection",
                    variant="primary",
                )

            with gr.Tab(TAB_LABELS["scenes"], id="scenes"):
                gr.HTML(
                    _section_lead(
                        "Step 04",
                        "Pick a scene",
                        "Rehearsal will run from the selected scene to the end of the side.",
                    )
                )
                scene_choice = gr.Radio(label="Available scenes")
                start_scene = gr.Button("Start rehearsing", variant="primary")
                scene_status = gr.Markdown()

            with gr.Tab(TAB_LABELS["practice"], id="practice"):
                gr.HTML(
                    _section_lead(
                        "Step 05",
                        "Rehearse the scene",
                        "AI partners speak their lines. Your lines stay silent until you finish.",
                    )
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        progress = gr.HTML(
                            '<div class="status-pill muted"><span class="dot"></span>'
                            "<span>Pick a scene to start rehearsing</span></div>"
                        )
                        scene_script = gr.HTML(
                            '<div class="line-card upcoming">'
                            '<div class="line-meta"><span class="line-meta-badge">READY</span>'
                            "<span>Press start in step 04</span></div>"
                            "</div>"
                        )
                        practice_audio = gr.Audio(
                            label="AI line audio",
                            autoplay=True,
                            show_label=False,
                            visible=True,
                        )
                        with gr.Row(elem_classes="controls-bar"):
                            pause_button = gr.Button("Pause", variant="secondary")
                            resume_button = gr.Button("Resume", variant="secondary")
                            skip_back_button = gr.Button("Back", variant="secondary")
                            skip_forward_button = gr.Button("Skip", variant="secondary")
                            done_button = gr.Button("I'm done", variant="primary")
                            restart_button = gr.Button("Restart scene", variant="secondary")
                        practice_status = gr.Markdown()
                    with gr.Column(scale=3, elem_classes="settings-panel"):
                        gr.HTML('<span class="label">AI dialogue pace</span>')
                        dialogue_pacing = gr.Slider(
                            minimum=0.75,
                            maximum=1.25,
                            value=1.0,
                            step=0.05,
                            label=" ",
                            info="Lower is slower, higher is faster.",
                        )
                        gr.HTML('<span class="label">Pause sensitivity</span>')
                        silence_threshold = gr.Slider(
                            minimum=500,
                            maximum=1200,
                            value=800,
                            step=50,
                            label=" ",
                            info="Milliseconds of silence before advancing.",
                        )
                practice_timer = gr.Timer(value=0.5, active=False)

        gr.HTML(
            '<div class="app-footer">'
            '<span><span class="symbol">&sect;</span> Audition Rehearsal &mdash; v1</span>'
            f'<span>{html.escape(PRIVACY_NOTICE)}</span>'
            "</div>"
        )

        upload_button.click(
            process_pdf,
            inputs=[upload, state],
            outputs=[
                state,
                characters,
                scenes,
                review_grid,
                review_summary,
                pdf_preview,
                upload_status,
                nav_target,
            ],
        ).success(
            fn=_after_parse_success,
            inputs=[state],
            outputs=[],
            js=_switch_to_tab_js("review"),
            show_progress="hidden",
        )
        upload.upload(
            process_pdf,
            inputs=[upload, state],
            outputs=[
                state,
                characters,
                scenes,
                review_grid,
                review_summary,
                pdf_preview,
                upload_status,
                nav_target,
            ],
        ).success(
            fn=_after_parse_success,
            inputs=[state],
            outputs=[],
            js=_switch_to_tab_js("review"),
            show_progress="hidden",
        )
        apply_review.click(
            apply_review_table,
            inputs=[state, review_grid],
            outputs=[
                state,
                characters,
                scenes,
                review_grid,
                review_summary,
                upload_status,
            ],
        )
        continue_casting.click(
            prepare_casting,
            inputs=[state],
            outputs=[user_character, casting_status, nav_target],
            queue=False,
        )
        cast_voice_row_outputs = []
        for row_group, character_label, voice_dropdown, character_holder in cast_voice_rows:
            cast_voice_row_outputs.extend(
                [row_group, character_label, voice_dropdown, character_holder]
            )
        user_character.change(
            update_cast_voice_rows,
            inputs=[state, user_character],
            outputs=[state, *cast_voice_row_outputs, voice_assignment_status],
            queue=False,
        )
        for _row_group, _character_label, row_voice_dropdown, row_character_holder in cast_voice_rows:
            row_voice_dropdown.change(
                update_voice_override,
                inputs=[state, user_character, row_character_holder, row_voice_dropdown],
                outputs=[state, voice_assignment_status],
                queue=False,
                show_progress="hidden",
            )
        preview_button.click(
            preview_voice,
            inputs=[state, preview_voice_id],
            outputs=[preview_audio, casting_status],
        )
        continue_scenes.click(
            commit_casting,
            inputs=[state, user_character],
            outputs=[state, scene_choice, scene_status, nav_target],
            queue=False,
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
                nav_target,
                practice_timer,
            ],
        )
        dialogue_pacing.change(
            update_dialogue_pacing,
            inputs=[state, dialogue_pacing],
            outputs=[state, practice_status],
        )
        parser_mode_choice.change(
            update_parser_mode,
            inputs=[state, parser_mode_choice],
            outputs=[state, upload_status],
            queue=False,
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
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        resume_button.click(
            resume_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        skip_back_button.click(
            skip_back,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        skip_forward_button.click(
            skip_forward,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        done_button.click(
            manual_done,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        restart_button.click(
            restart_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
        )
        practice_timer.tick(
            poll_practice,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
            queue=False,
            show_progress="hidden",
        )
        practice_audio.stop(
            audio_complete,
            inputs=[state],
            outputs=[state, progress, scene_script, practice_audio, practice_status],
            queue=False,
            show_progress="hidden",
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    config = load_config()
    demo.launch(
        server_name="127.0.0.1",
        server_port=config.ui.port,
        inbrowser=config.ui.auto_open_browser,
        head=FORCE_LIGHT_THEME_HEAD,
    )
