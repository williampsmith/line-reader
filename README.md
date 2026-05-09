# Audition Rehearsal

A local macOS web app for rehearsing audition sides. It parses a PDF script with Gemini Flash, lets you review the parsed screenplay, assigns Google Cloud Chirp 3 HD voices to the other characters, and runs a turn-taking practice session where your own lines stay silent and local VAD advances after you finish speaking.

PDF rasterization, Apple Vision OCR, and microphone audio are processed locally on your Mac. When Gemini parsing is enabled, script page images and OCR text are sent to Gemini for screenplay parsing. Only AI character dialogue text is sent to Google Cloud for voice synthesis.

## Requirements

- macOS
- Python 3.11 or 3.12
- Homebrew `poppler`
- Gemini Developer API key for AI script parsing
- Google Cloud Text-to-Speech service account key
- Microphone permission for the terminal or Platypus wrapper on first practice session

## Gemini Parser Setup

The app uses Gemini Flash as the primary parser because it can understand page layout better than OCR indentation heuristics.

1. Go to Google AI Studio.
2. Create a Gemini API key.
3. Save the key as plain text:

```bash
mkdir -p ~/.config/audition-app
printf '%s' 'YOUR_GEMINI_API_KEY' > ~/.config/audition-app/gemini-api-key.txt
chmod 600 ~/.config/audition-app/gemini-api-key.txt
```

For very infrequent personal use, Gemini's Developer API free tier should usually cover parsing a few audition PDFs. If Gemini is unavailable, the app surfaces the Gemini error instead of using the local OCR/heuristic fallback by default.

## Google Cloud Setup

1. Create a Google Cloud project.
2. Enable the Text-to-Speech API.
3. Create a service account with `roles/texttospeech.user`.
4. Download a JSON key.
5. Save it at:

```bash
mkdir -p ~/.config/audition-app
cp /path/to/key.json ~/.config/audition-app/gcp-key.json
```

## Install

```bash
brew install poppler

cd /Users/williampsmith/dev/git/line-reader
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.12` is not installed, install it first with your preferred Python manager. The app is intentionally not targeted at Python 3.13+ because some audio/OCR dependencies may lag current CPython releases.

## Optional Config

Create `~/.config/audition-app/config.toml` to override defaults:

```toml
[gcp]
credentials_path = "~/.config/audition-app/gcp-key.json"

[vad]
silence_threshold_ms = 800
min_speech_duration_ms = 250

[parser]
mode = "gemini"
gemini_api_key_path = "~/.config/audition-app/gemini-api-key.txt"
gemini_model = "gemini-2.5-flash-lite"
gemini_timeout_ms = 45000
fallback_to_local = false

[ui]
port = 7860
auto_open_browser = true
```

## Run

```bash
source .venv/bin/activate
python app.py
```

Open `http://127.0.0.1:7860` if the browser does not open automatically.

## Workflow

1. Upload a PDF script.
2. Review detected characters, scenes, and parsed lines.
3. Rename or merge OCR-split characters, and reclassify/delete any incorrect lines.
4. Select the character you are playing.
5. Assign AI voices to the remaining characters and preview voices.
6. Pick a scene and start practicing.
7. During your lines, the app is silent and local VAD listens for speech end. Use "I'm done - advance" if microphone detection fails.

## Platypus Wrapper

After command-line launch works:

1. Install Platypus from <https://sveinbjorn.org/platypus>.
2. Create a new app:
   - App Name: `Audition Rehearsal`
   - Script Type: `Shell`
   - Script Path: `/Users/williampsmith/dev/git/line-reader/start.sh`
   - Interface: `Status Menu` for normal use or `Text Window` while debugging
   - Identifier: `local.audition-app`
3. Build the app and place it in `~/Applications/` or on the Desktop.

Double-clicking the app starts the local Gradio server. Quitting the Platypus app stops the Python process.

## Smoke Tests

Run the unit tests:

```bash
python -m pytest
```

Manual checks:

- Upload a clean PDF side and confirm characters/scenes appear in Review.
- Merge an OCR variant such as `5ARAH=SARAH` and confirm dialogue counts update.
- Preview a Chirp 3 HD voice with a valid Google credential file.
- Start a scene where the first line is AI, confirm audio plays, then confirm your own line is silent.
- Speak your line and confirm VAD advances after sustained silence. If it does not, use "I'm done - advance" and check the mic permission prompt.

## Notes

- The app does not save scripts across launches.
- No accounts, telemetry, analytics, or remote deployment are included.
- Gemini is the primary parser. The local parser uses tunable indentation thresholds in `parser.py` and can be re-enabled as a fallback with `fallback_to_local = true`.
