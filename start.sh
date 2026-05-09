#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$SCRIPT_DIR"
VENV_DIR="$CODE_DIR/.venv"
CONFIG_DIR="$HOME/.config/audition-app"
KEY_PATH="$CONFIG_DIR/gcp-key.json"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

show_homebrew_dialog() {
  local button
  button="$(osascript <<'APPLESCRIPT'
set choice to button returned of (display dialog "Audition Rehearsal requires Homebrew to install the PDF helper tool Poppler. Please install Homebrew from https://brew.sh, then launch this app again." buttons {"OK", "Open Website"} default button "Open Website" with icon caution)
return choice
APPLESCRIPT
)"
  if [ "$button" = "Open Website" ]; then
    open "https://brew.sh"
  fi
}

show_gcp_key_dialog() {
  local button
  button="$(osascript <<APPLESCRIPT
set choice to button returned of (display dialog "Audition Rehearsal needs a Google Cloud Text-to-Speech service account JSON key.\n\nSave the key file here:\n$KEY_PATH\n\nAfter saving it, launch the app again." buttons {"OK", "Open Folder"} default button "Open Folder" with icon caution)
return choice
APPLESCRIPT
)"
  if [ "$button" = "Open Folder" ]; then
    open "$CONFIG_DIR"
  fi
}

venv_python_is_compatible() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    return 1
  fi

  local version
  version="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [ "$version" = "3.11" ] || [ "$version" = "3.12" ]
}

echo "Starting Audition Rehearsal..."
echo "Code directory: $CODE_DIR"

if ! command -v brew >/dev/null 2>&1; then
  show_homebrew_dialog
  exit 1
fi

if ! command -v pdftoppm >/dev/null 2>&1; then
  echo "Installing PDF support (Poppler). This can take about 30 seconds..."
  brew install poppler
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing Python environment manager (uv). This can take about 30 seconds..."
  brew install uv
fi

if [ -d "$VENV_DIR" ] && ! venv_python_is_compatible; then
  echo "Existing Python environment is not Python 3.11/3.12. Recreating it..."
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python environment, this takes about 2 minutes the first time..."
  uv venv --python 3.12 "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

REQUIREMENTS_PATH="$CODE_DIR/requirements.txt"
HASH_PATH="$VENV_DIR/.requirements.sha256"
CURRENT_HASH="$(shasum -a 256 "$REQUIREMENTS_PATH" | awk '{print $1}')"
PREVIOUS_HASH=""

if [ -f "$HASH_PATH" ]; then
  PREVIOUS_HASH="$(cat "$HASH_PATH")"
fi

if [ "$CURRENT_HASH" != "$PREVIOUS_HASH" ]; then
  echo "Installing/updating dependencies..."
  uv pip install --python "$VENV_DIR/bin/python" --upgrade pip --quiet
  uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS_PATH"
  echo "$CURRENT_HASH" > "$HASH_PATH"
fi

mkdir -p "$CONFIG_DIR"

if [ ! -f "$KEY_PATH" ]; then
  show_gcp_key_dialog
  exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS="$KEY_PATH"

cd "$CODE_DIR"
echo "Launching Audition Rehearsal in your browser..."
exec python -m gradio app.py
