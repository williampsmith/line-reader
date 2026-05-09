# Platypus Build Guide

This guide explains how to create a macOS `.app` wrapper for Audition Rehearsal using Platypus. You only need Platypus on the Mac that builds the app. The friend receiving the zipped app/code bundle does not need Platypus.

## 1. Install Platypus

Install Platypus from:

- https://sveinbjorn.org/platypus
- Or with Homebrew:

```bash
brew install --cask platypus
```

As of December 2025, the current Platypus version is 5.5.0 and requires macOS 11 or newer.

## 2. Build Configuration

Open Platypus and create a new app with these settings.

### Basic Settings

- **App Name:** `Audition Rehearsal`
- **Script Type:** Shell
- **Shell:** `/bin/bash`
- **Script Path:** browse to `~/code/audition-app/start.sh`
- **Identifier:** `local.audition-app`
- **Author:** your name, optional
- **Version:** `1.0.0`
- **Interface:** Text Window

Use **Text Window** for v1. It lets the user see bootstrap progress and any errors, including Poppler installs, Python environment creation, dependency downloads, and missing key-file messages.

Later, once the app is stable, **Status Menu** is cleaner, but it hides useful bootstrap output during debugging.

### Icon

Use the default icon for v1. You can provide a custom `.icns` later.

### Settings Tab

- **Runs in background:** unchecked
  - We want the app visible in the Dock so the user can quit it.
- **Remain running after script execution:** unchecked
  - Gradio runs forever in the foreground. If Gradio exits, the app should exit too. Keeping this checked would leave an empty wrapper process around after the server is gone.
- **Accept dropped items:** unchecked
  - The web UI handles PDF upload.

### Bundled Files

Do not add bundled files.

The script resolves the code folder relative to its own location. In the recommended setup, `start.sh` lives inside the source folder, and Platypus wraps that script.

## 3. Build and Place

1. Click **Create App**.
2. Save the app to:

```text
~/Applications/Audition Rehearsal.app
```

The Desktop is also fine for testing.

After this step, Platypus is no longer needed to run the app. The resulting `.app` invokes `start.sh`.

## 4. First-Launch Behavior

On first launch, expect:

1. **Gatekeeper warning.**
   macOS may say the app cannot be opened because it is from an unidentified developer. Use:
   - right-click `Audition Rehearsal.app`
   - choose **Open**
   - confirm **Open**

2. **Text Window opens.**
   It shows bootstrap progress:
   - Homebrew check
   - Poppler install if needed
   - uv install if needed
   - Python 3.12 virtual environment creation through uv
   - dependency installation
   - Google Cloud key check

3. **Dialogs for blockers.**
   If a required manual step is missing, the app shows a native macOS dialog explaining what to do.

4. **Browser opens.**
   On success, Gradio starts and opens:

```text
http://127.0.0.1:7860
```

Subsequent launches should take about 3 to 5 seconds because the virtual environment already exists and the requirements hash matches.

## 5. Distributing to a Friend

Before sharing, confirm your friend has:

- macOS 11 or newer
- Homebrew installed, or willingness to install from https://brew.sh

They do not need to install Python 3.11 or 3.12 manually. The launcher installs `uv` with Homebrew if needed, then uses `uv` to create a Python 3.12 virtual environment in the source folder.

### Recommended Zip Layout

Use this folder structure:

```text
AuditionRehearsal/
├── Audition Rehearsal.app
├── audition-app/
│   ├── app.py
│   ├── start.sh
│   ├── requirements.txt
│   └── ...
└── README.txt
```

### Important Path Note

The current Platypus configuration points at:

```text
~/code/audition-app/start.sh
```

That means the receiver must place the source folder at exactly:

```text
~/code/audition-app/
```

This is a known v1 limitation if the `.app` is built against an absolute external script path.

The `start.sh` script itself is already written to resolve `CODE_DIR` relative to its own location. That makes a more flexible zip-and-share workflow possible, but only if the Platypus app is built in a way that invokes the `start.sh` located next to the source folder or if a future wrapper script resolves the adjacent `audition-app/` folder from the `.app` location.

For v1 sharing, use the exact `~/code/audition-app/` location unless you rebuild the `.app` on the target machine.

### README.txt Template

Include a `README.txt` like this:

```text
Audition Rehearsal

1. Move the audition-app folder to:
   ~/code/audition-app/

2. Move Audition Rehearsal.app to Applications or keep it in this folder.

3. On first launch, right-click Audition Rehearsal.app and choose Open.
   This bypasses the macOS unidentified developer warning.

4. The app needs a Google Cloud Text-to-Speech service account JSON key.
   Save it here:
   ~/.config/audition-app/gcp-key.json

5. If Homebrew is missing, install it from:
   https://brew.sh

6. The first launch may install uv, Poppler, Python 3.12, and Python packages.
   This can take several minutes. Leave the text window open.
```

### Code Signing and Notarization

Code signing and notarization are out of scope for v1. They are the path to removing the Gatekeeper warning if distribution becomes more serious.

## 6. Troubleshooting

### App cannot be opened because it is from an unidentified developer

Right-click the app, choose **Open**, then confirm **Open**.

### App is damaged and cannot be opened

This can happen after downloading a zip. Run:

```bash
xattr -dr com.apple.quarantine "/path/to/Audition Rehearsal.app"
```

Then right-click and open again.

### Bootstrap fails on Homebrew

Install Homebrew from:

```text
https://brew.sh
```

Then launch the app again.

### Bootstrap fails on Poppler

Run:

```bash
brew install poppler
```

Then launch the app again.

### uv install fails

The launcher installs `uv` automatically with Homebrew. If that step fails, run:

```bash
brew install uv
```

Then launch the app again.

### Python environment creation fails

The launcher uses `uv` to create `.venv` with Python 3.12:

```bash
uv venv --python 3.12 .venv
```

If this fails, confirm Homebrew and `uv` work from Terminal, then launch the app again.

### Google Cloud key is missing

Save the service account JSON key here:

```text
~/.config/audition-app/gcp-key.json
```

The launcher will create the folder and offer to open it if the file is missing.

### Microphone is not detected during practice

Open:

```text
System Settings → Privacy & Security → Microphone
```

Grant permission to **Audition Rehearsal**. Depending on how the app is launched, macOS may instead show Terminal, Python, or the Platypus wrapper name.
