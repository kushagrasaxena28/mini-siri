#!/usr/bin/env bash
# One-command setup for mini-siri-local.
#
#   ./setup.sh
#
# Installs the toolchain, Python environment and models, then runs diagnostics.
# Safe to re-run: every step is idempotent.

set -euo pipefail

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'

step()  { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$RESET"; }
ok()    { printf "  %s✓%s %s\n" "$GREEN" "$RESET" "$1"; }
warn()  { printf "  %s!%s %s\n" "$YELLOW" "$RESET" "$1"; }
fail()  { printf "  %s✗%s %s\n" "$RED" "$RESET" "$1"; exit 1; }

cd "$(dirname "$0")"

# --- 1. platform -------------------------------------------------------------
step "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || fail "macOS required (this uses CoreAudio and Apple MLX)"
ok "macOS $(sw_vers -productVersion)"

if [[ "$(uname -m)" != "arm64" ]]; then
  warn "Not Apple Silicon. MLX requires an M-series chip; this will not run."
  fail "Apple Silicon (M1 or newer) required"
fi
ok "Apple Silicon ($(sysctl -n machdep.cpu.brand_string))"

# --- 2. toolchain ------------------------------------------------------------
step "Checking toolchain"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found -- installing"
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
fi
ok "uv $(uv --version | awk '{print $2}')"

# espeak-ng is the fallback pronunciation dictionary for speech synthesis.
if ! command -v espeak-ng >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    warn "espeak-ng not found -- installing"
    brew install espeak-ng
  else
    warn "espeak-ng missing and Homebrew unavailable; some words may mispronounce"
  fi
fi
command -v espeak-ng >/dev/null 2>&1 && ok "espeak-ng"

# --- 3. python environment ---------------------------------------------------
step "Installing Python environment (this takes a few minutes)"
# Pinned to 3.12: misaki, the grapheme-to-phoneme package, requires <3.13.
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --extra tts --extra dev
ok "dependencies installed"

# spaCy's English model is pulled in at runtime by the speech synthesiser and
# gets removed by `uv sync`, so install it explicitly and last.
step "Installing speech-synthesis language data"
if uv run python -c "import en_core_web_sm" >/dev/null 2>&1; then
  ok "en_core_web_sm already present"
else
  uv run python -m spacy download en_core_web_sm >/dev/null 2>&1 \
    && ok "en_core_web_sm installed" \
    || warn "could not install en_core_web_sm; synthesis may be degraded"
fi

# --- 4. models ---------------------------------------------------------------
step "Downloading models (~4 GB, one time)"
uv run python scripts/download_models.py

# --- 5. diagnostics ----------------------------------------------------------
step "Running diagnostics"
set +e
uv run mini-siri-local --check
CHECK_STATUS=$?
set -e

printf "\n%s================================================%s\n" "$BOLD" "$RESET"
if [[ $CHECK_STATUS -eq 0 ]]; then
  printf "%sSetup complete.%s  Start it with:\n\n" "$GREEN" "$RESET"
  printf "    uv run mini-siri-local                # terminal: watch every turn\n"
  printf "    uv run mini-siri-local --background   # menu-bar app\n\n"
else
  cat <<'EOF'
Setup finished, but diagnostics reported a problem.

If it was the MICROPHONE check: macOS denies microphone access to terminal
programs WITHOUT showing a prompt. Fix it by running this from Terminal.app
(not from an editor's built-in terminal), then approving the dialog:

    uv run mini-siri-local --check

If no dialog appears, enable your terminal manually under
System Settings > Privacy & Security > Microphone.
EOF
fi
