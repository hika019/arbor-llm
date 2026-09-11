#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_STEPS=25
ACTIVE_STEPS=2
CONFIG="configs/arbor.yaml"
OUTPUT="/tmp/arbor-nsys-$(date +%Y%m%d-%H%M%S)"
NSYS_BIN="${NSYS_BIN:-}"
TRAIN_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/profile_training_nsys.sh [options] [-- train-args...]

Options:
  --config PATH    Training config (default: configs/arbor.yaml)
  --wait N         Optimizer steps before capture (default: 25)
  --active N       Optimizer steps to capture (default: 2)
  --output PATH    Report path without .nsys-rep
  --nsys PATH      Nsight Systems CLI (also accepted via NSYS_BIN)
  -h, --help       Show this help

The script sources scripts/env.sh and profiles only the requested steady-state
optimizer steps. On WSL it works around CUPTI timestamp conversion issues and,
when necessary, uses the newest Windows Nsight target-linux-x64 installation.
EOF
}

while (($#)); do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --wait)
            WAIT_STEPS="$2"
            shift 2
            ;;
        --active)
            ACTIVE_STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --nsys)
            NSYS_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$WAIT_STEPS" =~ ^[0-9]+$ ]] || ! [[ "$ACTIVE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--wait must be >= 0 and --active must be > 0" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 2
fi

# shellcheck source=scripts/env.sh
source "$ROOT_DIR/scripts/env.sh"

is_wsl=false
if [[ -r /proc/sys/kernel/osrelease ]] && grep -qi microsoft /proc/sys/kernel/osrelease; then
    is_wsl=true
fi

nsys_version() {
    "$1" --version 2>/dev/null | sed -nE 's/.*version ([0-9]+\.[0-9]+).*/\1/p' | head -1
}

version_at_least() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]
}

if [[ -z "$NSYS_BIN" ]]; then
    NSYS_BIN="$(command -v nsys || true)"
fi

if $is_wsl; then
    current_version=""
    if [[ -n "$NSYS_BIN" ]]; then
        current_version="$(nsys_version "$NSYS_BIN")"
    fi
    if [[ -z "$current_version" ]] || ! version_at_least "$current_version" "2024.7"; then
        windows_nsys=""
        for candidate in \
            /mnt/c/Program\ Files/NVIDIA\ Corporation/Nsight\ Systems\ */target-linux-x64/nsys
        do
            if [[ -x "$candidate" ]]; then
                windows_nsys="$candidate"
            fi
        done
        if [[ -z "$windows_nsys" ]]; then
            echo "WSL CUDA tracing requires Nsight Systems >= 2024.7." >&2
            echo "Install a current Windows Nsight Systems or set NSYS_BIN." >&2
            exit 1
        fi
        windows_version="$(nsys_version "$windows_nsys")"
        if [[ -z "$windows_version" ]] || ! version_at_least "$windows_version" "2024.7"; then
            echo "Windows Nsight Systems is too old: ${windows_version:-unknown}" >&2
            exit 1
        fi

        # Running from a path containing spaces breaks the target's LD_PRELOAD
        # value. Copy the self-contained Linux target once to a stable path.
        local_target="/tmp/arbor-nsys-${windows_version}-root/target-linux-x64"
        if [[ ! -x "$local_target/nsys" ]]; then
            echo "Copying Nsight Systems $windows_version Linux target to $local_target"
            cp -a "$(dirname "$windows_nsys")" "$local_target"
        fi
        # The target CLI rejects direct execution from its installation
        # directory and explicitly requires an external symlink.
        local_link="/tmp/arbor-nsys-${windows_version}"
        ln -sfn "$local_target/nsys" "$local_link"
        NSYS_BIN="$local_link"
    fi

    nsys_config_dir="$HOME/.config/NVIDIA Corporation"
    nsys_config="$nsys_config_dir/nsys-config.ini"
    mkdir -p "$nsys_config_dir"
    if grep -q '^CuptiUseRawGpuTimestamps=' "$nsys_config" 2>/dev/null; then
        if ! grep -q '^CuptiUseRawGpuTimestamps=false$' "$nsys_config"; then
            echo "$nsys_config must set CuptiUseRawGpuTimestamps=false on WSL" >&2
            exit 1
        fi
    else
        printf '\nCuptiUseRawGpuTimestamps=false\n' >>"$nsys_config"
    fi
fi

if [[ -z "$NSYS_BIN" ]] || [[ ! -x "$NSYS_BIN" ]]; then
    echo "nsys executable not found; install Nsight Systems or set NSYS_BIN" >&2
    exit 1
fi

resolved_version="$(nsys_version "$NSYS_BIN" || true)"
echo "Using nsys=${NSYS_BIN} version=${resolved_version:-unknown}"
echo "Capturing wait=${WAIT_STEPS} active=${ACTIVE_STEPS} output=${OUTPUT}.nsys-rep"

set +e
ARBOR_NSYS_PROFILE="${WAIT_STEPS},${ACTIVE_STEPS}" "$NSYS_BIN" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --kill=sigkill \
    --force-overwrite=true \
    --output="$OUTPUT" \
    python -m src.train.train --config "$CONFIG" "${TRAIN_ARGS[@]}"
status=$?
set -e

if [[ -f "${OUTPUT}.nsys-rep" ]] && [[ $status -eq 137 ]]; then
    echo "Capture completed; exit 137 is expected from stop-shutdown --kill=sigkill."
    exit 0
fi
exit "$status"
