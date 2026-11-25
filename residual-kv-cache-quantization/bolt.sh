#!/usr/bin/env bash
# Developer entry point: environment setup, build, tests, benchmark.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLS_DIR="${BOLT_TOOLS_DIR:-$ROOT/.tools}"
UV_VERSION="0.7.12"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export PYTHONDONTWRITEBYTECODE=1

info() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit "${2:-1}"; }

find_uv() {
    if [[ -n "${UV:-}" ]]; then
        printf '%s\n' "$UV"
    elif [[ -x "$TOOLS_DIR/bin/uv" ]]; then
        printf '%s\n' "$TOOLS_DIR/bin/uv"
    elif command -v uv >/dev/null 2>&1; then
        command -v uv
    fi
}

uv_cmd() {
    local uv
    uv=$(find_uv)
    [[ -n "$uv" ]] || die "uv not found; run ./bolt.sh setup first"
    RUST_LOG=warn "$uv" "$@"
}

install_uv() {
    [[ -n "$(find_uv)" ]] && return
    local target
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
        Linux/aarch64) target=aarch64-unknown-linux-gnu ;;
        *) die "unsupported platform $(uname -s)/$(uname -m)" ;;
    esac
    info "Installing uv $UV_VERSION into $TOOLS_DIR/bin"
    mkdir -p "$TOOLS_DIR/bin"
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" |
        tar -xz -C "$TOOLS_DIR/bin" --strip-components=1 "uv-$target/uv"
}

python_run() {
    [[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || die "environment missing; run ./bolt.sh setup first"
    (cd "$ROOT" && uv_cmd run --project "$ROOT" --frozen --no-sync python "$@")
}

cmd_setup() {
    command -v nvidia-smi >/dev/null 2>&1 || info "nvidia-smi not found; GPU tests and benchmarks need an NVIDIA driver"
    install_uv
    info "Syncing Python environment from uv.lock"
    uv_cmd sync --project "$ROOT" --locked --managed-python
    cmd_build
    ok "Ready. Try ./bolt.sh info, ./bolt.sh test, ./bolt.sh bench --quick"
}

cmd_build() {
    local force=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force) force=(--force) ;;
            *) die "unknown build option: $1" 2 ;;
        esac
        shift
    done
    info "Building CUDA extension"
    python_run "$ROOT/build.py" "${force[@]}"
}

cmd_test() {
    local paths=() args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cpu) paths+=(tests/test_oracle.py) ;;
            --gpu) paths+=(tests/test_correctness.py tests/test_contract.py) ;;
            --all) paths+=(tests) ;;
            *) args+=("$1") ;;
        esac
        shift
    done
    [[ ${#paths[@]} -gt 0 ]] || paths=(tests)
    python_run -m pytest -p no:cacheprovider "${paths[@]}" "${args[@]}"
}

cmd_bench() {
    python_run "$ROOT/benchmarks/bench.py" "$@"
}

cmd_cases() {
    local kind=correctness
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quick) kind=quick ;;
            --perf) kind=performance ;;
            --correctness) kind=correctness ;;
            *) die "unknown cases option: $1" 2 ;;
        esac
        shift
    done
    python_run -c "
from residual_kv.testing import correctness_cases, performance_cases, quick_cases
cases = {'quick': quick_cases, 'performance': performance_cases,
         'correctness': correctness_cases}['$kind']()
print('\n'.join(c.name for c in cases))
print(f'# {len(cases)} $kind cases')"
}

cmd_info() {
    python_run -c "
import platform, subprocess, sys, torch, residual_kv
from build import wheel_cuda_home
home = wheel_cuda_home()
nvcc = subprocess.run([str(home / 'bin' / 'nvcc'), '--version'], capture_output=True, text=True).stdout.split('release ')[-1].split(',')[0] if home else 'not installed'
print(f'python      {platform.python_version()}')
print(f'torch       {torch.__version__} (CUDA {torch.version.cuda})')
print(f'nvcc        {nvcc}')
if not torch.cuda.is_available():
    sys.exit('no CUDA device available')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'cuda:{i}      {p.name}, sm_{p.major}{p.minor}, {p.multi_processor_count} SMs, {p.total_memory / 2**30:.1f} GiB')
best = residual_kv.best_device()
print(f'selected    {best} ({torch.cuda.get_device_name(best)})')
try:
    print(f'backend     {residual_kv.backend(best)} (available: {\", \".join(residual_kv.available_backends(best))})')
except RuntimeError as error:
    print(f'backend     unknown: {error}')"
}

cmd_clean() {
    local all=0
    [[ "${1:-}" == --all ]] && all=1
    info "Removing build output"
    rm -rf -- "$ROOT/.build" "$ROOT/.pytest_cache"
    find "$ROOT" -path "$ROOT/.venv" -prune -o -name __pycache__ -type d -exec rm -rf -- {} +
    if (( all )); then
        info "Removing Python environment and local tools"
        rm -rf -- "$ROOT/.venv" "$TOOLS_DIR"
    fi
    ok "Clean"
}

cmd_help() {
    cat <<EOF
Usage: ./bolt.sh COMMAND [OPTIONS]

Commands:
  setup                 Install uv if missing, sync the locked environment, build
  build [--force]       Build the CUDA extension (skipped when sources are unchanged)
  test [MODE] [ARGS]    Run the test suite; extra ARGS go to pytest
  bench [OPTIONS]       Measure latency on the selected GPU
  cases [KIND]          List test cases
  info                  Show toolchain, GPUs, selected GPU and backend
  clean [--all]         Remove build output (--all: also .venv and local tools)
  help                  Show this help

Test modes:
  --cpu                 Fixture, oracle and device-ranking checks (no GPU needed)
  --gpu                 Correctness against the FP64 oracle and runtime checks
  --all                 Everything (default)
  --full                Correctness cases with every seed and pattern length
  -k EXPR, -x, -v ...   Any pytest option

Benchmark options:
  --quick               Five cases instead of 24
  --backend NAME        Force fast or portable (default: automatic)
  --compare             Also time the portable backend and report the speedup
  --samples N           Timed batches per case (default 10)
  --warmup N            Untimed batches per case (default 2)
  --output PATH         Write results as JSON

Case kinds:
  --correctness         Every correctness case (default)
  --quick               Five quick cases
  --perf                24 benchmark cases

Environment:
  RESIDUAL_KV_DEVICE      Use this device instead of the strongest GPU (e.g. cuda:1)
  RESIDUAL_KV_CUDA_ARCHS  Target architectures for build (default "75;90")
  UV                      Path to a uv binary to use
  BOLT_TOOLS_DIR          Where setup installs uv (default .tools)

Examples:
  ./bolt.sh setup
  ./bolt.sh test --cpu
  ./bolt.sh test --gpu -k "n32768 and loop3"
  ./bolt.sh test --full
  ./bolt.sh bench --quick
  ./bolt.sh bench --compare --output results.json
EOF
}

command_name="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$command_name" in
    setup) cmd_setup "$@" ;;
    build) cmd_build "$@" ;;
    test) cmd_test "$@" ;;
    bench) cmd_bench "$@" ;;
    cases) cmd_cases "$@" ;;
    info) cmd_info "$@" ;;
    clean) cmd_clean "$@" ;;
    help|-h|--help) cmd_help ;;
    *) cmd_help >&2; exit 2 ;;
esac
