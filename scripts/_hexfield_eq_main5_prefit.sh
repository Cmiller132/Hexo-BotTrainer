#!/usr/bin/env bash
# WARNING: THIS IS A GPU TRAINING JOB. Execute this script intentionally.
# Sourcing it only defines helper functions; it never starts training.
#
# main_5 BC prefit: FRESH 15-block CCAx5 weights (no --init-from by default —
# the arch differs from main_4, and a fresh fit avoids inheriting the mature
# net's sharpness), trained on the ten highest-numbered main_4 sample epoch
# dirs. The main_4 run is input only: this script never writes beneath
# MAIN4_RUN. GPU CONTENTION: do NOT run this while the main_4 trainer is live
# on the same GPU (12 GB WDDM line; see the 2026-07-12 oversubscription
# incident) — stop main_4 first (kill SUPERVISOR pid, then child) or wait.
#
# Smoke first (exercises the cell_q-less forward + losses end to end, no real
# GPU spend):   ... -- --limit-steps 2
# The successful final artifact is OUT/soak_init.pt, using raw prefit weights;
# configs/hexfield_eq_main_5.toml initialize_from points at it.

_hexfield_eq_main5_prefit_usage() {
  cat <<'EOF'
Usage: scripts/_hexfield_eq_main5_prefit.sh [options] [-- PREFIT_ARGS...]

GPU prefit options (each also has the uppercase environment override shown):
  --main4-run DIR          main_4 run root              (MAIN4_RUN)
  --init-from FILE         OPTIONAL warm-start checkpoint loaded strict=False
                           (default: none — fresh weights) (INIT_FROM)
  --samples-root DIR       main_4 samples directory     (SAMPLES_ROOT)
  --sample-count N         newest epoch dirs to use     (SAMPLE_EPOCH_COUNT, 10)
  --data-epochs DIR...     explicit corpus dirs, bypassing sample discovery
  --data-epoch DIR         add one explicit corpus dir (repeatable)
  --out DIR                prefit/output directory      (OUT)
  --python FILE            GPU Python interpreter       (PYTHON)
  --epochs N               BC epochs                    (PREFIT_EPOCHS, 6)
  --workers N              data workers                 (PREFIT_WORKERS, 6)
  --pretrained-lr-scale X  LR multiplier for loaded params under --init-from
                                                        (PRETRAINED_LR_SCALE, 0.1)
  -h, --help               show this help without starting the GPU job

Arguments following -- are forwarded to `python -m hexfield_eq.prefit`.
EOF
}

_hexfield_eq_main5_latest_file() {
  local dir="$1"
  local pattern="$2"
  local -a files=()
  mapfile -t files < <(
    find "$dir" -maxdepth 1 -type f -name "$pattern" -print 2>/dev/null |
      while IFS= read -r file; do
        [[ "$(basename "$file")" =~ ^epoch_[0-9]+\.pt$ ]] && printf '%s\n' "$file"
      done |
      sort -V
  )
  ((${#files[@]})) || return 1
  printf '%s\n' "${files[-1]}"
}

hexfield_eq_main5_prefit_main() {
  set -euo pipefail

  local script_dir root main4_run init_from samples_root out python
  local sample_count prefit_epochs workers pretrained_lr_scale
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="${ROOT:-$(dirname "$script_dir")}"
  main4_run="${MAIN4_RUN:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4}"
  # Fresh weights by default: the 15-block trunk has no shape-for-shape donor.
  init_from="${INIT_FROM:-}"
  samples_root="${SAMPLES_ROOT:-}"
  out="${OUT:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main5_prefit/cca15}"
  python="${PYTHON:-/root/.venvs/hexgt-build/bin/python}"
  sample_count="${SAMPLE_EPOCH_COUNT:-10}"
  prefit_epochs="${PREFIT_EPOCHS:-6}"
  workers="${PREFIT_WORKERS:-6}"
  pretrained_lr_scale="${PRETRAINED_LR_SCALE:-0.1}"

  local -a explicit_epochs=() prefit_extra=()
  while (($#)); do
    case "$1" in
      --main4-run) main4_run="$2"; shift 2 ;;
      --init-from) init_from="$2"; shift 2 ;;
      --samples-root) samples_root="$2"; shift 2 ;;
      --sample-count) sample_count="$2"; shift 2 ;;
      --data-epoch) explicit_epochs+=("$2"); shift 2 ;;
      --data-epochs)
        shift
        while (($#)) && [[ "$1" != --* ]]; do
          explicit_epochs+=("$1")
          shift
        done
        ;;
      --out) out="$2"; shift 2 ;;
      --python) python="$2"; shift 2 ;;
      --epochs) prefit_epochs="$2"; shift 2 ;;
      --workers) workers="$2"; shift 2 ;;
      --pretrained-lr-scale) pretrained_lr_scale="$2"; shift 2 ;;
      -h|--help) _hexfield_eq_main5_prefit_usage; return 0 ;;
      --) shift; prefit_extra=("$@"); break ;;
      *) echo "unknown option: $1" >&2; _hexfield_eq_main5_prefit_usage >&2; return 2 ;;
    esac
  done

  [[ "$sample_count" =~ ^[1-9][0-9]*$ ]] || {
    echo "--sample-count must be a positive integer: $sample_count" >&2
    return 2
  }
  [[ -x "$python" ]] || { echo "GPU Python is not executable: $python" >&2; return 1; }
  [[ -f "$root/scripts/prefit_env/hexfield_eq_main5_cca15.env" ]] || {
    echo "missing main5 architecture env under ROOT=$root" >&2
    return 1
  }
  if [[ -n "$init_from" && ! -f "$init_from" ]]; then
    echo "warm-start checkpoint not found: $init_from" >&2
    return 1
  fi

  local -a epoch_dirs=() all_epochs=()
  if ((${#explicit_epochs[@]})); then
    epoch_dirs=("${explicit_epochs[@]}")
  else
    samples_root="${samples_root:-$main4_run/samples}"
    mapfile -t all_epochs < <(
      find "$samples_root" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' -print 2>/dev/null |
        while IFS= read -r epoch_dir; do
          [[ "$(basename "$epoch_dir")" =~ ^epoch_[0-9]+$ ]] && printf '%s\n' "$epoch_dir"
        done |
        sort -V
    )
    ((${#all_epochs[@]})) || {
      echo "no sample epoch dirs found in $samples_root" >&2
      return 1
    }
    local start=0
    if ((${#all_epochs[@]} > sample_count)); then
      start=$((${#all_epochs[@]} - sample_count))
    fi
    epoch_dirs=("${all_epochs[@]:start}")
  fi
  local epoch_dir
  for epoch_dir in "${epoch_dirs[@]}"; do
    [[ -d "$epoch_dir" ]] || { echo "sample epoch dir not found: $epoch_dir" >&2; return 1; }
  done

  # Export every architecture assignment from the env file, then mirror the
  # allocator/compile/pair-budget settings the main_4 supervisor runs with.
  set -a
  # shellcheck source=prefit_env/hexfield_eq_main5_cca15.env
  source "$root/scripts/prefit_env/hexfield_eq_main5_cca15.env"
  set +a
  export MALLOC_TRIM_THRESHOLD_=536870912
  export MALLOC_MMAP_THRESHOLD_=536870912
  export MALLOC_TOP_PAD_=134217728
  export TORCHINDUCTOR_COMPILE_THREADS=8
  # Caching-allocator mode every eq GPU job runs with (12 GB WDDM line; the
  # 15-block trunk nearly doubles activation memory vs the 8-block A5 — if the
  # prefit OOMs, lower -- --batch-rows before touching anything else).
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HEXFIELD_TRAIN_PAIR_BUDGET="${HEXFIELD_TRAIN_PAIR_BUDGET:-1.6e7}"
  # prefit's batching module reads the EQ-specific knob at import time; keep
  # the trainer-facing variable above and mirror the same budget into it.
  export HEXFIELD_EQ_PAIR_BUDGET="$HEXFIELD_TRAIN_PAIR_BUDGET"
  export PYTHONPATH="$root/packages/hexfield_eq/python${PYTHONPATH:+:$PYTHONPATH}"

  mkdir -p "$out"
  local log="$out/prefit.log"
  {
    printf 'main_5 cca15 prefit start: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'ROOT=%s\ninit_from=%s\nout=%s\n' "$root" "${init_from:-<fresh>}" "$out"
    printf 'sample epochs (%d):\n' "${#epoch_dirs[@]}"
    printf '  %s\n' "${epoch_dirs[@]}"
  } | tee -a "$log"

  local -a init_args=()
  if [[ -n "$init_from" ]]; then
    init_args=(--init-from "$init_from" --pretrained-lr-scale "$pretrained_lr_scale")
  fi

  "$python" -u -m hexfield_eq.prefit \
    --data-epochs "${epoch_dirs[@]}" \
    --out "$out" \
    --epochs "$prefit_epochs" \
    --workers "$workers" \
    --device cuda \
    --seed 1 \
    --policy-target gumbel \
    "${init_args[@]}" \
    --soak-init "$out/soak_init.pt" \
    "${prefit_extra[@]}" 2>&1 | tee -a "$log"

  [[ -f "$out/soak_init.pt" ]] || {
    echo "prefit returned without creating $out/soak_init.pt" >&2
    return 1
  }
  echo "main_5 soak init ready: $out/soak_init.pt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  hexfield_eq_main5_prefit_main "$@"
fi
