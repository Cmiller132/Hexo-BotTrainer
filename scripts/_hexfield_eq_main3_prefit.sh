#!/usr/bin/env bash
# WARNING: THIS IS A GPU TRAINING JOB. Execute this script intentionally.
# Sourcing it only defines helper functions; it never starts training.
#
# By default this warm-starts additive ray7lut2 from the highest main_2
# checkpoint and trains on the ten highest-numbered main_2 sample epoch dirs.
# Checkpoint and sample discovery are deliberately independent because sample
# production can be ahead of checkpoint production. The main_2 run is input
# only: this script never writes beneath MAIN2_RUN.

_hexfield_eq_main3_prefit_usage() {
  cat <<'EOF'
Usage: scripts/_hexfield_eq_main3_prefit.sh [options] [-- PREFIT_ARGS...]

GPU prefit options (each also has the uppercase environment override shown):
  --main2-run DIR          main_2 run root              (MAIN2_RUN)
  --init-from FILE         warm-start checkpoint; default: newest epoch_*.pt
                                                        (INIT_FROM)
  --samples-root DIR       main_2 samples directory     (SAMPLES_ROOT)
  --sample-count N         newest epoch dirs to use     (SAMPLE_EPOCH_COUNT, 10)
  --data-epochs DIR...     explicit corpus dirs, bypassing sample discovery
  --data-epoch DIR         add one explicit corpus dir (repeatable)
  --out DIR                prefit/output directory      (OUT)
  --python FILE            GPU Python interpreter       (PYTHON)
  --epochs N               BC epochs                    (PREFIT_EPOCHS, 4)
  --workers N              data workers                 (PREFIT_WORKERS, 6)
  --pretrained-lr-scale X  LR multiplier for loaded params
                                                        (PRETRAINED_LR_SCALE, 0.1)
  -h, --help               show this help without starting the GPU job

Arguments following -- are forwarded to `python -m hexfield_eq.prefit`.
The successful final artifact is OUT/soak_init.pt, using raw prefit weights.
EOF
}

_hexfield_eq_latest_file() {
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

hexfield_eq_main3_prefit_main() {
  set -euo pipefail

  local script_dir root main2_run init_from samples_root out python
  local sample_count prefit_epochs workers pretrained_lr_scale
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="${ROOT:-$(dirname "$script_dir")}"
  main2_run="${MAIN2_RUN:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_2}"
  init_from="${INIT_FROM:-}"
  samples_root="${SAMPLES_ROOT:-}"
  out="${OUT:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main3_prefit/additive}"
  python="${PYTHON:-/root/.venvs/hexgt-build/bin/python}"
  sample_count="${SAMPLE_EPOCH_COUNT:-10}"
  prefit_epochs="${PREFIT_EPOCHS:-4}"
  workers="${PREFIT_WORKERS:-6}"
  pretrained_lr_scale="${PRETRAINED_LR_SCALE:-0.1}"

  local -a explicit_epochs=() prefit_extra=()
  while (($#)); do
    case "$1" in
      --main2-run) main2_run="$2"; shift 2 ;;
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
      -h|--help) _hexfield_eq_main3_prefit_usage; return 0 ;;
      --) shift; prefit_extra=("$@"); break ;;
      *) echo "unknown option: $1" >&2; _hexfield_eq_main3_prefit_usage >&2; return 2 ;;
    esac
  done

  [[ "$sample_count" =~ ^[1-9][0-9]*$ ]] || {
    echo "--sample-count must be a positive integer: $sample_count" >&2
    return 2
  }
  [[ -x "$python" ]] || { echo "GPU Python is not executable: $python" >&2; return 1; }
  [[ -f "$root/scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env" ]] || {
    echo "missing lut2 architecture env under ROOT=$root" >&2
    return 1
  }

  if [[ -z "$init_from" ]]; then
    init_from="$(_hexfield_eq_latest_file "$main2_run/checkpoints" 'epoch_*.pt')" || {
      echo "no main_2 epoch checkpoint found in $main2_run/checkpoints" >&2
      return 1
    }
  fi
  [[ -f "$init_from" ]] || { echo "warm-start checkpoint not found: $init_from" >&2; return 1; }

  local -a epoch_dirs=() all_epochs=()
  if ((${#explicit_epochs[@]})); then
    epoch_dirs=("${explicit_epochs[@]}")
  else
    samples_root="${samples_root:-$main2_run/samples}"
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
  # allocator/compile/pair-budget settings used to launch the A5 main_2 run.
  set -a
  # shellcheck source=prefit_env/hexfield_eq_raytap_a5_lut2.env
  source "$root/scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env"
  set +a
  export MALLOC_TRIM_THRESHOLD_=536870912
  export MALLOC_MMAP_THRESHOLD_=536870912
  export MALLOC_TOP_PAD_=134217728
  export TORCHINDUCTOR_COMPILE_THREADS=8
  # Same allocator mode the main_2 supervisor always ran with. Without it the
  # caching allocator fragments and the additive path's extra per-direction
  # (B,N,5,C) transients pushed peak VRAM over the 12GB WDDM line at ~step 200
  # (2026-07-12: 50x oversubscription thrash, 61W @ "100%" util).
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HEXFIELD_TRAIN_PAIR_BUDGET="${HEXFIELD_TRAIN_PAIR_BUDGET:-1.6e7}"
  # prefit's batching module reads the EQ-specific knob at import time; keep
  # the trainer-facing variable above and mirror the same budget into it.
  export HEXFIELD_EQ_PAIR_BUDGET="$HEXFIELD_TRAIN_PAIR_BUDGET"
  export PYTHONPATH="$root/packages/hexfield_eq/python${PYTHONPATH:+:$PYTHONPATH}"

  mkdir -p "$out"
  local log="$out/prefit.log"
  {
    printf 'main_3 additive prefit start: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'ROOT=%s\ninit_from=%s\nout=%s\n' "$root" "$init_from" "$out"
    printf 'sample epochs (%d):\n' "${#epoch_dirs[@]}"
    printf '  %s\n' "${epoch_dirs[@]}"
  } | tee -a "$log"

  "$python" -u -m hexfield_eq.prefit \
    --data-epochs "${epoch_dirs[@]}" \
    --out "$out" \
    --epochs "$prefit_epochs" \
    --workers "$workers" \
    --device cuda \
    --seed 1 \
    --policy-target gumbel \
    --init-from "$init_from" \
    --pretrained-lr-scale "$pretrained_lr_scale" \
    --soak-init "$out/soak_init.pt" \
    "${prefit_extra[@]}" 2>&1 | tee -a "$log"

  [[ -f "$out/soak_init.pt" ]] || {
    echo "prefit returned without creating $out/soak_init.pt" >&2
    return 1
  }
  echo "main_3 soak init ready: $out/soak_init.pt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  hexfield_eq_main3_prefit_main "$@"
fi
