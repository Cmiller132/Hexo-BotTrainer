#!/bin/bash
# R-RS1 rescoped binding rung (orchestrator-run, 07-17 evening):
# top-4 families (ranks 1-3 stab-2 + rank-4 stab-1 control) x 4 transforms
# x A/B x cap 1k (rescope 2: first 10k arm exceeded 4.5h) under the official 1 GiB profile. Rescope rationale in
# JOURNAL: prescribed top-10 x 12 x {10k,100k} is a 100h+ campaign; stab-1
# families beyond one control add only overhead noise already characterized
# at cap-128.
cd /e/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-root-stabilizer || exit 1

# one cargo host-wide
while tasklist //FI "IMAGENAME eq cargo.exe" 2>/dev/null | grep -qi "cargo.exe"; do
  echo "$(date -Is) waiting: cargo busy" >> ROOT_STABILIZER_ATLAS_1K_GATE.log
  sleep 300
done

# RAM gate (gate-class: availability >= 12 GiB AND free >= 6 GiB)
read FREE STANDBY <<< "$(powershell.exe -NoProfile -Command '$os=Get-CimInstance Win32_OperatingSystem; $f=[double]$os.FreePhysicalMemory*1KB/1GB; $s=[double](Get-Counter "\\Memory\\Standby Cache Normal Priority Bytes").CounterSamples[0].CookedValue/1GB; "{0:N2} {1:N2}" -f $f,$s' | tr -d '\r')"
AVAIL=$(echo "$FREE $STANDBY" | awk '{print $1+$2}')
echo "$(date -Is) gate free=$FREE standby=$STANDBY avail=$AVAIL req=12/6" >> ROOT_STABILIZER_ATLAS_1K_GATE.log
awk -v f="$FREE" -v a="$AVAIL" 'BEGIN{exit !(f>=6 && a>=12)}' || { echo "RAM gate FAILED" >> ROOT_STABILIZER_ATLAS_1K_GATE.log; exit 2; }

export CARGO_TARGET_DIR=.target-rs
export TSS_BACKWALK_TT_BYTES=1073741824
export TSS_LAZY_FRONTIER=1
export TSS_INTERIOR_CENSUS_GATE=1
export TSS_INCR_DEFENDER=1
export TSS_ROOT_STABILIZER_FAMILIES=2
export TSS_ROOT_STABILIZER_TRANSFORMS=2
export TSS_ROOT_STABILIZER_CAPS=1000

cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_root_stabilizer_atlas_campaign -- --ignored --test-threads=1 --nocapture > ROOT_STABILIZER_ATLAS_1K_RAW.log 2>&1
echo "SIZING_JOB_EXIT=$?" >> ROOT_STABILIZER_ATLAS_1K_RAW.log
