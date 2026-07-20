# Throughput benchmark integration gaps

## Standard-tier invocation

The current harness has the contract, gates, adapter, frozen sets, canaries,
archive, and diff layers, but no quick/standard/full runner or standard-tier
entry point.  There is therefore no existing standard-run control flow into
which `bench.py` can be honestly wired in this change without inventing the
remaining orchestration layer.

When that runner lands, every standard invocation must run the equivalent of:

```text
/root/.venvs/hexo-bottrainer-wsl/bin/python \
  scripts/tss_harness/bench.py --full \
  --config-json '<standard arm TSS config>' \
  --out '<archive>/bench_scorecard.json'
```

The runner must archive the scorecard and surface its report-only `adoption`
classification.  It must propagate benchmark exit status for quiet-machine
refusal, runtime errors, and especially `verify_failed_total != 0`; it must not
turn the throughput tolerance classification into a hard gate.

No production module or Rust hook is missing for the benchmark itself.  The
existing `run_continuous` callback protocol cleanly drains active slots by
returning `None`, and the existing `ContinuousDriver` supplies seeding and all
required TSS telemetry.
