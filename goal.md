# Hexo Runner Performance Goal

Performance is fully optimized only when a generic runner match sends each player a complete mutable clone of the current engine state on every decision, validates and applies the chosen move through the authoritative Rust engine, records the compact game, and sustains at least 100,000 accepted moves per second across 28 local worker processes on this 7950X / 4070 Ti machine.

The benchmark must use the normal public runner and player interfaces, two reusable long-lived player instances per worker, legal move enumeration from the real engine state, observer notifications enabled, compact record writing enabled, and at least 1,000,000 accepted moves after worker warmup.

A run is invalid if it bypasses player `decide()`, skips state cloning, disables legality validation, suppresses recording, replaces legal move generation with precomputed scripts, uses a benchmark-only engine path, ignores aborted games, or reports throughput before all worker processes have exited cleanly.

The optimized target also requires median per-worker RSS below 512 MB, zero illegal authoritative-state mutations by players, deterministic replay from every emitted record, and no correctness regression in the full Rust and Python test suites.
