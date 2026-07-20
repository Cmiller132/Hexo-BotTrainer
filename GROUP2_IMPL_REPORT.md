# GROUP2 implementation report

## Outcome

**STOPPED AT THE PRE-IMPLEMENTATION SOUNDNESS BOUNDARY. Phase A is not
green and Phase B was not started.**

The frozen spec cannot be implemented simultaneously with the requirement
that `tss_verify.rs` remain byte-for-byte unchanged. In addition, the frozen
authority commit and SHA-256 identify different repository blobs. The exact
evidence is recorded in `.codex-group2-next/OBSTRUCTIONS.md`.

No Rust source was changed. In particular, the strict verifier remains at
SHA-256
`9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`.

## Implementation map

No implementation was made. The audited seams were:

- certificate grammar: `packages/hexfield_eq/rust/src/tss_verify.rs:100-153`;
- independent uniform-zone reconstruction: `tss_verify.rs:1094-1150`;
- mandatory coverage check: `tss_verify.rs:1216-1268`;
- current finder closure: `tss_solver.rs:7804-7947` and
  `tss_solver.rs:9094-9150`;
- frozen class-tag/FHW requirements: `DESIGN_GROUP2_NEXT.md:247-266`,
  `:585-604`, and `:887-966`.

## Tests and measurements

- Strict-verifier worktree diff: PASS (unchanged).
- Toolchain/target audit: PASS (`x86_64-pc-windows-msvc`).
- Unit/property/cargo tests: NOT RUN; an implementation satisfying the
  frozen interfaces cannot be represented.
- Promotion battery: NOT RUN.
- All-19 2 GiB gate: NOT RUN.
- Verdict identity: NOT MEASURED; therefore no verdict flip occurred or is
  claimed.
- Economics gates: NOT EVALUABLE.

No TT >=512 MiB command was started, so the host-memory wait protocol was
not triggered.

## Honest residual list

1. Resolve the authority commit/digest mismatch.
2. Resolve the verifier/certificate representation contradiction.
3. Implement the selector and all mandatory charge/guard tests.
4. Implement and freeze `J_zone`, `I_FHW`, and `J_zone^FHW` bindings.
5. Run the complete Phase A suite with `CARGO_TARGET_DIR=.target-g2` and
   `--target x86_64-pc-windows-msvc`.
6. Only then run the frozen Phase B battery, memory-gated all-19 corpus gate,
   raw-log manifest, and economics evaluation.

## Artifacts

- `.codex-group2-next/BASELINE.md`
- `.codex-group2-next/OBSTRUCTIONS.md`
- `.codex-group2-next/SHA256SUMS.txt`

The pre-existing untracked `.codex-g2/` directory was not modified.
