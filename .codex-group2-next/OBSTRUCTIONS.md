# R-G2-IMPL preregistered obstruction

Status: **BLOCKED BEFORE IMPLEMENTATION**. No selector was enabled and no promotion result was inspected.

## O1: the frozen authority commit and authority digest identify different blobs

`DESIGN_GROUP2_NEXT.md` names `docs/PROOF_TSS_DEFENDER_ZONES.md` at commit
`148536cd` with SHA-256
`39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B`.
It also says a mismatch makes the lane `INELIGIBLE`.

The objects available in this repository are:

```text
148536cd blob c6788a114c51b16bc69bb6396ef04d434665e275
           SHA-256 121F94C93630A42EA4D4D215F98865451C87AF5D95209E7BEFDAF8BE9551EBFF
           2,007 logical lines

6dc08d7a blob 05da6f39b4af8523ef1884a53e00d1ae84c9e9f2
           SHA-256 39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B
           2,012 logical lines
```

Thus the frozen digest exists, but at `6dc08d7a`, not at the frozen
`148536cd` authority commit. Choosing either side would rewrite the
preregistered authority identity after seeing the mismatch.

## O2: material FHW restriction cannot pass the byte-unchanged verifier

The frozen build requires an FHW class tag and sufficient verifier data
(`DESIGN_GROUP2_NEXT.md:266`), a per-edge FHW-T3-R decision-tree row
(`:601`), and independent re-verification of a refined set that is a subset
of the uniform set (`:597`). A positive materiality result requires the
variant to be strictly smaller than the uniform wrapper (`:912-958`).

The normative certificate grammar is defined in the file that must remain
byte-for-byte untouched:

- `tss_verify.rs:100`: `ZoneInfo` contains only `d` and `build_horizon`.
- `tss_verify.rs:108`: `CertNode::Universal` has only edges, the legacy
  dispatch bit, `Option<ZoneInfo>`, and commutations.
- `tss_verify.rs:144`: `TssCertificate` has no class tag, D22 map, RC/WC
  evidence, role/window clocks, authority digests, child-plan digest, or
  finder-summary digest.
- `tss_verify.rs:1259-1261`: the verifier independently reconstructs the
  uniform zone and accepts only when every uniform move occurs in the
  explicit edge set.

Let `U` be that reconstructed uniform set and `F` the requested FHW set.
The frozen materiality bar requires `F` to be a proper subset of `U` on at
least one promoted node. The unchanged verifier requires `U` to be a subset
of the explicit searched edges. If the selector searches exactly `F`, the
certificate is rejected. If the finder adds `U\\F`, the certificate may pass,
but defender enumeration is no longer FHW-restricted and its positive gain
has not been consumed.

There is also no byte-preserving place to serialize the mandatory guard
outcomes or the frozen `J_zone`/`I_FHW` bindings for the verifier to check.
A finder-only sidecar cannot satisfy the frozen requirement that the strict
verifier independently validate the class.

## Why fail-open does not repair the contract

Falling back to existing full or uniform enumeration is sound, but it makes
every materially narrowed FHW selection ineligible or unconsumed. A flag
whose only conforming behavior is fallback/no-op does not implement the
requested Group-2 FHW selector and cannot honestly enter the promotion
battery.

## Required owner decision

Both frozen identities need an explicit amendment before implementation:

1. choose the authority pair (`148536cd` plus its actual digest, or
   `6dc08d7a` plus digest `39197460...064B`); and
2. choose either (a) permit a reviewed change to the certificate grammar and
   strict verifier so D22/RC/WC evidence is independently checked, or (b)
   redefine this round as shadow-only FHW telemetry with uniform certificate
   consumption and remove FHW Consume promotion from its definition of done.

No radius-nine substitution, population inference, verdict claim, or
economics claim was made.
