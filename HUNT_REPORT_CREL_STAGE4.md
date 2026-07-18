# R-CREL-4 — C_rel Stage 4 scoped library/routing economics

## Verdict

**PASS. No binding kill criterion fired.** The selected cell is the one with the greatest source-clustered 95% lower bound: **8 MiB reservation, fanout 1**, with **90.159% measured net wall gain** and an **83.243% clustered lower bound** over the strongest exact-fragment baseline. The maximum observed paired process-RSS regression over the complete campaign was **0.511%**, below 5%. The maximum accounted peak was **86,080,035 bytes (82.092 MiB)**, below 512 MiB.

The binding criteria, quoted verbatim, resolved as follows:

- “KILL if the source-clustered 95% lower bound of net gain is <= 5% over the strongest exact baseline in EVERY budget/fanout cell.” **Did not fire:** every cell's lower bound was at least 77.248%; the selected cell's was 83.243%.
- “KILL if no bounded fanout pays.” **Did not fire:** every bounded-fanout cell had positive measured net gain; the selected fanout-1 cell gained 90.159%.
- “KILL if accounted cache exceeds 512 MiB in any cell.” **Did not fire:** the maximum observed accounted peak was 86,080,035 bytes.
- “KILL if max observed paired process-RSS regression exceeds 5% (RSS is never compared directly with 512 MiB).” **Did not fire:** the maximum paired regression was 0.511% (`1 MiB × fanout 32`, pair 2).

This is a Stage-4 shadow result, not a production gate candidate. No Stage 5 or Stage 6 experiment was run.

## Fixed protocol and arms

Input provenance is commit `3e4808c6`; the working branch was `hunt/cert-support`. No commit was created.

The campaign rebuilt the frozen acquisition manifest in every process. Consolidated HEAD reproduced 48 admitted templates and 46 eligible `FirstStone` bodies: 45 WIN sources plus the `forced_loss_firststone` LOSS source. The two explicit non-`FirstStone` exclusions remained `hayes_20260712_placement31` and `forced_loss_secondstone`. All K=1/K=2 mutations were constructed, giving 46 source-root clusters and 368 queries in frozen `(source-kind, source-id, K, trial)` order.

Source solves were already demanded queries, so **`G=0`**. Their complete acquisition outcomes remain in every raw for drift checking, but their wall is not charged to either target-query arm. Target cold caps followed the frozen source profile: 10k for ordinary forcing sources, 100k for `zrugh2x`, `strongloss_a_prefix6`, and `hayes_20260712_turn16`, 30k for human roots, and 1 for the hand LOSS fixture. Semantic horizons were identical between paired arms.

The arms were:

- Baseline: current exact fragments enabled, with 512 MiB total solver/fragment capacity; target wall is `S0_j`.
- C_rel: exact fragments enabled, shadow C_rel routing and unchanged strict verification, plus mandatory cold fallback on every miss. Solver/fragment capacity was `512 MiB − reservation`; fallback wall is `SR_j`.

For each reservation, bodies were sorted by `(source-kind, source-id, artifact_id)`. A whole canonical artifact plus its canonical routing-index record was admitted only if it fit; there was no eviction. The 46 artifacts used 561,419 bytes and the index used 3,395 bytes, so **all 46 bodies were admitted and zero refused in every reservation**, including 1 MiB. Candidate probes were independently ordered by `(status, phase, projection_cell_count, artifact_id, g)`; the known parent was never directly selected or privileged.

The lane was strictly shadow-only. Every accepted body was submitted to the unchanged `TssVerifier`; no warm hard value was returned or installed. `hard_without_strict=0` in all 144 process summaries. `packages/hexfield_eq/rust/src/tss_verify.rs` was untouched.

## Statistics

Each cell used six separate serialized processes in order `AB`, `BA`, `AB`. For each source root, the eight mutation-query wall deltas were summed within a pair, one forty-sixth of that C_rel invocation's `E` was charged to the cluster, and the three paired cluster deltas were averaged. Summing the 46 cluster means gives the point estimator and exactly recovers the mean end-to-end equation:

```text
T_base = sum_j S0_j
T_Crel = G + E + sum_j [L_j + sum_i(I_ji + M_ji + V_ji) + (1-A_j) SR_j]
net gain = (T_base - T_Crel) / T_base
```

The interval is a 10,000-resample percentile cluster bootstrap over the 46 source roots, using fixed seed `0xC0DE_C0DE_5EED_0001` and the 2.5th percentile as the reported 95% lower bound. Individual mutations were never resampled as independent observations, and no per-root parameter selection occurred.

## Per-cell results

Accounted peak is shown as baseline/C_rel MiB. RSS delta is the maximum of the three paired process-peak regressions for that cell, not a comparison with 512 MiB. The selected cell is bold.

| Reservation (MiB) | Fanout | Net wall gain | Clustered 95% LB | Accounted peak MiB, base/C_rel | Max paired RSS delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 88.576% | 80.525% | 82.092 / 66.807 | +0.273% |
| 1 | 2 | 89.849% | 82.151% | 82.092 / 66.807 | -0.090% |
| 1 | 4 | 89.418% | 82.549% | 82.092 / 66.807 | -0.062% |
| 1 | 8 | 89.484% | 81.127% | 82.092 / 66.807 | +0.278% |
| 1 | 16 | 88.108% | 77.611% | 82.092 / 66.807 | +0.445% |
| 1 | 32 | 89.611% | 81.102% | 82.092 / 66.807 | +0.511% |
| **8** | **1** | **90.159%** | **83.243%** | **82.092 / 66.797** | **+0.017%** |
| 8 | 2 | 88.943% | 80.721% | 82.092 / 66.797 | +0.118% |
| 8 | 4 | 89.484% | 80.633% | 82.092 / 66.797 | +0.358% |
| 8 | 8 | 87.497% | 77.248% | 82.092 / 66.797 | +0.170% |
| 8 | 16 | 89.960% | 80.126% | 82.092 / 66.797 | +0.187% |
| 8 | 32 | 88.133% | 79.341% | 82.092 / 66.797 | +0.367% |
| 32 | 1 | 89.884% | 81.798% | 82.092 / 66.744 | +0.282% |
| 32 | 2 | 89.511% | 81.043% | 82.092 / 66.744 | +0.085% |
| 32 | 4 | 88.240% | 77.699% | 82.092 / 66.744 | +0.408% |
| 32 | 8 | 89.625% | 81.198% | 82.092 / 66.744 | +0.416% |
| 32 | 16 | 88.862% | 77.590% | 82.092 / 66.744 | +0.120% |
| 32 | 32 | 89.227% | 82.588% | 82.092 / 66.744 | +0.005% |
| 64 | 1 | 89.804% | 82.094% | 82.092 / 66.686 | +0.055% |
| 64 | 2 | 90.439% | 82.191% | 82.092 / 66.686 | +0.321% |
| 64 | 4 | 88.336% | 79.719% | 82.092 / 66.686 | +0.463% |
| 64 | 8 | 89.603% | 81.052% | 82.092 / 66.686 | +0.430% |
| 64 | 16 | 89.749% | 80.574% | 82.092 / 66.686 | -0.055% |
| 64 | 32 | 89.214% | 79.126% | 82.092 / 66.686 | +0.122% |

## Selected-cell cost decomposition

The selected `8 MiB × fanout 1` values are three-invocation means.

| Component | Mean wall |
|---|---:|
| `S0`, strongest exact-fragment baseline | 156.306861 s |
| `G`, source solve incremental charge | 0 s |
| `E`, extraction/serialization/index build | 0.147025 s |
| `L`, lookup | 0.009625 s |
| `I`, every performed finite-interface check | 0.003886 s |
| `M`, deterministic materialization | 0.007778 s |
| `V`, unchanged strict verification | 1.001068 s |
| `SR`, mandatory residual-budget cold fallbacks | 14.212218 s |
| Complete C_rel path, `G+E+L+I+M+V+SR` | 15.381600 s |
| Net saving | 140.925261 s (90.159%) |

The selected cell strict-accepted 350/368 targets (95.109%) and cold-fell back on 18/368. It performed 98,258 hint checks and 368 strict probes. Every accepted target's first-accepted rank was 1. Baseline diagnostic work was 1,039,733 expansions; fallback diagnostic work was 110,750 expansions. These expansion counts do not enter the decision.

Memory decomposition at the selected cell:

- Baseline observed local-TT peak: 57,709,371 bytes; exact-fragment peak: 28,370,664 bytes; accounted peak: 86,080,035 bytes.
- C_rel fallback observed local-TT peak: 28,148,182 bytes; exact-fragment peak: 2,368,052 bytes; artifact: 561,419 bytes; index: 3,395 bytes.
- The unchanged verifier's full 67,108,864-byte memo cap was conservatively charged during probe phases. Verifier temporaries and fallback local TT do not overlap, so the C_rel accounted peak is the larger of the fallback phase and verifier phase: 70,041,730 bytes.
- The C_rel solver/fragment cap was 528,482,304 bytes (504 MiB), exactly 8 MiB below baseline.

The selected cell's mean process peaks were 164,089,856 bytes baseline and 163,898,709 bytes C_rel. Its maximum paired regression was +0.017%. Across the entire campaign the maximum was +0.511%, from 163,516,416 to 164,352,000 bytes. Process RSS is reported independently and is never compared directly with 512 MiB.

## Exact-fragment comparator

The retained 139-root comparator reported exact-fragment cold overhead of +0.809% eager and +0.989% lazy, then warm gains of 16.476% eager and 16.322% lazy, with 124,725 expansions saved. Stage 4 enabled the same current exact-fragment path in both arms, so its 90.159% selected-cell net gain (83.243% lower bound) is incremental over exact fragments rather than a replacement comparison. C_rel therefore beats the exact-fragment baseline on this scoped K=1/K=2 cohort.

This is not a cross-corpus claim: the old comparator used 139 retained roots, whereas Stage 4 uses 368 deterministic mutations clustered under 46 source roots. The valid statement is that C_rel adds a large measured saving after exact fragments are already enabled under the Stage-4 fixed envelope.

## Resource-law audit

All Cargo commands used `CARGO_TARGET_DIR=.target-hunt`, target `x86_64-pc-windows-msvc`, release mode, and serial tests. The 144 statistical letters were separate test-binary processes, one at a time. There were 158 resource readings: 14 low-available-memory readings caused waits and rechecks. At actual launch, the minimum available memory was 10,760,761,344 bytes, minimum free physical memory was 10,754,404,352 bytes, and no foreign `cargo.exe` was present. The maximum process invocation was 259.893 s, below ten minutes. Campaign execution was split into chunks of at most 100.9 minutes.

## Caveats and residual question

- The frozen library again contained zero zoned `Universal` declarations. Zone-hint extraction and correspondence remain structural-only; this campaign supplies no positive zoned-certificate economics.
- Every accepted target had first-accepted rank 1. The fanout sweep therefore mostly measures extra failed probes on the 18 misses, not competition among multiple useful bodies.
- All 46 records fit in 1 MiB, so reservation primarily changed the residual solver cap. This campaign does not locate a useful capacity-refusal frontier.
- Process peak RSS covers the whole invocation, including demanded source acquisition. That is conservative and paired, but it can mask a smaller target-phase-only RSS change.
- The verifier temporary charge is its unchanged 64 MiB cap, not an instrumented allocation trace. This makes cache accounting conservative.
- Shadow evidence was never installed. Passing Stage 4 authorizes no production deployment and does not close Stage 5/6 or proof obligations.

**Deploy recommendation:** do not deploy from Stage 4; carry only the `8 MiB × fanout 1` cell into the authorized Phase-3 leaf-relevance workload measurement, still shadow/default-off until later gates close.

**Sharpest residual question:** does the 83.243% lower-bound gain persist on a frozen Phase-3 leaf-relevance cohort where useful bodies are not merely mutations of their source roots and where zoned certificates exercise the currently structural-only hint path?

## Authoritative raw evidence and SHA-256

Each cell raw contains six process invocations, all 2,208 query rows, all acquisition outcomes, resource gates, summaries, and process exits. `CREL_STAGE4_ANALYSIS_RAW.log` contains every paired aggregate, all 24 bootstrap results, the fixed-seed verdict, and the launch-gate audit.

| Raw | Bytes | SHA-256 |
|---|---:|---|
| `CREL_STAGE4_ANALYSIS_RAW.log` | 17,935 | `A013FBA1EB9DCFA5F548EF20DDB1EE0E5B582CEC4D4F9DA16F9CFE1D86094A18` |
| `CREL_STAGE4_BUILD_RAW.log` | 775 | `7778DF8FDE1D1AB3C8EB902D527C5412AADF0707CDD3562CD0CFE9866DDFFB1C` |
| `CREL_STAGE4_FMT_RAW.log` | 280 | `ED48F95C253821BA7A1E32520ACACB9723A71703CFCC4E6A9D4CA49726C7566B` |
| `CREL_STAGE4_R1_F1_RAW.log` | 1,035,792 | `15F4230D08B67E743D57E8CB2BF3E61A4127568A3B0E1D21EC5713AA5B5CF8A3` |
| `CREL_STAGE4_R1_F2_RAW.log` | 1,036,180 | `969ECB13B5657F851FDD0DE08F395AF26B98FE75481E57155C246090BA4911A7` |
| `CREL_STAGE4_R1_F4_RAW.log` | 1,035,901 | `CB2449355706D7F6A9051CC04DADF76662A77F9BE09F0F8F41645CD4463BD900` |
| `CREL_STAGE4_R1_F8_RAW.log` | 1,035,931 | `D23A558704FBCA0DA18B23EC5CB0A77852A7DAFBE2AD97374D508EBA601E73D5` |
| `CREL_STAGE4_R1_F16_RAW.log` | 1,036,004 | `E73D3F8F5C2C5332C153E47D30EAFD8D94FA6F789BD70356727129E3953623AC` |
| `CREL_STAGE4_R1_F32_RAW.log` | 1,035,887 | `B5560AD4D6D93B83C37E47ED6D035A4BAEE2DBAD134627312494ADBD82170859` |
| `CREL_STAGE4_R8_F1_RAW.log` | 1,035,831 | `344E3C22C96EC7EE3D4FC55F9A59E174911AAC77DBC8315C3934A8E9B7D37643` |
| `CREL_STAGE4_R8_F2_RAW.log` | 1,035,886 | `9F9F6C7E8846FE8B32851957B9A2FA15F41DBE4ECBE491FB05393ECCDC1551DA` |
| `CREL_STAGE4_R8_F4_RAW.log` | 1,035,888 | `618CFD25B5FB1F88BAB1911CD6778EE3CD3D40CA162CBD2937D4F5A64E85E693` |
| `CREL_STAGE4_R8_F8_RAW.log` | 1,035,996 | `F7E2EDC34945A97B0BE058D3456B09F82CF6C4E983FA82A42A2DA3DA939D5C9E` |
| `CREL_STAGE4_R8_F16_RAW.log` | 1,035,912 | `FDB979E5139F21695CEA0763727989E4C65F1B7B622024B44C7E1AC2673F2E37` |
| `CREL_STAGE4_R8_F32_RAW.log` | 1,035,806 | `6B7B1644A3DFC2F5E7438C32A15075C2DBA58B7F0208155226365C397198EB51` |
| `CREL_STAGE4_R32_F1_RAW.log` | 1,036,314 | `2A7EBC8CB7F0FC3C3C3D485AD72054FDFC0E924A013C679D089029A0F4BEBD78` |
| `CREL_STAGE4_R32_F2_RAW.log` | 1,036,201 | `C0CB224A4D7CD2C60F3816E3874FE5BC76BAF4A5298A1CA10CA1945E109CF8D6` |
| `CREL_STAGE4_R32_F4_RAW.log` | 1,036,563 | `131F731D47C21D319C7862DCFB472147AD2B8215A6CA34C191C9B11857C1A74F` |
| `CREL_STAGE4_R32_F8_RAW.log` | 1,036,121 | `132038C6085F125414F3F5D1BE69B0153165AEBD4265FF2A83ED5D2E36CDE44C` |
| `CREL_STAGE4_R32_F16_RAW.log` | 1,036,322 | `2EB187250E13881557337AAA6BE28432384DD9C1CAAF9B62EB7C2F39B55A7895` |
| `CREL_STAGE4_R32_F32_RAW.log` | 1,036,303 | `50427E2CEDFA2697B4E191BFE9508DCB2D2DE8E0341E04950644CA0736883E5D` |
| `CREL_STAGE4_R64_F1_RAW.log` | 1,035,816 | `80CD82F0917AD93C990AF1E11BC982C049E40B96B2E2EC6F3DFC856FBF52D4EC` |
| `CREL_STAGE4_R64_F2_RAW.log` | 1,035,913 | `AD3B94F8D71878A0CE56E2F940AD5CFE9FBC5B1EDA951207AD39BF24483B9A8F` |
| `CREL_STAGE4_R64_F4_RAW.log` | 1,035,973 | `235918240A23943DDEA7B3DD91DE5D9B7E97E2959B628D2A29FA24269EF96207` |
| `CREL_STAGE4_R64_F8_RAW.log` | 1,036,014 | `C7319CEC0AC2801A4B64FE3540FA6FEC91A11D9035D65CFE64B6C5DAB31B603E` |
| `CREL_STAGE4_R64_F16_RAW.log` | 1,036,179 | `97FCC90A87D5E33D46BCFAE2773E3760BAC98C701E2F2865535EC651E7A3D67A` |
| `CREL_STAGE4_R64_F32_RAW.log` | 1,035,919 | `97A4572F31ADC509242FCAF0E3423040D6ED9BF46F3DC4C54589FFBCCB2252F3` |

The machine-readable manifest is `CREL_STAGE4_HASHES_RAW.log`, 3,369 bytes, SHA-256 `F542528FDEB6ECA4CDEB98CD22DFE34C17E3C229F47F1CCE65DC581CA63B8292`.
