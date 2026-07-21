# Residue map aggregate tables

Row p90/p95/max values are computed after taking each logical job's median share across its three repetitions.

## F19 protocol wall (34 logical jobs; median 483872.317 ms)

| category | median sum ms | wall % | p90 row % | p95 row % | max row %/id | disposition | measured value estimate | estimate method | eliminability upper bound |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| D_FORCED_GEN | 174223.582 | 36.006107 | 36.758912 | 38.140333 | 39.550560/f19/0l4291i_live/cap20000000/unbounded | implemented/local | NOT_MEASURED | ceiling only | 36.006107 |
| D_UNFORCED_FHW_ELIGIBLE_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | unobserved/no selector | NOT_MEASURED | no eligible events | 0.000000 |
| D_UNFORCED_NONFHW_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | unobserved/no selector | NOT_MEASURED | no classified events | 0.000000 |
| D_UNFORCED_UNCLASSIFIED_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | OPEN/audit debt | NOT_MEASURED | ceiling only | 0.000000 |
| A_OR_GEN | 222188.374 | 45.918802 | 65.234707 | 67.960161 | 70.368010/f19/hu01jk4/cap10000/unbounded | implemented/local | NOT_MEASURED | ceiling only | 45.918802 |
| A_OR_WINNER_PATH | 4497.675 | 0.929517 | 4.564763 | 5.455545 | 5.462116/f19/acly7kb/cap10000/unbounded | productive ceiling | NOT_MEASURED | ceiling only | 0.929517 |
| A_OR_ORDERING_MISS | 8870.335 | 1.833197 | 7.887807 | 12.041170 | 15.134228/f19/strongloss_b_prefix8/cap10000/unbounded | OPEN | measured wall | direct exclusive wall | 1.833197 |
| A_OR_UNRESOLVED | 58772.403 | 12.146263 | 25.482935 | 33.032769 | 33.514462/f19/l9mxn59/cap100000/unbounded | OPEN | NOT_MEASURED | ceiling only | 12.146263 |
| TT_PROBE | 2930.948 | 0.605728 | 0.465673 | 0.627870 | 0.704511/f19/0l4291i_live/cap20000000/unbounded | OPEN/local | NOT_MEASURED | ceiling only | 0.605728 |
| TT_STORE | 4426.020 | 0.914708 | 0.948676 | 1.025891 | 1.056417/f19/lz60mfb/cap100000/unbounded | OPEN/local | NOT_MEASURED | ceiling only | 0.914708 |
| CENSUS_GATE | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | implemented/gated | NOT_MEASURED | ceiling only | 0.000000 |
| SEARCH_BOOKKEEPING | 4682.664 | 0.967748 | 3.985743 | 5.715745 | 9.254750/f19/xsnfyll/cap10000/unbounded | OPEN | NOT_MEASURED | ceiling only | 0.967748 |
| CERT_BUILD | 2926.305 | 0.604768 | 4.452423 | 8.965988 | 9.000386/f19/hu01jk4/cap10000/unbounded | verified/mandatory | NOT_MEASURED | ceiling only | 0.604768 |
| CERT_VERIFY | 356.354 | 0.073646 | 4.253412 | 43.718593 | 44.791667/f19/dy3dg99/cap10000/unbounded | verified/mandatory | NOT_MEASURED | ceiling only | 0.073646 |
| HORIZON_LADDER_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| CAP_RESUME_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap100000/unbounded | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| OTHER_MEASURED | 0.379 | 0.000078 | 0.081558 | 55.208333 | 56.281407/f19/8is963b/cap10000/unbounded | audit debt | measured wall | direct timer | 0.000078 |

## F19 final attempts (19 logical jobs; median 335452.817 ms)

| category | median sum ms | wall % | p90 row % | p95 row % | max row %/id | disposition | measured value estimate | estimate method | eliminability upper bound |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| D_FORCED_GEN | 121328.592 | 36.168601 | 37.045832 | 39.550560 | 39.550560/f19/0l4291i_live/cap20000000/unbounded | implemented/local | NOT_MEASURED | ceiling only | 36.168601 |
| D_UNFORCED_FHW_ELIGIBLE_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | unobserved/no selector | NOT_MEASURED | no eligible events | 0.000000 |
| D_UNFORCED_NONFHW_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | unobserved/no selector | NOT_MEASURED | no classified events | 0.000000 |
| D_UNFORCED_UNCLASSIFIED_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | OPEN/audit debt | NOT_MEASURED | ceiling only | 0.000000 |
| A_OR_GEN | 151554.456 | 45.179068 | 65.510507 | 70.368010 | 70.368010/f19/hu01jk4/cap10000/unbounded | implemented/local | NOT_MEASURED | ceiling only | 45.179068 |
| A_OR_WINNER_PATH | 2942.483 | 0.877167 | 5.441545 | 5.462116 | 5.462116/f19/acly7kb/cap10000/unbounded | productive ceiling | NOT_MEASURED | ceiling only | 0.877167 |
| A_OR_ORDERING_MISS | 5940.890 | 1.771006 | 12.041170 | 15.134228 | 15.134228/f19/strongloss_b_prefix8/cap10000/unbounded | OPEN | measured wall | direct exclusive wall | 1.771006 |
| A_OR_UNRESOLVED | 42194.013 | 12.578226 | 17.082307 | 33.032769 | 33.032769/f19/l9mxn59/cap1000000/unbounded | OPEN | NOT_MEASURED | ceiling only | 12.578226 |
| TT_PROBE | 2113.851 | 0.630148 | 0.566857 | 0.704511 | 0.704511/f19/0l4291i_live/cap20000000/unbounded | OPEN/local | NOT_MEASURED | ceiling only | 0.630148 |
| TT_STORE | 3114.430 | 0.928426 | 1.020486 | 1.025891 | 1.025891/f19/jnzzmcm/cap10000/unbounded | OPEN/local | NOT_MEASURED | ceiling only | 0.928426 |
| CENSUS_GATE | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | implemented/gated | NOT_MEASURED | ceiling only | 0.000000 |
| SEARCH_BOOKKEEPING | 3413.922 | 1.017705 | 5.715745 | 9.254750 | 9.254750/f19/xsnfyll/cap10000/unbounded | OPEN | NOT_MEASURED | ceiling only | 1.017705 |
| CERT_BUILD | 2340.763 | 0.697792 | 8.965988 | 9.000386 | 9.000386/f19/hu01jk4/cap10000/unbounded | verified/mandatory | NOT_MEASURED | ceiling only | 0.697792 |
| CERT_VERIFY | 356.354 | 0.106231 | 43.718593 | 44.791667 | 44.791667/f19/dy3dg99/cap10000/unbounded | verified/mandatory | NOT_MEASURED | ceiling only | 0.106231 |
| HORIZON_LADDER_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| CAP_RESUME_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/f19/mvp2lvc/cap1000000/unbounded | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| OTHER_MEASURED | 0.233 | 0.000069 | 55.208333 | 56.281407 | 56.281407/f19/8is963b/cap10000/unbounded | audit debt | measured wall | direct timer | 0.000069 |

## S2 (2 logical jobs; median 0.480 ms)

| category | median sum ms | wall % | p90 row % | p95 row % | max row %/id | disposition | measured value estimate | estimate method | eliminability upper bound |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| D_FORCED_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | implemented/local | NOT_MEASURED | ceiling only | 0.000000 |
| D_UNFORCED_FHW_ELIGIBLE_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | unobserved/no selector | NOT_MEASURED | no eligible events | 0.000000 |
| D_UNFORCED_NONFHW_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | unobserved/no selector | NOT_MEASURED | no classified events | 0.000000 |
| D_UNFORCED_UNCLASSIFIED_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | OPEN/audit debt | NOT_MEASURED | ceiling only | 0.000000 |
| A_OR_GEN | 0.375 | 78.187500 | 78.736842 | 78.736842 | 78.736842/s2/compact_urgent_spare/cap1000000/base | implemented/local | NOT_MEASURED | ceiling only | 78.187500 |
| A_OR_WINNER_PATH | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | productive ceiling | NOT_MEASURED | ceiling only | 0.000000 |
| A_OR_ORDERING_MISS | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | OPEN | measured wall | direct exclusive wall | 0.000000 |
| A_OR_UNRESOLVED | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | OPEN | NOT_MEASURED | ceiling only | 0.000000 |
| TT_PROBE | 0.002 | 0.354167 | 0.397661 | 0.397661 | 0.397661/s2/compact_urgent_spare/cap1000000/base | OPEN/local | NOT_MEASURED | ceiling only | 0.354167 |
| TT_STORE | 0.006 | 1.208333 | 1.286550 | 1.286550 | 1.286550/s2/compact_urgent_spare/cap1000000/base | OPEN/local | NOT_MEASURED | ceiling only | 1.208333 |
| CENSUS_GATE | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | implemented/gated | NOT_MEASURED | ceiling only | 0.000000 |
| SEARCH_BOOKKEEPING | 0.065 | 13.520833 | 15.619048 | 15.619048 | 15.619048/s2/strongloss_a_backoff_7/cap1000000/base | OPEN | NOT_MEASURED | ceiling only | 13.520833 |
| CERT_BUILD | 0.008 | 1.645833 | 1.732321 | 1.732321 | 1.732321/s2/compact_urgent_spare/cap1000000/base | verified/mandatory | NOT_MEASURED | ceiling only | 1.645833 |
| CERT_VERIFY | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | verified/mandatory | NOT_MEASURED | ceiling only | 0.000000 |
| HORIZON_LADDER_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| CAP_RESUME_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/s2/strongloss_a_backoff_7/cap1000000/base | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| OTHER_MEASURED | 0.023 | 4.729167 | 4.904695 | 4.904695 | 4.904695/s2/compact_urgent_spare/cap1000000/base | audit debt | measured wall | direct timer | 4.729167 |

## Human 160 (160 logical jobs; median 156709.042 ms)

| category | median sum ms | wall % | p90 row % | p95 row % | max row %/id | disposition | measured value estimate | estimate method | eliminability upper bound |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| D_FORCED_GEN | 271.706 | 0.173383 | 0.277080 | 0.717362 | 3.464932/human160/human_a4feec1337a8570d_p138/cap50000/base_rel10 | implemented/local | NOT_MEASURED | ceiling only | 0.173383 |
| D_UNFORCED_FHW_ELIGIBLE_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | unobserved/no selector | NOT_MEASURED | no eligible events | 0.000000 |
| D_UNFORCED_NONFHW_GEN | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | unobserved/no selector | NOT_MEASURED | no classified events | 0.000000 |
| D_UNFORCED_UNCLASSIFIED_GEN | 63398.570 | 40.456230 | 49.086909 | 53.113044 | 63.839269/human160/human_0b39ea1a6c8eee20_p198/cap50000/base_rel10 | OPEN/audit debt | NOT_MEASURED | ceiling only | 40.456230 |
| A_OR_GEN | 173.878 | 0.110956 | 0.186903 | 0.225844 | 0.593414/human160/human_bbdce53d470eb3fa_p34/cap50000/base_rel10 | implemented/local | NOT_MEASURED | ceiling only | 0.110956 |
| A_OR_WINNER_PATH | 0.475 | 0.000303 | 0.000000 | 0.005371 | 0.413605/human160/human_bbdce53d470eb3fa_p34/cap50000/base_rel10 | productive ceiling | NOT_MEASURED | ceiling only | 0.000303 |
| A_OR_ORDERING_MISS | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | OPEN | measured wall | direct exclusive wall | 0.000000 |
| A_OR_UNRESOLVED | 61775.218 | 39.420328 | 54.553240 | 56.323300 | 89.667487/human160/human_839fee4ea8f55074_p16/cap50000/base_rel10 | OPEN | NOT_MEASURED | ceiling only | 39.420328 |
| TT_PROBE | 2326.662 | 1.484702 | 1.718118 | 1.759178 | 3.012741/human160/human_fe3ffa0933af690c_p25/cap50000/base_rel10 | OPEN/local | NOT_MEASURED | ceiling only | 1.484702 |
| TT_STORE | 1144.655 | 0.730433 | 0.844294 | 0.897800 | 0.985941/human160/human_0b39ea1a6c8eee20_p198/cap50000/base_rel10 | OPEN/local | NOT_MEASURED | ceiling only | 0.730433 |
| CENSUS_GATE | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | implemented/gated | NOT_MEASURED | ceiling only | 0.000000 |
| SEARCH_BOOKKEEPING | 26318.861 | 16.794731 | 19.735062 | 21.220173 | 96.027061/human160/human_9790adbbfeea6c17_p24/cap50000/base_rel10 | OPEN | NOT_MEASURED | ceiling only | 16.794731 |
| CERT_BUILD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | verified/mandatory | NOT_MEASURED | ceiling only | 0.000000 |
| CERT_VERIFY | 0.480 | 0.000307 | 0.000000 | 0.198047 | 43.402778/human160/human_f4527f814c42f2f7_p20/cap50000/base_rel10 | verified/mandatory | NOT_MEASURED | ceiling only | 0.000307 |
| HORIZON_LADDER_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| CAP_RESUME_OVERHEAD | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.000000/human160/human_eb80339fce963cd0_p15/cap50000/base_rel10 | protocol | NOT_MEASURED | ceiling only | 0.000000 |
| OTHER_MEASURED | 1315.947 | 0.839739 | 1.052155 | 3.808505 | 64.259928/human160/human_b5bfa0cdcfe9b56b_p21/cap50000/base_rel10 | audit debt | measured wall | direct timer | 0.839739 |

