window.__ATLAS__ = {
  "schema": 1,
  "generated_from": "db96d1b136021212ef32e1f1fdf747bc2262e1c7",
  "census": {
    "ply2_raw": 216,
    "ply2_d6": 24,
    "ply3_raw": 42768,
    "ply3_d6": 3684
  },
  "summary": {
    "total": 122,
    "win": 37,
    "loss": 6,
    "unknown": 79,
    "certified": 43
  },
  "sharp_examples": [
    {
      "kind": "verdict_flip",
      "game": "004759ff34cefdc2",
      "corpus_winner": "P1",
      "description": "Adjacent certificate-backed verdict flip: the proven winner changes from P0 to P1 after a single placement, so (14,-3) is a certified losing blunder in this exact opening.",
      "flip_move": [
        14,
        -3
      ],
      "source_ply": 44,
      "before": {
        "prefix": 44,
        "side": "P0",
        "phase": "SecondStone",
        "verdict": "CERTIFIED P0 WIN",
        "nodes": 2,
        "derived_horizon": 49
      },
      "after": {
        "prefix": 45,
        "side": "P1",
        "phase": "FirstStone",
        "verdict": "CERTIFIED P1 WIN",
        "nodes": 1,
        "derived_horizon": 47
      }
    },
    {
      "kind": "compact_win",
      "source": "xsnfyll",
      "description": "Compact 13-stone P1 win, certified in only 82 nodes at the 10k rung.",
      "stones": 13,
      "side": "P1",
      "verdict": "CERTIFIED WIN",
      "nodes": 82,
      "rung": 10000,
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -2,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -2
        ],
        [
          1,
          -3
        ],
        [
          0,
          -3
        ],
        [
          2,
          -5
        ],
        [
          2,
          -4
        ],
        [
          1,
          -4
        ],
        [
          3,
          -4
        ],
        [
          3,
          -2
        ]
      ]
    },
    {
      "kind": "certified_loss",
      "sources": [
        "8is963b",
        "dy3dg99"
      ],
      "description": "Genuine dual results: both are P0-to-move CERTIFIED LOSS roots, each resolved in one solver node (not merely NO/UNKNOWN controls).",
      "side": "P0",
      "verdict": "CERTIFIED LOSS",
      "nodes": 1
    },
    {
      "kind": "deepest_new_proof",
      "id": "oa-558f79a590c31b6a",
      "game": "002f5360162bac9b",
      "prefix": 48,
      "description": "Deepest new proof by node count: P0 to move at SecondStone, CERTIFIED WIN in 6,619 nodes (18 certificate nodes, derived T=57); preceding prefix 47 is also a P0 win but needs only 148 nodes.",
      "side": "P0",
      "phase": "SecondStone",
      "verdict": "CERTIFIED WIN",
      "nodes": 6619,
      "cert_nodes": 18,
      "derived_horizon": 57
    }
  ],
  "rows": [
    {
      "id": "oa-c38a6c3bd98476fb",
      "source": "shallow:empty",
      "source_prefix": 0,
      "placements": 0,
      "side": "P0",
      "phase": "Opening",
      "orbit": 1,
      "cap": 100000,
      "horizon": 12,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 60,
      "ms": 0.111,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": []
    },
    {
      "id": "oa-9a50a451078d31d6",
      "source": "shallow:origin",
      "source_prefix": 1,
      "placements": 1,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 1,
      "cap": 100000,
      "horizon": 13,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 62,
      "ms": 0.015,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ]
      ]
    },
    {
      "id": "oa-482abd5c64c0ce97",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.009,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -8,
          0
        ]
      ]
    },
    {
      "id": "oa-f25d526168a338c3",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -8,
          1
        ]
      ]
    },
    {
      "id": "oa-c52d48821351a8fb",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -8,
          2
        ]
      ]
    },
    {
      "id": "oa-2e244728690d2657",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -8,
          3
        ]
      ]
    },
    {
      "id": "oa-bd2258627b6dfb2f",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -8,
          4
        ]
      ]
    },
    {
      "id": "oa-13a35c3087151e73",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -7,
          0
        ]
      ]
    },
    {
      "id": "oa-3460e9f1a099e7ef",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -7,
          1
        ]
      ]
    },
    {
      "id": "oa-c138d5ed5c176bcb",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -7,
          2
        ]
      ]
    },
    {
      "id": "oa-a867af1ad109e047",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -7,
          3
        ]
      ]
    },
    {
      "id": "oa-7a3e5a4a5a60c207",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -6,
          0
        ]
      ]
    },
    {
      "id": "oa-e18f79777c811c5f",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -6,
          1
        ]
      ]
    },
    {
      "id": "oa-617df6eb4b49a88b",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -6,
          2
        ]
      ]
    },
    {
      "id": "oa-1760597455ccbe23",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -6,
          3
        ]
      ]
    },
    {
      "id": "oa-8ecd5c0035569c8b",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -5,
          0
        ]
      ]
    },
    {
      "id": "oa-7bc5c795b96294d3",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -5,
          1
        ]
      ]
    },
    {
      "id": "oa-d320d4e7e826f38b",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -5,
          2
        ]
      ]
    },
    {
      "id": "oa-ea2a0ab866ac0a4f",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -4,
          0
        ]
      ]
    },
    {
      "id": "oa-d8d80de5ab322833",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -4,
          1
        ]
      ]
    },
    {
      "id": "oa-dfa168c17da282f3",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -4,
          2
        ]
      ]
    },
    {
      "id": "oa-653dbed0a1e148e3",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-5f1061038f276897",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-2c12add68f48426f",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -2,
          0
        ]
      ]
    },
    {
      "id": "oa-eb63b78334b2e0c7",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-f15ece8564ca6f2b",
      "source": "shallow:first-reply",
      "source_prefix": 2,
      "placements": 2,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 6,
      "cap": 100000,
      "horizon": 14,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 66,
      "ms": 0.006,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          0
        ]
      ]
    },
    {
      "id": "oa-39674c790c5437da",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 78,
      "placements": 78,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 90,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.013,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 79,
      "cert_fnv1a64_debug_v1": "99fb25ae232ea434",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          0,
          -2
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          0,
          -3
        ],
        [
          -1,
          2
        ],
        [
          1,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          -3
        ],
        [
          2,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          1
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          2,
          -3
        ],
        [
          1,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -1
        ],
        [
          3,
          0
        ],
        [
          3,
          1
        ],
        [
          4,
          -1
        ],
        [
          4,
          0
        ],
        [
          4,
          -3
        ],
        [
          -2,
          3
        ],
        [
          -3,
          4
        ],
        [
          -4,
          5
        ],
        [
          -2,
          2
        ],
        [
          4,
          -4
        ],
        [
          5,
          -5
        ],
        [
          6,
          -6
        ],
        [
          3,
          -4
        ],
        [
          -3,
          3
        ],
        [
          4,
          -5
        ],
        [
          3,
          -5
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -6,
          3
        ],
        [
          -1,
          3
        ],
        [
          -3,
          5
        ],
        [
          6,
          -5
        ],
        [
          7,
          -5
        ],
        [
          -3,
          6
        ],
        [
          -4,
          6
        ],
        [
          2,
          2
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          2
        ],
        [
          -5,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          0
        ],
        [
          -2,
          4
        ],
        [
          -5,
          7
        ],
        [
          -6,
          8
        ],
        [
          2,
          -4
        ],
        [
          4,
          -6
        ],
        [
          4,
          -7
        ],
        [
          4,
          -8
        ],
        [
          -4,
          1
        ],
        [
          -7,
          3
        ],
        [
          -8,
          4
        ],
        [
          -9,
          5
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          -2
        ],
        [
          0,
          4
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          4
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-ee5eae78c9a82d05",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 77,
      "placements": 77,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 89,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.005,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 79,
      "cert_fnv1a64_debug_v1": "6bca2ee0bec627fb",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          0,
          3
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -1,
          2
        ],
        [
          -1,
          4
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          2,
          -3
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          3,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          4,
          -3
        ],
        [
          4,
          -4
        ],
        [
          4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          0
        ],
        [
          4,
          1
        ],
        [
          3,
          2
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -3,
          -2
        ],
        [
          6,
          -1
        ],
        [
          7,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -4,
          -2
        ],
        [
          2,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          4
        ],
        [
          -5,
          4
        ],
        [
          -3,
          2
        ],
        [
          -4,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          4,
          2
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          -4,
          3
        ],
        [
          -7,
          4
        ],
        [
          -8,
          4
        ],
        [
          -9,
          4
        ],
        [
          -4,
          5
        ],
        [
          -4,
          6
        ],
        [
          0,
          -4
        ],
        [
          -2,
          2
        ],
        [
          -2,
          3
        ],
        [
          -2,
          4
        ],
        [
          -1,
          -3
        ]
      ]
    },
    {
      "id": "oa-ba95a2fd5bd51c8c",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 76,
      "placements": 76,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 88,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 214,
      "ms": 0.475,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          1
        ],
        [
          -2,
          2
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          -3,
          3
        ],
        [
          1,
          -2
        ],
        [
          -1,
          2
        ],
        [
          -3,
          2
        ],
        [
          -2,
          1
        ],
        [
          -4,
          3
        ],
        [
          0,
          2
        ],
        [
          1,
          2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          0,
          3
        ],
        [
          2,
          2
        ],
        [
          2,
          1
        ],
        [
          3,
          0
        ],
        [
          4,
          -1
        ],
        [
          3,
          1
        ],
        [
          4,
          0
        ],
        [
          1,
          3
        ],
        [
          1,
          -3
        ],
        [
          1,
          -4
        ],
        [
          1,
          -5
        ],
        [
          0,
          -2
        ],
        [
          0,
          4
        ],
        [
          0,
          5
        ],
        [
          0,
          6
        ],
        [
          -1,
          4
        ],
        [
          0,
          -3
        ],
        [
          -1,
          5
        ],
        [
          -2,
          5
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -3
        ],
        [
          -2,
          -3
        ],
        [
          -3,
          -3
        ],
        [
          2,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          5
        ],
        [
          2,
          5
        ],
        [
          3,
          -6
        ],
        [
          2,
          -6
        ],
        [
          4,
          -2
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -4,
          -2
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          0
        ],
        [
          2,
          -4
        ],
        [
          2,
          -7
        ],
        [
          2,
          -8
        ],
        [
          -2,
          4
        ],
        [
          -2,
          6
        ],
        [
          -3,
          7
        ],
        [
          -4,
          8
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -3
        ],
        [
          -4,
          -4
        ],
        [
          -4,
          -5
        ],
        [
          -5,
          1
        ],
        [
          -6,
          2
        ],
        [
          4,
          -4
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ]
      ]
    },
    {
      "id": "oa-60357784b5d9c345",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 75,
      "placements": 75,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 87,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 210,
      "ms": 0.464,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          0,
          3
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -1,
          2
        ],
        [
          -1,
          4
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          2,
          -3
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          3,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          4,
          -3
        ],
        [
          4,
          -4
        ],
        [
          4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          0
        ],
        [
          4,
          1
        ],
        [
          3,
          2
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -3,
          -2
        ],
        [
          6,
          -1
        ],
        [
          7,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -4,
          -2
        ],
        [
          2,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          4
        ],
        [
          -5,
          4
        ],
        [
          -3,
          2
        ],
        [
          -4,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          4,
          2
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          -4,
          3
        ],
        [
          -7,
          4
        ],
        [
          -8,
          4
        ],
        [
          -9,
          4
        ],
        [
          -4,
          5
        ],
        [
          -4,
          6
        ],
        [
          0,
          -4
        ],
        [
          -2,
          2
        ],
        [
          -2,
          3
        ]
      ]
    },
    {
      "id": "oa-bc4e83c6a7520d60",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 74,
      "placements": 74,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 86,
      "status": "UNKNOWN",
      "nodes": 294,
      "expansions": 293,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 114704,
      "ms": 43.116,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          0,
          -2
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          0,
          -3
        ],
        [
          -1,
          2
        ],
        [
          1,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          -3
        ],
        [
          2,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          1
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          2,
          -3
        ],
        [
          1,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -1
        ],
        [
          3,
          0
        ],
        [
          3,
          1
        ],
        [
          4,
          -1
        ],
        [
          4,
          0
        ],
        [
          4,
          -3
        ],
        [
          -2,
          3
        ],
        [
          -3,
          4
        ],
        [
          -4,
          5
        ],
        [
          -2,
          2
        ],
        [
          4,
          -4
        ],
        [
          5,
          -5
        ],
        [
          6,
          -6
        ],
        [
          3,
          -4
        ],
        [
          -3,
          3
        ],
        [
          4,
          -5
        ],
        [
          3,
          -5
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -6,
          3
        ],
        [
          -1,
          3
        ],
        [
          -3,
          5
        ],
        [
          6,
          -5
        ],
        [
          7,
          -5
        ],
        [
          -3,
          6
        ],
        [
          -4,
          6
        ],
        [
          2,
          2
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          2
        ],
        [
          -5,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          0
        ],
        [
          -2,
          4
        ],
        [
          -5,
          7
        ],
        [
          -6,
          8
        ],
        [
          2,
          -4
        ],
        [
          4,
          -6
        ],
        [
          4,
          -7
        ],
        [
          4,
          -8
        ],
        [
          -4,
          1
        ],
        [
          -7,
          3
        ],
        [
          -8,
          4
        ],
        [
          -9,
          5
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          -2
        ],
        [
          0,
          4
        ],
        [
          -2,
          0
        ]
      ]
    },
    {
      "id": "oa-59b0bb4cee141a3f",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 73,
      "placements": 73,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 85,
      "status": "WIN",
      "nodes": 7,
      "expansions": 6,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 1268,
      "ms": 0.925,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 8,
      "cert_edges": 3,
      "cert_commutations": 1,
      "cert_zones": 0,
      "derived_horizon": 83,
      "cert_fnv1a64_debug_v1": "4f9184dff6d83f32",
      "d6_verified": 6,
      "d6_mask": "0x1e3",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          0,
          3
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -1,
          2
        ],
        [
          -1,
          4
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          2,
          -3
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          3,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          4,
          -3
        ],
        [
          4,
          -4
        ],
        [
          4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          0
        ],
        [
          4,
          1
        ],
        [
          3,
          2
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -3,
          -2
        ],
        [
          6,
          -1
        ],
        [
          7,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -4,
          -2
        ],
        [
          2,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          4
        ],
        [
          -5,
          4
        ],
        [
          -3,
          2
        ],
        [
          -4,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          4,
          2
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          -4,
          3
        ],
        [
          -7,
          4
        ],
        [
          -8,
          4
        ],
        [
          -9,
          4
        ],
        [
          -4,
          5
        ],
        [
          -4,
          6
        ],
        [
          0,
          -4
        ]
      ]
    },
    {
      "id": "oa-0919a323b65cb1be",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 72,
      "placements": 72,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 84,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 206,
      "ms": 0.2,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          1
        ],
        [
          -2,
          2
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          -3,
          3
        ],
        [
          1,
          -2
        ],
        [
          -1,
          2
        ],
        [
          -3,
          2
        ],
        [
          -2,
          1
        ],
        [
          -4,
          3
        ],
        [
          0,
          2
        ],
        [
          1,
          2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          0,
          3
        ],
        [
          2,
          2
        ],
        [
          2,
          1
        ],
        [
          3,
          0
        ],
        [
          4,
          -1
        ],
        [
          3,
          1
        ],
        [
          4,
          0
        ],
        [
          1,
          3
        ],
        [
          1,
          -3
        ],
        [
          1,
          -4
        ],
        [
          1,
          -5
        ],
        [
          0,
          -2
        ],
        [
          0,
          4
        ],
        [
          0,
          5
        ],
        [
          0,
          6
        ],
        [
          -1,
          4
        ],
        [
          0,
          -3
        ],
        [
          -1,
          5
        ],
        [
          -2,
          5
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -3
        ],
        [
          -2,
          -3
        ],
        [
          -3,
          -3
        ],
        [
          2,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          5
        ],
        [
          2,
          5
        ],
        [
          3,
          -6
        ],
        [
          2,
          -6
        ],
        [
          4,
          -2
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -4,
          -2
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          0
        ],
        [
          2,
          -4
        ],
        [
          2,
          -7
        ],
        [
          2,
          -8
        ],
        [
          -2,
          4
        ],
        [
          -2,
          6
        ],
        [
          -3,
          7
        ],
        [
          -4,
          8
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -3
        ],
        [
          -4,
          -4
        ],
        [
          -4,
          -5
        ],
        [
          -5,
          1
        ],
        [
          -6,
          2
        ]
      ]
    },
    {
      "id": "oa-3a6d6ec14107199e",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 71,
      "placements": 71,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 83,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 202,
      "ms": 0.291,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          0,
          3
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -1,
          2
        ],
        [
          -1,
          4
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          2,
          -3
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          3,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          4,
          -3
        ],
        [
          4,
          -4
        ],
        [
          4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          0
        ],
        [
          4,
          1
        ],
        [
          3,
          2
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -3,
          -2
        ],
        [
          6,
          -1
        ],
        [
          7,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -4,
          -2
        ],
        [
          2,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          4
        ],
        [
          -5,
          4
        ],
        [
          -3,
          2
        ],
        [
          -4,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          4,
          2
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          -4,
          3
        ],
        [
          -7,
          4
        ],
        [
          -8,
          4
        ],
        [
          -9,
          4
        ],
        [
          -4,
          5
        ]
      ]
    },
    {
      "id": "oa-5696eb40c1e55608",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 70,
      "placements": 70,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 82,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 202,
      "ms": 0.396,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          0,
          3
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -1,
          3
        ],
        [
          -1,
          2
        ],
        [
          -1,
          4
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          2,
          -3
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          3,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          4,
          -3
        ],
        [
          4,
          -4
        ],
        [
          4,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -4,
          -1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          0
        ],
        [
          4,
          1
        ],
        [
          3,
          2
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -3,
          -2
        ],
        [
          6,
          -1
        ],
        [
          7,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -4,
          -2
        ],
        [
          2,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          4
        ],
        [
          -5,
          4
        ],
        [
          -3,
          2
        ],
        [
          -4,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          4,
          2
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          -4,
          3
        ],
        [
          -7,
          4
        ],
        [
          -8,
          4
        ],
        [
          -9,
          4
        ]
      ]
    },
    {
      "id": "oa-170dc3f41f697519",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 69,
      "placements": 69,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 81,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 198,
      "ms": 0.546,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          -2,
          2
        ],
        [
          3,
          -3
        ],
        [
          -2,
          1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          3,
          -4
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          -1,
          3
        ],
        [
          0,
          2
        ],
        [
          1,
          1
        ],
        [
          3,
          -1
        ],
        [
          3,
          -2
        ],
        [
          3,
          0
        ],
        [
          2,
          2
        ],
        [
          1,
          2
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          1,
          3
        ],
        [
          0,
          4
        ],
        [
          3,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          4,
          -1
        ],
        [
          -3,
          0
        ],
        [
          5,
          -1
        ],
        [
          5,
          -2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -5,
          2
        ],
        [
          5,
          1
        ],
        [
          5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          2
        ],
        [
          -2,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          -4
        ],
        [
          -1,
          -4
        ],
        [
          -1,
          -2
        ],
        [
          0,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -7,
          2
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          6,
          -2
        ],
        [
          7,
          -3
        ],
        [
          8,
          -4
        ],
        [
          -1,
          -3
        ],
        [
          -3,
          -4
        ],
        [
          -4,
          -4
        ]
      ]
    },
    {
      "id": "oa-6c2538ccf6158d28",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 68,
      "placements": 68,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 80,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 198,
      "ms": 0.264,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          0,
          -2
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          0,
          -3
        ],
        [
          -1,
          2
        ],
        [
          1,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          -3
        ],
        [
          2,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          1
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          2,
          -3
        ],
        [
          1,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -1
        ],
        [
          3,
          0
        ],
        [
          3,
          1
        ],
        [
          4,
          -1
        ],
        [
          4,
          0
        ],
        [
          4,
          -3
        ],
        [
          -2,
          3
        ],
        [
          -3,
          4
        ],
        [
          -4,
          5
        ],
        [
          -2,
          2
        ],
        [
          4,
          -4
        ],
        [
          5,
          -5
        ],
        [
          6,
          -6
        ],
        [
          3,
          -4
        ],
        [
          -3,
          3
        ],
        [
          4,
          -5
        ],
        [
          3,
          -5
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -6,
          3
        ],
        [
          -1,
          3
        ],
        [
          -3,
          5
        ],
        [
          6,
          -5
        ],
        [
          7,
          -5
        ],
        [
          -3,
          6
        ],
        [
          -4,
          6
        ],
        [
          2,
          2
        ],
        [
          -4,
          2
        ],
        [
          -2,
          1
        ],
        [
          -6,
          2
        ],
        [
          -5,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          0
        ],
        [
          -2,
          4
        ],
        [
          -5,
          7
        ],
        [
          -6,
          8
        ],
        [
          2,
          -4
        ],
        [
          4,
          -6
        ],
        [
          4,
          -7
        ],
        [
          4,
          -8
        ],
        [
          -4,
          1
        ],
        [
          -7,
          3
        ]
      ]
    },
    {
      "id": "oa-7a6cc6a4413dd39b",
      "source": "human:00070cdd8fb87f42:winner=-1",
      "source_prefix": 67,
      "placements": 67,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 79,
      "status": "WIN",
      "nodes": 14,
      "expansions": 13,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 5018,
      "ms": 1.403,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 16,
      "cert_edges": 7,
      "cert_commutations": 3,
      "cert_zones": 0,
      "derived_horizon": 77,
      "cert_fnv1a64_debug_v1": "9160fc65dba9a217",
      "d6_verified": 6,
      "d6_mask": "0x3b1",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          -2,
          2
        ],
        [
          3,
          -3
        ],
        [
          -2,
          1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          3,
          -4
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          -1,
          3
        ],
        [
          0,
          2
        ],
        [
          1,
          1
        ],
        [
          3,
          -1
        ],
        [
          3,
          -2
        ],
        [
          3,
          0
        ],
        [
          2,
          2
        ],
        [
          1,
          2
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          1,
          3
        ],
        [
          0,
          4
        ],
        [
          3,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          1
        ],
        [
          -2,
          0
        ],
        [
          4,
          0
        ],
        [
          5,
          0
        ],
        [
          6,
          0
        ],
        [
          4,
          -1
        ],
        [
          -3,
          0
        ],
        [
          5,
          -1
        ],
        [
          5,
          -2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          -2
        ],
        [
          -3,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -5,
          2
        ],
        [
          5,
          1
        ],
        [
          5,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          2
        ],
        [
          -2,
          4
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          -4
        ],
        [
          -1,
          -4
        ],
        [
          -1,
          -2
        ],
        [
          0,
          -4
        ],
        [
          -4,
          2
        ],
        [
          -7,
          2
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          6,
          -2
        ],
        [
          7,
          -3
        ],
        [
          8,
          -4
        ],
        [
          -1,
          -3
        ]
      ]
    },
    {
      "id": "oa-677c5db6ef8d57a2",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 20,
      "placements": 20,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 32,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 21,
      "cert_fnv1a64_debug_v1": "d6c28f7cea852216",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          0
        ],
        [
          2,
          0
        ],
        [
          -1,
          2
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -1
        ],
        [
          4,
          -1
        ],
        [
          3,
          0
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          0,
          -2
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-b22d9e7bf339df90",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 19,
      "placements": 19,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 31,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.002,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 21,
      "cert_fnv1a64_debug_v1": "746c78f1d8bdfa35",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          0
        ],
        [
          1,
          1
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          0,
          2
        ]
      ]
    },
    {
      "id": "oa-25069868e2973abc",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 18,
      "placements": 18,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 30,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 98,
      "ms": 0.322,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          0
        ],
        [
          2,
          0
        ],
        [
          -1,
          2
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -1
        ],
        [
          4,
          -1
        ],
        [
          3,
          0
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-8e989a6f8402dc5a",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 17,
      "placements": 17,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 29,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 94,
      "ms": 0.167,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          0
        ],
        [
          1,
          1
        ],
        [
          2,
          0
        ]
      ]
    },
    {
      "id": "oa-5690c817bfb2c49f",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 16,
      "placements": 16,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 28,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 94,
      "ms": 0.274,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          1
        ],
        [
          0,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          0
        ],
        [
          1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          2
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          2,
          1
        ],
        [
          3,
          1
        ],
        [
          3,
          0
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-449228735484d2b1",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 15,
      "placements": 15,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 27,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 90,
      "ms": 0.125,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-89d4e28892c24021",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 14,
      "placements": 14,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 26,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 90,
      "ms": 0.097,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ]
      ]
    },
    {
      "id": "oa-1736249785997468",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 13,
      "placements": 13,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 25,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 86,
      "ms": 0.064,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-78e03fc084a68b42",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 12,
      "placements": 12,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 24,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 86,
      "ms": 0.214,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          1,
          -2
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-780f0bfcd39a0844",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 11,
      "placements": 11,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 23,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 82,
      "ms": 0.09,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          -1
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          -2
        ],
        [
          1,
          1
        ]
      ]
    },
    {
      "id": "oa-546708b18342ee4d",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 10,
      "placements": 10,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 22,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 82,
      "ms": 0.175,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          1,
          -1
        ],
        [
          0,
          1
        ],
        [
          0,
          2
        ],
        [
          2,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-5d04d73b939934c3",
      "source": "human:001165e4e1d7e246:winner=1",
      "source_prefix": 9,
      "placements": 9,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 21,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 78,
      "ms": 0.045,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          -1
        ],
        [
          0,
          1
        ],
        [
          -1,
          0
        ],
        [
          -2,
          0
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ]
      ]
    },
    {
      "id": "oa-d049938e91bc7c71",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 56,
      "placements": 56,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 68,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "4c2b103fc5060717",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -2
        ],
        [
          -2,
          3
        ],
        [
          -7,
          1
        ],
        [
          -10,
          2
        ],
        [
          -6,
          2
        ]
      ]
    },
    {
      "id": "oa-0ba4d353ceed58dd",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 55,
      "placements": 55,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 67,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "e971642aeaa76494",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -2
        ],
        [
          -2,
          3
        ],
        [
          -7,
          1
        ],
        [
          -10,
          2
        ]
      ]
    },
    {
      "id": "oa-f3a7604310f03c6a",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 54,
      "placements": 54,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 66,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 170,
      "ms": 0.172,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -2
        ],
        [
          -2,
          3
        ],
        [
          -7,
          1
        ]
      ]
    },
    {
      "id": "oa-33148db6d16b08c9",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 53,
      "placements": 53,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 65,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 166,
      "ms": 0.12,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -2
        ],
        [
          -2,
          3
        ]
      ]
    },
    {
      "id": "oa-4d01c576f4671a1e",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 52,
      "placements": 52,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 64,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 166,
      "ms": 0.803,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          1,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -1
        ],
        [
          1,
          1
        ],
        [
          1,
          2
        ],
        [
          1,
          3
        ],
        [
          -2,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          2
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          2,
          1
        ],
        [
          0,
          -2
        ],
        [
          -3,
          0
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          0,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          2
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          3,
          0
        ],
        [
          -2,
          5
        ],
        [
          0,
          4
        ],
        [
          -1,
          3
        ],
        [
          -1,
          1
        ],
        [
          0,
          5
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          -2,
          4
        ],
        [
          4,
          -2
        ],
        [
          -1,
          5
        ],
        [
          2,
          3
        ],
        [
          2,
          4
        ],
        [
          -1,
          6
        ],
        [
          -2,
          3
        ],
        [
          -3,
          5
        ],
        [
          -3,
          4
        ],
        [
          -6,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -5,
          -3
        ],
        [
          -5,
          -2
        ],
        [
          -7,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          -4,
          2
        ]
      ]
    },
    {
      "id": "oa-5799a06a546f5980",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 51,
      "placements": 51,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 63,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 53,
      "cert_fnv1a64_debug_v1": "c53189d3af9d87af",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ]
      ]
    },
    {
      "id": "oa-86caa7e76759059b",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 50,
      "placements": 50,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 62,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 162,
      "ms": 0.257,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-def69919403f0e6a",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 49,
      "placements": 49,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 61,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 158,
      "ms": 0.194,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ],
        [
          -9,
          2
        ]
      ]
    },
    {
      "id": "oa-8cbd41c6cc89590d",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 48,
      "placements": 48,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 60,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 158,
      "ms": 0.596,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ],
        [
          -7,
          2
        ]
      ]
    },
    {
      "id": "oa-d92383d91efafe89",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 47,
      "placements": 47,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 59,
      "status": "UNKNOWN",
      "nodes": 399,
      "expansions": 398,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 128452,
      "ms": 99.311,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          3
        ]
      ]
    },
    {
      "id": "oa-831a4ddfb41323aa",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 46,
      "placements": 46,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 58,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 154,
      "ms": 0.182,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          -1
        ],
        [
          1,
          -1
        ],
        [
          1,
          -2
        ],
        [
          0,
          -1
        ],
        [
          1,
          1
        ],
        [
          1,
          2
        ],
        [
          1,
          3
        ],
        [
          -2,
          1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          2
        ],
        [
          -2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          2,
          1
        ],
        [
          0,
          -2
        ],
        [
          -3,
          0
        ],
        [
          -6,
          -2
        ],
        [
          2,
          2
        ],
        [
          0,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          2
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          3,
          0
        ],
        [
          -2,
          5
        ],
        [
          0,
          4
        ],
        [
          -1,
          3
        ],
        [
          -1,
          1
        ],
        [
          0,
          5
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          -2,
          4
        ],
        [
          4,
          -2
        ],
        [
          -1,
          5
        ],
        [
          2,
          3
        ],
        [
          2,
          4
        ],
        [
          -1,
          6
        ],
        [
          -2,
          3
        ],
        [
          -3,
          5
        ],
        [
          -3,
          4
        ],
        [
          -6,
          -3
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-6dfd7c37ff5ef261",
      "source": "human:001c0059e69f6973:winner=1",
      "source_prefix": 45,
      "placements": 45,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 57,
      "status": "UNKNOWN",
      "nodes": 10,
      "expansions": 9,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 2508,
      "ms": 0.606,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          -4,
          2
        ],
        [
          -3,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          0
        ],
        [
          -8,
          2
        ],
        [
          4,
          -2
        ],
        [
          2,
          -2
        ],
        [
          -1,
          -2
        ],
        [
          5,
          -2
        ],
        [
          3,
          -3
        ],
        [
          3,
          -4
        ],
        [
          3,
          0
        ],
        [
          3,
          -5
        ],
        [
          4,
          -4
        ],
        [
          2,
          -3
        ],
        [
          0,
          -1
        ],
        [
          5,
          -5
        ],
        [
          2,
          0
        ],
        [
          2,
          1
        ],
        [
          2,
          -4
        ],
        [
          2,
          2
        ],
        [
          4,
          -5
        ],
        [
          5,
          -3
        ],
        [
          6,
          -4
        ],
        [
          5,
          -6
        ],
        [
          1,
          -3
        ],
        [
          2,
          -5
        ],
        [
          1,
          -4
        ],
        [
          -9,
          3
        ]
      ]
    },
    {
      "id": "oa-fe7e1980dc545256",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 56,
      "placements": 56,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 68,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "36129f2f3ae04d19",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          -1
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          1,
          1
        ],
        [
          3,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          0
        ],
        [
          3,
          -2
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -2,
          0
        ],
        [
          -2,
          3
        ],
        [
          -3,
          0
        ],
        [
          0,
          -2
        ],
        [
          3,
          0
        ],
        [
          5,
          -2
        ],
        [
          4,
          -2
        ],
        [
          3,
          1
        ],
        [
          5,
          -1
        ],
        [
          4,
          -1
        ],
        [
          6,
          -3
        ],
        [
          -2,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          3
        ],
        [
          -3,
          3
        ],
        [
          -4,
          2
        ],
        [
          -5,
          5
        ],
        [
          -6,
          5
        ],
        [
          -5,
          4
        ],
        [
          -4,
          5
        ],
        [
          -6,
          4
        ],
        [
          -6,
          3
        ],
        [
          -6,
          2
        ],
        [
          -6,
          6
        ],
        [
          -7,
          5
        ],
        [
          -8,
          5
        ],
        [
          -9,
          5
        ],
        [
          -9,
          6
        ],
        [
          3,
          -3
        ],
        [
          -8,
          3
        ],
        [
          -7,
          3
        ],
        [
          3,
          -4
        ],
        [
          -8,
          4
        ],
        [
          -8,
          6
        ],
        [
          -9,
          7
        ],
        [
          -8,
          7
        ],
        [
          -8,
          2
        ]
      ]
    },
    {
      "id": "oa-daa537edfe2edefd",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 55,
      "placements": 55,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 67,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "379a54de40d8f27a",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ],
        [
          -7,
          4
        ],
        [
          3,
          1
        ],
        [
          -8,
          4
        ],
        [
          -8,
          2
        ],
        [
          -9,
          2
        ],
        [
          -8,
          1
        ]
      ]
    },
    {
      "id": "oa-6a586cb7d101df61",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 54,
      "placements": 54,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 66,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "8f119f9c91d2767e",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ],
        [
          -7,
          4
        ],
        [
          3,
          1
        ],
        [
          -8,
          4
        ],
        [
          -8,
          2
        ],
        [
          -9,
          2
        ]
      ]
    },
    {
      "id": "oa-49923acd8f7ff846",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 53,
      "placements": 53,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 65,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.005,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "ba33930acee7979e",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ],
        [
          -7,
          4
        ],
        [
          3,
          1
        ],
        [
          -8,
          4
        ],
        [
          -8,
          2
        ]
      ]
    },
    {
      "id": "oa-21e88affe3597bbf",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 52,
      "placements": 52,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 64,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 166,
      "ms": 0.263,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 2,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "4c73530263d1f655",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ],
        [
          -7,
          4
        ],
        [
          3,
          1
        ],
        [
          -8,
          4
        ]
      ]
    },
    {
      "id": "oa-8879bbcab94a3163",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 51,
      "placements": 51,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 63,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 162,
      "ms": 0.251,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "fdc81cd5d6d53f8e",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ],
        [
          -7,
          4
        ],
        [
          3,
          1
        ]
      ]
    },
    {
      "id": "oa-cff86461902c8ec0",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 50,
      "placements": 50,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 62,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 162,
      "ms": 0.57,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -1,
          -1
        ],
        [
          1,
          0
        ],
        [
          -1,
          1
        ],
        [
          1,
          1
        ],
        [
          3,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          0
        ],
        [
          3,
          -2
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          -1,
          2
        ],
        [
          -2,
          0
        ],
        [
          -2,
          3
        ],
        [
          -3,
          0
        ],
        [
          0,
          -2
        ],
        [
          3,
          0
        ],
        [
          5,
          -2
        ],
        [
          4,
          -2
        ],
        [
          3,
          1
        ],
        [
          5,
          -1
        ],
        [
          4,
          -1
        ],
        [
          6,
          -3
        ],
        [
          -2,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -5,
          3
        ],
        [
          -3,
          3
        ],
        [
          -4,
          2
        ],
        [
          -5,
          5
        ],
        [
          -6,
          5
        ],
        [
          -5,
          4
        ],
        [
          -4,
          5
        ],
        [
          -6,
          4
        ],
        [
          -6,
          3
        ],
        [
          -6,
          2
        ],
        [
          -6,
          6
        ],
        [
          -7,
          5
        ],
        [
          -8,
          5
        ],
        [
          -9,
          5
        ],
        [
          -9,
          6
        ],
        [
          3,
          -3
        ],
        [
          -8,
          3
        ],
        [
          -7,
          3
        ]
      ]
    },
    {
      "id": "oa-366cf50f63f1de92",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 49,
      "placements": 49,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 61,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 158,
      "ms": 0.453,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ],
        [
          3,
          0
        ],
        [
          -8,
          5
        ]
      ]
    },
    {
      "id": "oa-558f79a590c31b6a",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 48,
      "placements": 48,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 60,
      "status": "WIN",
      "nodes": 6619,
      "expansions": 6618,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 1803682,
      "ms": 421.087,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 18,
      "cert_edges": 8,
      "cert_commutations": 4,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "d83569d7731ed943",
      "d6_verified": 6,
      "d6_mask": "0x8e3",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          -3,
          2
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          1
        ],
        [
          -2,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          1
        ],
        [
          2,
          -2
        ],
        [
          2,
          1
        ],
        [
          3,
          -3
        ],
        [
          0,
          -2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          3
        ],
        [
          -4,
          2
        ],
        [
          -3,
          4
        ],
        [
          -5,
          4
        ],
        [
          -4,
          3
        ],
        [
          -6,
          3
        ],
        [
          2,
          0
        ],
        [
          3,
          -1
        ],
        [
          4,
          -1
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          5,
          -2
        ],
        [
          3,
          0
        ],
        [
          4,
          -2
        ],
        [
          5,
          0
        ],
        [
          6,
          -1
        ],
        [
          5,
          -1
        ],
        [
          4,
          1
        ],
        [
          6,
          -2
        ],
        [
          6,
          -3
        ],
        [
          6,
          -4
        ],
        [
          6,
          0
        ],
        [
          7,
          -2
        ],
        [
          8,
          -3
        ],
        [
          9,
          -4
        ],
        [
          9,
          -3
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-1f451345a3cd82e0",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 47,
      "placements": 47,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 59,
      "status": "WIN",
      "nodes": 148,
      "expansions": 147,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 78148,
      "ms": 12.954,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 19,
      "cert_edges": 8,
      "cert_commutations": 4,
      "cert_zones": 0,
      "derived_horizon": 57,
      "cert_fnv1a64_debug_v1": "84e40a1d9099657e",
      "d6_verified": 6,
      "d6_mask": "0x8e3",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ],
        [
          -9,
          3
        ]
      ]
    },
    {
      "id": "oa-b4051fb62a1d40dd",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 46,
      "placements": 46,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 58,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 154,
      "ms": 0.336,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ],
        [
          -9,
          4
        ]
      ]
    },
    {
      "id": "oa-38a59422210cfa0e",
      "source": "human:002f5360162bac9b:winner=1",
      "source_prefix": 45,
      "placements": 45,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 57,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 150,
      "ms": 0.236,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          2
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -3,
          3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          5,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -4
        ],
        [
          5,
          -4
        ],
        [
          4,
          -3
        ],
        [
          6,
          -3
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -2,
          1
        ],
        [
          -3,
          2
        ],
        [
          -4,
          3
        ],
        [
          -5,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          2
        ],
        [
          -5,
          0
        ],
        [
          -6,
          1
        ],
        [
          -5,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -6,
          2
        ],
        [
          -6,
          3
        ],
        [
          -6,
          4
        ],
        [
          -6,
          0
        ],
        [
          -7,
          2
        ],
        [
          -8,
          3
        ]
      ]
    },
    {
      "id": "oa-ac4e139c0c235214",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 58,
      "placements": 58,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 70,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "372bde58941d5390",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -2
        ],
        [
          -15,
          5
        ],
        [
          -11,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          -7,
          1
        ]
      ]
    },
    {
      "id": "oa-804525958e2600db",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 57,
      "placements": 57,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 69,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "c09eff2909665867",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -2
        ],
        [
          -15,
          5
        ],
        [
          -11,
          -2
        ],
        [
          -6,
          -2
        ]
      ]
    },
    {
      "id": "oa-a1ea2be39377a6bb",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 56,
      "placements": 56,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 68,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "057c643bac45c29f",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -3,
          -1
        ],
        [
          -5,
          1
        ],
        [
          0,
          -4
        ],
        [
          -3,
          -2
        ],
        [
          -4,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -4,
          -1
        ],
        [
          -3,
          -3
        ],
        [
          -3,
          -4
        ],
        [
          -3,
          1
        ],
        [
          -3,
          -5
        ],
        [
          -5,
          -1
        ],
        [
          -6,
          0
        ],
        [
          -7,
          1
        ],
        [
          -2,
          -4
        ],
        [
          -4,
          1
        ],
        [
          -6,
          -3
        ],
        [
          -7,
          0
        ],
        [
          -5,
          -3
        ],
        [
          -8,
          -1
        ],
        [
          -7,
          -1
        ],
        [
          -8,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          0
        ],
        [
          -7,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          0,
          -2
        ],
        [
          -10,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -10,
          2
        ],
        [
          -16,
          -2
        ],
        [
          -17,
          -1
        ],
        [
          -16,
          -1
        ],
        [
          -15,
          -3
        ],
        [
          -1,
          -7
        ],
        [
          -10,
          -2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          -4
        ],
        [
          -10,
          0
        ],
        [
          -12,
          2
        ],
        [
          -13,
          3
        ],
        [
          -8,
          -2
        ],
        [
          -9,
          2
        ],
        [
          -10,
          -5
        ],
        [
          -13,
          2
        ]
      ]
    },
    {
      "id": "oa-4308de9abc3b9f54",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 55,
      "placements": 55,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 67,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.005,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "d0afdff2d315b049",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -2
        ],
        [
          -15,
          5
        ]
      ]
    },
    {
      "id": "oa-4a91f3b61b890567",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 54,
      "placements": 54,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 66,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 170,
      "ms": 0.337,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 2,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "3a82102abf45b66e",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -3,
          -1
        ],
        [
          -5,
          1
        ],
        [
          0,
          -4
        ],
        [
          -3,
          -2
        ],
        [
          -4,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -4,
          -1
        ],
        [
          -3,
          -3
        ],
        [
          -3,
          -4
        ],
        [
          -3,
          1
        ],
        [
          -3,
          -5
        ],
        [
          -5,
          -1
        ],
        [
          -6,
          0
        ],
        [
          -7,
          1
        ],
        [
          -2,
          -4
        ],
        [
          -4,
          1
        ],
        [
          -6,
          -3
        ],
        [
          -7,
          0
        ],
        [
          -5,
          -3
        ],
        [
          -8,
          -1
        ],
        [
          -7,
          -1
        ],
        [
          -8,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          0
        ],
        [
          -7,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          0,
          -2
        ],
        [
          -10,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -10,
          2
        ],
        [
          -16,
          -2
        ],
        [
          -17,
          -1
        ],
        [
          -16,
          -1
        ],
        [
          -15,
          -3
        ],
        [
          -1,
          -7
        ],
        [
          -10,
          -2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          -4
        ],
        [
          -10,
          0
        ],
        [
          -12,
          2
        ],
        [
          -13,
          3
        ],
        [
          -8,
          -2
        ],
        [
          -9,
          2
        ]
      ]
    },
    {
      "id": "oa-2b428a33ac46597c",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 53,
      "placements": 53,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 65,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 166,
      "ms": 0.525,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "b45f7a6a6616c6f1",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          2
        ]
      ]
    },
    {
      "id": "oa-80bd8709b18c08e3",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 52,
      "placements": 52,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 64,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 166,
      "ms": 0.373,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          1
        ],
        [
          0,
          -1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -3,
          -1
        ],
        [
          -5,
          1
        ],
        [
          0,
          -4
        ],
        [
          -3,
          -2
        ],
        [
          -4,
          -2
        ],
        [
          -5,
          -2
        ],
        [
          -4,
          -1
        ],
        [
          -3,
          -3
        ],
        [
          -3,
          -4
        ],
        [
          -3,
          1
        ],
        [
          -3,
          -5
        ],
        [
          -5,
          -1
        ],
        [
          -6,
          0
        ],
        [
          -7,
          1
        ],
        [
          -2,
          -4
        ],
        [
          -4,
          1
        ],
        [
          -6,
          -3
        ],
        [
          -7,
          0
        ],
        [
          -5,
          -3
        ],
        [
          -8,
          -1
        ],
        [
          -7,
          -1
        ],
        [
          -8,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          0
        ],
        [
          -7,
          -2
        ],
        [
          -6,
          -2
        ],
        [
          0,
          -2
        ],
        [
          -10,
          3
        ],
        [
          -1,
          -2
        ],
        [
          -10,
          2
        ],
        [
          -16,
          -2
        ],
        [
          -17,
          -1
        ],
        [
          -16,
          -1
        ],
        [
          -15,
          -3
        ],
        [
          -1,
          -7
        ],
        [
          -10,
          -2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          2
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          -4
        ],
        [
          -10,
          0
        ],
        [
          -12,
          2
        ],
        [
          -13,
          3
        ]
      ]
    },
    {
      "id": "oa-f42fef14e00732a9",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 51,
      "placements": 51,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 63,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 162,
      "ms": 0.282,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -2
        ]
      ]
    },
    {
      "id": "oa-01dc3c7bd7e2deca",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 50,
      "placements": 50,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 62,
      "status": "WIN",
      "nodes": 135,
      "expansions": 134,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 40124,
      "ms": 16.256,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 15,
      "cert_edges": 7,
      "cert_commutations": 3,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "ee5c6b76e86b4473",
      "d6_verified": 6,
      "d6_mask": "0x3b1",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -10,
          0
        ]
      ]
    },
    {
      "id": "oa-c43be8cf141a00b3",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 49,
      "placements": 49,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 61,
      "status": "WIN",
      "nodes": 8,
      "expansions": 7,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 818,
      "ms": 2.014,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 16,
      "cert_edges": 7,
      "cert_commutations": 3,
      "cert_zones": 0,
      "derived_horizon": 59,
      "cert_fnv1a64_debug_v1": "f1901c206243d869",
      "d6_verified": 6,
      "d6_mask": "0x3b1",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ]
      ]
    },
    {
      "id": "oa-6d84e805496fc4ac",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 48,
      "placements": 48,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 60,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 158,
      "ms": 0.27,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ],
        [
          -13,
          3
        ]
      ]
    },
    {
      "id": "oa-28244dbc5f5a03d0",
      "source": "human:0035f32035e5468b:winner=-1",
      "source_prefix": 47,
      "placements": 47,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 59,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 154,
      "ms": 0.19,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          1
        ],
        [
          -1,
          0
        ],
        [
          -4,
          0
        ],
        [
          -2,
          0
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -4,
          -1
        ],
        [
          -4,
          4
        ],
        [
          -5,
          2
        ],
        [
          -6,
          2
        ],
        [
          -7,
          2
        ],
        [
          -5,
          1
        ],
        [
          -6,
          3
        ],
        [
          -7,
          4
        ],
        [
          -2,
          -1
        ],
        [
          -8,
          5
        ],
        [
          -6,
          1
        ],
        [
          -6,
          0
        ],
        [
          -6,
          -1
        ],
        [
          -6,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -9,
          3
        ],
        [
          -7,
          0
        ],
        [
          -8,
          3
        ],
        [
          -9,
          1
        ],
        [
          -8,
          1
        ],
        [
          -7,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -9,
          0
        ],
        [
          -9,
          2
        ],
        [
          -8,
          2
        ],
        [
          -2,
          2
        ],
        [
          -7,
          -3
        ],
        [
          -3,
          2
        ],
        [
          -8,
          -2
        ],
        [
          -18,
          2
        ],
        [
          -18,
          1
        ],
        [
          -17,
          1
        ],
        [
          -18,
          3
        ],
        [
          -8,
          7
        ],
        [
          -12,
          2
        ],
        [
          -10,
          -1
        ],
        [
          -9,
          -2
        ]
      ]
    },
    {
      "id": "oa-c5d0933314357fb6",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 38,
      "placements": 38,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 50,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 39,
      "cert_fnv1a64_debug_v1": "7e6f922459ccfca6",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          -2
        ],
        [
          2,
          0
        ],
        [
          1,
          1
        ],
        [
          3,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          1
        ],
        [
          0,
          -2
        ],
        [
          3,
          0
        ],
        [
          -1,
          -2
        ],
        [
          3,
          -4
        ],
        [
          2,
          -3
        ],
        [
          -1,
          0
        ],
        [
          3,
          -3
        ],
        [
          1,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          1,
          -4
        ],
        [
          0,
          -3
        ],
        [
          -1,
          -3
        ],
        [
          -2,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -4,
          0
        ],
        [
          -2,
          2
        ],
        [
          -5,
          0
        ],
        [
          0,
          -4
        ],
        [
          -4,
          1
        ],
        [
          4,
          -5
        ],
        [
          4,
          -3
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-61a6af9aed05053d",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 37,
      "placements": 37,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 49,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 39,
      "cert_fnv1a64_debug_v1": "7fe2d8c1c8fccf11",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          1,
          2
        ],
        [
          2,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          3,
          0
        ],
        [
          -3,
          2
        ],
        [
          -1,
          4
        ],
        [
          -1,
          3
        ],
        [
          -1,
          0
        ],
        [
          0,
          3
        ],
        [
          -2,
          3
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -3,
          4
        ],
        [
          -3,
          3
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -1,
          -1
        ],
        [
          -4,
          0
        ],
        [
          0,
          -2
        ],
        [
          -5,
          0
        ],
        [
          -4,
          4
        ],
        [
          -3,
          -1
        ],
        [
          -1,
          5
        ],
        [
          1,
          3
        ]
      ]
    },
    {
      "id": "oa-1b42abe4c7f5c99c",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 36,
      "placements": 36,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 48,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 134,
      "ms": 0.17,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          -1,
          0
        ],
        [
          0,
          -1
        ],
        [
          -2,
          1
        ],
        [
          0,
          -2
        ],
        [
          1,
          -2
        ],
        [
          -2,
          -1
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          1,
          -3
        ],
        [
          -2,
          2
        ],
        [
          0,
          -3
        ],
        [
          -2,
          3
        ],
        [
          -4,
          1
        ],
        [
          -3,
          1
        ],
        [
          0,
          1
        ],
        [
          -3,
          0
        ],
        [
          -3,
          2
        ],
        [
          -1,
          2
        ],
        [
          0,
          2
        ],
        [
          -4,
          3
        ],
        [
          -3,
          3
        ],
        [
          -3,
          4
        ],
        [
          -3,
          5
        ],
        [
          -1,
          3
        ],
        [
          -2,
          4
        ],
        [
          -1,
          4
        ],
        [
          1,
          1
        ],
        [
          0,
          4
        ],
        [
          2,
          0
        ],
        [
          0,
          5
        ],
        [
          -4,
          4
        ],
        [
          1,
          3
        ],
        [
          -5,
          1
        ]
      ]
    },
    {
      "id": "oa-829c58832fdf8c8b",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 35,
      "placements": 35,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 47,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 130,
      "ms": 0.162,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          1,
          2
        ],
        [
          2,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          3,
          0
        ],
        [
          -3,
          2
        ],
        [
          -1,
          4
        ],
        [
          -1,
          3
        ],
        [
          -1,
          0
        ],
        [
          0,
          3
        ],
        [
          -2,
          3
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -3,
          4
        ],
        [
          -3,
          3
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -1,
          -1
        ],
        [
          -4,
          0
        ],
        [
          0,
          -2
        ],
        [
          -5,
          0
        ],
        [
          -4,
          4
        ],
        [
          -3,
          -1
        ]
      ]
    },
    {
      "id": "oa-a946100f697bbcf4",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 34,
      "placements": 34,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 46,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 130,
      "ms": 0.291,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          1,
          -3
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          0
        ],
        [
          3,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -1,
          -3
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          0,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          2
        ],
        [
          -4,
          3
        ],
        [
          -1,
          2
        ],
        [
          -4,
          4
        ],
        [
          0,
          2
        ],
        [
          -5,
          5
        ],
        [
          -4,
          0
        ]
      ]
    },
    {
      "id": "oa-048d6263244991f9",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 33,
      "placements": 33,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 45,
      "status": "UNKNOWN",
      "nodes": 231,
      "expansions": 230,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 59262,
      "ms": 26.277,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          -1,
          2
        ],
        [
          2,
          0
        ],
        [
          2,
          -1
        ],
        [
          1,
          2
        ],
        [
          2,
          1
        ],
        [
          1,
          1
        ],
        [
          0,
          2
        ],
        [
          3,
          -1
        ],
        [
          -2,
          2
        ],
        [
          3,
          0
        ],
        [
          -3,
          2
        ],
        [
          -1,
          4
        ],
        [
          -1,
          3
        ],
        [
          -1,
          0
        ],
        [
          0,
          3
        ],
        [
          -2,
          3
        ],
        [
          -2,
          1
        ],
        [
          -2,
          0
        ],
        [
          -3,
          4
        ],
        [
          -3,
          3
        ],
        [
          -4,
          3
        ],
        [
          -5,
          3
        ],
        [
          -3,
          1
        ],
        [
          -4,
          2
        ],
        [
          -4,
          1
        ],
        [
          -1,
          -1
        ],
        [
          -4,
          0
        ],
        [
          0,
          -2
        ],
        [
          -5,
          0
        ]
      ]
    },
    {
      "id": "oa-01b6ac75721cc215",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 32,
      "placements": 32,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 44,
      "status": "UNKNOWN",
      "nodes": 61,
      "expansions": 60,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 15932,
      "ms": 5.09,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          1
        ],
        [
          2,
          -2
        ],
        [
          1,
          -2
        ],
        [
          3,
          -1
        ],
        [
          3,
          -2
        ],
        [
          2,
          -1
        ],
        [
          2,
          0
        ],
        [
          2,
          -3
        ],
        [
          0,
          2
        ],
        [
          3,
          -3
        ],
        [
          -1,
          3
        ],
        [
          3,
          1
        ],
        [
          2,
          1
        ],
        [
          -1,
          1
        ],
        [
          3,
          0
        ],
        [
          1,
          2
        ],
        [
          -1,
          2
        ],
        [
          -2,
          2
        ],
        [
          1,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -2,
          5
        ],
        [
          -2,
          3
        ],
        [
          -2,
          4
        ],
        [
          -3,
          4
        ],
        [
          -2,
          1
        ],
        [
          -4,
          4
        ],
        [
          -2,
          0
        ]
      ]
    },
    {
      "id": "oa-8b2f9b5a954134a5",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 31,
      "placements": 31,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 43,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 122,
      "ms": 0.204,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 37,
      "cert_fnv1a64_debug_v1": "86545bd080fdf094",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          1,
          -3
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          0
        ],
        [
          3,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -1,
          -3
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          0,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          2
        ],
        [
          -4,
          3
        ],
        [
          -1,
          2
        ],
        [
          -4,
          4
        ]
      ]
    },
    {
      "id": "oa-990689321c889325",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 30,
      "placements": 30,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 42,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 122,
      "ms": 0.356,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          1,
          -2
        ],
        [
          2,
          0
        ],
        [
          1,
          1
        ],
        [
          3,
          -2
        ],
        [
          3,
          -1
        ],
        [
          2,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          1
        ],
        [
          0,
          -2
        ],
        [
          3,
          0
        ],
        [
          -1,
          -2
        ],
        [
          3,
          -4
        ],
        [
          2,
          -3
        ],
        [
          -1,
          0
        ],
        [
          3,
          -3
        ],
        [
          1,
          -3
        ],
        [
          -1,
          -1
        ],
        [
          -2,
          0
        ],
        [
          1,
          -4
        ],
        [
          0,
          -3
        ],
        [
          -1,
          -3
        ],
        [
          -2,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          -2
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          1
        ]
      ]
    },
    {
      "id": "oa-a8d5d3202f1b088d",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 29,
      "placements": 29,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 41,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 118,
      "ms": 0.192,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          1,
          -3
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          0
        ],
        [
          3,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -1,
          -3
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          0,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          2
        ],
        [
          -4,
          3
        ]
      ]
    },
    {
      "id": "oa-508c613e4f7cc1c1",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 28,
      "placements": 28,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 40,
      "status": "UNKNOWN",
      "nodes": 14,
      "expansions": 13,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 2272,
      "ms": 1.028,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          1,
          -3
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          0
        ],
        [
          3,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -1,
          -3
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          0,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -3,
          2
        ],
        [
          -4,
          2
        ]
      ]
    },
    {
      "id": "oa-f902f3534eff1449",
      "source": "human:00386e2d3c6f65fd:winner=-1",
      "source_prefix": 27,
      "placements": 27,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 39,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 114,
      "ms": 0.131,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 33,
      "cert_fnv1a64_debug_v1": "99729ba8d8fe8c32",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          0
        ],
        [
          0,
          -1
        ],
        [
          1,
          -1
        ],
        [
          -1,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -1
        ],
        [
          1,
          -3
        ],
        [
          2,
          -3
        ],
        [
          1,
          -2
        ],
        [
          0,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          0
        ],
        [
          3,
          -3
        ],
        [
          -3,
          1
        ],
        [
          -1,
          -3
        ],
        [
          -1,
          -2
        ],
        [
          -1,
          1
        ],
        [
          0,
          -3
        ],
        [
          -2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          -2,
          2
        ],
        [
          -3,
          -1
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -5,
          2
        ],
        [
          -3,
          2
        ]
      ]
    },
    {
      "id": "oa-5e92071b23b32555",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 40,
      "placements": 40,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 52,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.003,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "3e9ecebfa9af6bd9",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          2
        ],
        [
          1,
          -2
        ],
        [
          -1,
          3
        ],
        [
          1,
          -3
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          1,
          0
        ],
        [
          2,
          -2
        ],
        [
          2,
          0
        ],
        [
          2,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -3
        ],
        [
          4,
          -1
        ],
        [
          3,
          -4
        ],
        [
          2,
          3
        ],
        [
          3,
          -1
        ],
        [
          3,
          0
        ],
        [
          5,
          -1
        ],
        [
          3,
          1
        ],
        [
          4,
          -4
        ],
        [
          1,
          -1
        ],
        [
          5,
          -5
        ],
        [
          0,
          -1
        ],
        [
          1,
          2
        ],
        [
          2,
          1
        ],
        [
          5,
          -2
        ],
        [
          0,
          3
        ],
        [
          1,
          3
        ],
        [
          5,
          -3
        ],
        [
          0,
          -2
        ],
        [
          -2,
          0
        ],
        [
          0,
          1
        ],
        [
          0,
          -3
        ],
        [
          -1,
          0
        ],
        [
          -1,
          -1
        ],
        [
          -3,
          1
        ],
        [
          2,
          -4
        ],
        [
          -3,
          0
        ]
      ]
    },
    {
      "id": "oa-4cf66ead127526be",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 39,
      "placements": 39,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 51,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.002,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "d15b0bcd7738d472",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -5,
          2
        ],
        [
          0,
          -2
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          0,
          -3
        ],
        [
          1,
          -1
        ],
        [
          1,
          -2
        ],
        [
          3,
          -2
        ],
        [
          -2,
          -2
        ]
      ]
    },
    {
      "id": "oa-eff21276358dddc3",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 38,
      "placements": 38,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 50,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.002,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "3ceb0fc6337acf9c",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          2
        ],
        [
          1,
          -2
        ],
        [
          -1,
          3
        ],
        [
          1,
          -3
        ],
        [
          2,
          -1
        ],
        [
          3,
          -2
        ],
        [
          4,
          -3
        ],
        [
          1,
          0
        ],
        [
          2,
          -2
        ],
        [
          2,
          0
        ],
        [
          2,
          -3
        ],
        [
          4,
          -2
        ],
        [
          3,
          -3
        ],
        [
          4,
          -1
        ],
        [
          3,
          -4
        ],
        [
          2,
          3
        ],
        [
          3,
          -1
        ],
        [
          3,
          0
        ],
        [
          5,
          -1
        ],
        [
          3,
          1
        ],
        [
          4,
          -4
        ],
        [
          1,
          -1
        ],
        [
          5,
          -5
        ],
        [
          0,
          -1
        ],
        [
          1,
          2
        ],
        [
          2,
          1
        ],
        [
          5,
          -2
        ],
        [
          0,
          3
        ],
        [
          1,
          3
        ],
        [
          5,
          -3
        ],
        [
          0,
          -2
        ],
        [
          -2,
          0
        ],
        [
          0,
          1
        ],
        [
          0,
          -3
        ],
        [
          -1,
          0
        ],
        [
          -1,
          -1
        ],
        [
          -3,
          1
        ]
      ]
    },
    {
      "id": "oa-c781cbf30f111f01",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 37,
      "placements": 37,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 49,
      "status": "LOSS",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.004,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "14bdb5b0eda494df",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -5,
          2
        ],
        [
          0,
          -2
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          0,
          -3
        ],
        [
          1,
          -1
        ],
        [
          1,
          -2
        ]
      ]
    },
    {
      "id": "oa-82095adae0b94f52",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 36,
      "placements": 36,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 48,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 134,
      "ms": 0.207,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 2,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "ab4c23c848aaf818",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -2
        ],
        [
          -1,
          2
        ],
        [
          2,
          -3
        ],
        [
          -2,
          3
        ],
        [
          1,
          1
        ],
        [
          1,
          2
        ],
        [
          1,
          3
        ],
        [
          1,
          0
        ],
        [
          0,
          2
        ],
        [
          2,
          0
        ],
        [
          -1,
          3
        ],
        [
          2,
          2
        ],
        [
          0,
          3
        ],
        [
          3,
          1
        ],
        [
          -1,
          4
        ],
        [
          5,
          -3
        ],
        [
          2,
          1
        ],
        [
          3,
          0
        ],
        [
          4,
          1
        ],
        [
          4,
          -1
        ],
        [
          0,
          4
        ],
        [
          0,
          1
        ],
        [
          0,
          5
        ],
        [
          -1,
          1
        ],
        [
          3,
          -2
        ],
        [
          3,
          -1
        ],
        [
          3,
          2
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          2,
          3
        ],
        [
          -2,
          2
        ],
        [
          -2,
          0
        ],
        [
          1,
          -1
        ],
        [
          -3,
          3
        ],
        [
          -1,
          0
        ]
      ]
    },
    {
      "id": "oa-f6170888a8d617a1",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 35,
      "placements": 35,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 47,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 130,
      "ms": 0.25,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 41,
      "cert_fnv1a64_debug_v1": "e674f59510157ec1",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -5,
          2
        ],
        [
          0,
          -2
        ],
        [
          2,
          -2
        ],
        [
          0,
          1
        ],
        [
          0,
          -3
        ]
      ]
    },
    {
      "id": "oa-972166d805126280",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 34,
      "placements": 34,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 46,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 130,
      "ms": 0.246,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -2,
          -1
        ],
        [
          2,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          3
        ],
        [
          -1,
          4
        ],
        [
          -1,
          1
        ],
        [
          0,
          2
        ],
        [
          -2,
          2
        ],
        [
          1,
          2
        ],
        [
          -2,
          4
        ],
        [
          0,
          3
        ],
        [
          -3,
          4
        ],
        [
          1,
          3
        ],
        [
          -5,
          2
        ],
        [
          -2,
          3
        ],
        [
          -3,
          3
        ],
        [
          -4,
          5
        ],
        [
          -4,
          3
        ],
        [
          0,
          4
        ],
        [
          0,
          1
        ],
        [
          0,
          5
        ],
        [
          1,
          0
        ],
        [
          -3,
          1
        ],
        [
          -3,
          2
        ],
        [
          -3,
          5
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ],
        [
          -2,
          5
        ],
        [
          2,
          0
        ],
        [
          2,
          -2
        ],
        [
          -1,
          0
        ]
      ]
    },
    {
      "id": "oa-d5dadc85d1040f77",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 33,
      "placements": 33,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 45,
      "status": "UNKNOWN",
      "nodes": 8,
      "expansions": 7,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 1190,
      "ms": 0.471,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -5,
          2
        ],
        [
          0,
          -2
        ],
        [
          2,
          -2
        ]
      ]
    },
    {
      "id": "oa-b6ef5d0781559bd0",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 32,
      "placements": 32,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 44,
      "status": "UNKNOWN",
      "nodes": 311,
      "expansions": 310,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 60470,
      "ms": 74.143,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          2,
          -1
        ],
        [
          -2,
          1
        ],
        [
          3,
          -1
        ],
        [
          -3,
          1
        ],
        [
          -1,
          2
        ],
        [
          -2,
          3
        ],
        [
          -3,
          4
        ],
        [
          0,
          1
        ],
        [
          -2,
          2
        ],
        [
          0,
          2
        ],
        [
          -3,
          2
        ],
        [
          -2,
          4
        ],
        [
          -3,
          3
        ],
        [
          -1,
          4
        ],
        [
          -4,
          3
        ],
        [
          3,
          2
        ],
        [
          -1,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          5
        ],
        [
          1,
          3
        ],
        [
          -4,
          4
        ],
        [
          -1,
          1
        ],
        [
          -5,
          5
        ],
        [
          -1,
          0
        ],
        [
          2,
          1
        ],
        [
          1,
          2
        ],
        [
          -2,
          5
        ],
        [
          3,
          0
        ],
        [
          3,
          1
        ],
        [
          -3,
          5
        ],
        [
          -2,
          0
        ]
      ]
    },
    {
      "id": "oa-398b8932621ac136",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 31,
      "placements": 31,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 43,
      "status": "UNKNOWN",
      "nodes": 317,
      "expansions": 316,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 61754,
      "ms": 73.045,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ],
        [
          -1,
          4
        ],
        [
          -5,
          2
        ]
      ]
    },
    {
      "id": "oa-af37004d57e48af5",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 30,
      "placements": 30,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 42,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 122,
      "ms": 0.25,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          -1,
          -1
        ],
        [
          1,
          1
        ],
        [
          -2,
          -1
        ],
        [
          2,
          1
        ],
        [
          -1,
          2
        ],
        [
          -1,
          3
        ],
        [
          -1,
          4
        ],
        [
          -1,
          1
        ],
        [
          0,
          2
        ],
        [
          -2,
          2
        ],
        [
          1,
          2
        ],
        [
          -2,
          4
        ],
        [
          0,
          3
        ],
        [
          -3,
          4
        ],
        [
          1,
          3
        ],
        [
          -5,
          2
        ],
        [
          -2,
          3
        ],
        [
          -3,
          3
        ],
        [
          -4,
          5
        ],
        [
          -4,
          3
        ],
        [
          0,
          4
        ],
        [
          0,
          1
        ],
        [
          0,
          5
        ],
        [
          1,
          0
        ],
        [
          -3,
          1
        ],
        [
          -3,
          2
        ],
        [
          -3,
          5
        ],
        [
          -3,
          0
        ],
        [
          -4,
          1
        ]
      ]
    },
    {
      "id": "oa-ec6fdb676697a62b",
      "source": "human:003c115aa968eb5a:winner=1",
      "source_prefix": 29,
      "placements": 29,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 41,
      "status": "UNKNOWN",
      "nodes": 8,
      "expansions": 7,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 1118,
      "ms": 0.458,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          1
        ],
        [
          -1,
          -1
        ],
        [
          1,
          2
        ],
        [
          -1,
          -2
        ],
        [
          -2,
          1
        ],
        [
          -3,
          1
        ],
        [
          -4,
          1
        ],
        [
          -1,
          1
        ],
        [
          -2,
          0
        ],
        [
          -2,
          2
        ],
        [
          -2,
          -1
        ],
        [
          -4,
          2
        ],
        [
          -3,
          0
        ],
        [
          -4,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -2,
          5
        ],
        [
          -3,
          2
        ],
        [
          -3,
          3
        ],
        [
          -5,
          4
        ],
        [
          -3,
          4
        ],
        [
          -4,
          0
        ],
        [
          -1,
          0
        ],
        [
          -5,
          0
        ],
        [
          0,
          -1
        ],
        [
          -1,
          3
        ],
        [
          -2,
          3
        ],
        [
          -5,
          3
        ],
        [
          0,
          3
        ]
      ]
    },
    {
      "id": "oa-4d38ecd758048686",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 46,
      "placements": 46,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 58,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.005,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 47,
      "cert_fnv1a64_debug_v1": "81299d17fe55ef25",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ],
        [
          -7,
          0
        ],
        [
          -12,
          0
        ],
        [
          -10,
          -1
        ],
        [
          -14,
          3
        ],
        [
          -10,
          3
        ]
      ]
    },
    {
      "id": "oa-b7bbc03e1327f7b5",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 45,
      "placements": 45,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 57,
      "status": "WIN",
      "nodes": 1,
      "expansions": 0,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 0,
      "ms": 0.005,
      "certified": 1,
      "claimant": "P1",
      "cert_nodes": 1,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 47,
      "cert_fnv1a64_debug_v1": "a1e494eb2efc8b11",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ],
        [
          -7,
          0
        ],
        [
          -12,
          0
        ],
        [
          -10,
          -1
        ],
        [
          -14,
          3
        ]
      ]
    },
    {
      "id": "oa-42551fb87e915ae5",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 44,
      "placements": 44,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 56,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 150,
      "ms": 0.342,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 2,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 49,
      "cert_fnv1a64_debug_v1": "5a84f0bb26557967",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          0,
          1
        ],
        [
          1,
          0
        ],
        [
          1,
          1
        ],
        [
          -8,
          0
        ],
        [
          -1,
          2
        ],
        [
          -3,
          0
        ],
        [
          -2,
          3
        ],
        [
          -3,
          -1
        ],
        [
          -1,
          1
        ],
        [
          0,
          2
        ],
        [
          -1,
          3
        ],
        [
          0,
          3
        ],
        [
          1,
          3
        ],
        [
          -3,
          3
        ],
        [
          -8,
          -1
        ],
        [
          -8,
          -2
        ],
        [
          -8,
          1
        ],
        [
          1,
          2
        ],
        [
          -8,
          -3
        ],
        [
          -2,
          2
        ],
        [
          -8,
          -4
        ],
        [
          -2,
          1
        ],
        [
          -9,
          -2
        ],
        [
          -10,
          -1
        ],
        [
          -12,
          1
        ],
        [
          -7,
          -4
        ],
        [
          -10,
          -2
        ],
        [
          -11,
          -2
        ],
        [
          -7,
          -2
        ],
        [
          -12,
          -2
        ],
        [
          -10,
          0
        ],
        [
          -10,
          -3
        ],
        [
          -10,
          -4
        ],
        [
          -10,
          1
        ],
        [
          -9,
          -1
        ],
        [
          -11,
          -1
        ],
        [
          -13,
          -1
        ],
        [
          -7,
          -1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ],
        [
          -7,
          0
        ],
        [
          -12,
          0
        ],
        [
          -11,
          1
        ]
      ]
    },
    {
      "id": "oa-fcdeed9ad5ba60c9",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 43,
      "placements": 43,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 55,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 146,
      "ms": 0.447,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 49,
      "cert_fnv1a64_debug_v1": "8d36d4682724ea53",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ],
        [
          -7,
          0
        ],
        [
          -12,
          0
        ]
      ]
    },
    {
      "id": "oa-19b761730b30bb45",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 42,
      "placements": 42,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 54,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 146,
      "ms": 0.756,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ],
        [
          -7,
          0
        ]
      ]
    },
    {
      "id": "oa-f62d9f746c4c2884",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 41,
      "placements": 41,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 53,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 142,
      "ms": 1.138,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ],
        [
          -9,
          0
        ]
      ]
    },
    {
      "id": "oa-3c2762e55c2af7e2",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 40,
      "placements": 40,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 52,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 142,
      "ms": 0.28,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 2,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 45,
      "cert_fnv1a64_debug_v1": "b6b4dcef3d09b2fd",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ],
        [
          -11,
          0
        ]
      ]
    },
    {
      "id": "oa-9a06a71943b29908",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 39,
      "placements": 39,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 51,
      "status": "WIN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 138,
      "ms": 0.423,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 3,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": 45,
      "cert_fnv1a64_debug_v1": "0921fba2849d8279",
      "d6_verified": 12,
      "d6_mask": "0xfff",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ],
        [
          -8,
          1
        ]
      ]
    },
    {
      "id": "oa-394a5720b6bb3df9",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 38,
      "placements": 38,
      "side": "P1",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 50,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 138,
      "ms": 0.604,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ],
        [
          -14,
          1
        ]
      ]
    },
    {
      "id": "oa-da080614de44aa2a",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 37,
      "placements": 37,
      "side": "P1",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 100000,
      "horizon": 49,
      "status": "UNKNOWN",
      "nodes": 2,
      "expansions": 1,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 134,
      "ms": 0.86,
      "certified": 0,
      "claimant": null,
      "cert_nodes": 0,
      "cert_edges": 0,
      "cert_commutations": 0,
      "cert_zones": 0,
      "derived_horizon": null,
      "cert_fnv1a64_debug_v1": null,
      "d6_verified": 0,
      "d6_mask": "0x000",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ],
        [
          -12,
          1
        ]
      ]
    },
    {
      "id": "oa-26e8190400f6cc6d",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 36,
      "placements": 36,
      "side": "P0",
      "phase": "SecondStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 48,
      "status": "WIN",
      "nodes": 8,
      "expansions": 7,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 682,
      "ms": 1.516,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 15,
      "cert_edges": 7,
      "cert_commutations": 3,
      "cert_zones": 0,
      "derived_horizon": 45,
      "cert_fnv1a64_debug_v1": "68bb6da13372bdc9",
      "d6_verified": 6,
      "d6_mask": "0x1e3",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ],
        [
          -10,
          1
        ]
      ]
    },
    {
      "id": "oa-a5cfb27716697b1d",
      "source": "human:004759ff34cefdc2:winner=-1",
      "source_prefix": 35,
      "placements": 35,
      "side": "P0",
      "phase": "FirstStone",
      "orbit": 12,
      "cap": 10000,
      "horizon": 47,
      "status": "WIN",
      "nodes": 23,
      "expansions": 22,
      "tt_bytes": 536870912,
      "peak_tt_bytes": 6706,
      "ms": 2.412,
      "certified": 1,
      "claimant": "P0",
      "cert_nodes": 16,
      "cert_edges": 7,
      "cert_commutations": 3,
      "cert_zones": 0,
      "derived_horizon": 45,
      "cert_fnv1a64_debug_v1": "96415a7f19e786d5",
      "d6_verified": 6,
      "d6_mask": "0x3b1",
      "moves": [
        [
          0,
          0
        ],
        [
          1,
          -1
        ],
        [
          1,
          0
        ],
        [
          2,
          -1
        ],
        [
          -8,
          0
        ],
        [
          1,
          -2
        ],
        [
          -3,
          0
        ],
        [
          1,
          -3
        ],
        [
          -4,
          1
        ],
        [
          0,
          -1
        ],
        [
          2,
          -2
        ],
        [
          2,
          -3
        ],
        [
          3,
          -3
        ],
        [
          4,
          -3
        ],
        [
          0,
          -3
        ],
        [
          -9,
          1
        ],
        [
          -10,
          2
        ],
        [
          -7,
          -1
        ],
        [
          3,
          -2
        ],
        [
          -11,
          3
        ],
        [
          0,
          -2
        ],
        [
          -12,
          4
        ],
        [
          -1,
          -1
        ],
        [
          -11,
          2
        ],
        [
          -11,
          1
        ],
        [
          -11,
          -1
        ],
        [
          -11,
          4
        ],
        [
          -12,
          2
        ],
        [
          -13,
          2
        ],
        [
          -9,
          2
        ],
        [
          -14,
          2
        ],
        [
          -10,
          0
        ],
        [
          -13,
          3
        ],
        [
          -14,
          4
        ],
        [
          -9,
          -1
        ]
      ]
    }
  ]
};
