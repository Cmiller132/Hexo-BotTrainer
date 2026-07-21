"""V1 SOAK — shared position sources and state replay.

Position representation is a plain list of ``[q, r]`` placement moves from the
engine's opening (``api.new_game()``); ``build_state(moves)`` replays them into a
live ``HexoState`` for the deep-solve probe. This keeps every position source
(fresh self-play, forcing/spare corpus, human corpus) portable, JSON-serialisable
and re-solvable without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

REPO = Path("/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/v1-soak")
CORPUS_DIR = REPO / "packages" / "hexfield_eq" / "rust" / "corpus"
HUMAN_JSONL = Path(
    "/mnt/e/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl"
)


def build_state(moves):
    """Replay a [[q,r], ...] move list into a fresh HexoState. Raises on an
    illegal replay so a malformed position never silently becomes a legal one."""
    s = api.new_game()
    for q, r in moves:
        res = api.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        if res is None:
            raise ValueError(f"illegal replay at ({q},{r}) in {moves}")
    return s


def _parse_corpus_file(path: Path):
    """Parse a forcing/spare corpus file (POS header + nstones lines + END)."""
    out = []
    lines = iter(path.read_text().splitlines())
    for header in lines:
        header = header.strip()
        # Skip blanks, comment lines, and any stray END between blocks.
        if not header or header.startswith("#") or header == "END":
            continue
        assert header.startswith("POS "), f"bad header: {header}"
        meta = {}
        for tok in header.split()[1:]:
            k, _, v = tok.partition("=")
            meta[k] = v
        nstones = int(meta["nstones"])
        moves = []
        got = 0
        while got < nstones:
            line = next(lines).strip()
            if not line or line.startswith("#"):
                continue
            q, r = line.split()[:2]
            moves.append([int(q), int(r)])
            got += 1
        out.append(
            {
                "id": meta.get("id", ""),
                "attacker": int(meta.get("attacker", 0)),
                "expect_win": meta.get("expect") == "WIN",
                "expect": meta.get("expect", ""),
                "nstones": nstones,
                "moves": moves,
            }
        )
    return out


def load_forcing_corpus():
    return _parse_corpus_file(CORPUS_DIR / "forcing_corpus_moves.txt")


def load_spare_corpus():
    return _parse_corpus_file(CORPUS_DIR / "spare_corpus_moves.txt")


def load_human_positions(n_games, plies_per_game, *, seed=1234, min_ply=8, max_ply=None):
    """Sample OOD tactical positions from the human corpus. Games are the FIRST
    ``n_games`` lines of the jsonl (a file-order slice, NOT a random game sample
    — state this bias in the report). From each game we take up to
    ``plies_per_game`` interior prefixes chosen by a SEEDED RANDOM sample of ply
    indices in [min_ply, len-2] (then sorted); each prefix is a legal
    non-terminal position by construction (the human game was legal)."""
    import random

    rng = random.Random(seed)
    out = []
    with open(HUMAN_JSONL) as fh:
        for line_idx, line in enumerate(fh):
            if len(out) >= n_games * plies_per_game and line_idx > n_games:
                break
            g = json.loads(line)
            moves = g["moves"]
            hi = (max_ply or len(moves) - 2)
            hi = min(hi, len(moves) - 2)
            if hi <= min_ply:
                continue
            cand = list(range(min_ply, hi))
            rng.shuffle(cand)
            picked = sorted(cand[:plies_per_game])
            for p in picked:
                prefix = moves[:p]
                # Replay-validate; skip any game whose prefix hits an illegal
                # move (data noise) rather than aborting the whole slice.
                try:
                    build_state(prefix)
                except Exception:
                    continue
                out.append(
                    {
                        "id": f"human_{g['game_hash']}_p{p}",
                        "source": "human",
                        "moves": prefix,
                        "placements": p,
                        "elo": g.get("elo"),
                    }
                )
            if line_idx + 1 >= n_games:
                break
    return out
