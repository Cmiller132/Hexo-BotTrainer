from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexfield_eq import _rust


def play(state, coords):
    for q, r in coords:
        api.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return state


DEEP_WIN_MOVES = [(0, 0), (0, 8), (2, 7), (1, 0), (2, 0), (4, 6), (6, 5), (0, 4),
                  (1, 4), (8, 4), (10, 3), (2, 4), (16, 0), (12, 2), (14, 1)]
s = play(api.new_game(), DEEP_WIN_MOVES)
for wide in (False, True):
    r = _rust.hexfield_eq_deep_solve_probe(s, 20000, "both", 16, False, False, wide, True)
    print("wide=%s status=%s depth=%s stats_nodes=%s deep_nodes=%s cert_univ=%s "
          "gate_eval=%s dismiss=%s vfail=%s certv=%s wall_us=%.1f" % (
              wide, r["status"], r["cert_depth"], r.get("stats_nodes"), r["deep_nodes"],
              r["cert_universal_nodes"], r.get("stats_interior_gate_evaluations"),
              r.get("stats_interior_gate_dismissals"), r["deep_verify_failed"],
              r["cert_version"], r["wall_nanos"] / 1000))

q = play(api.new_game(), [(0, 0), (0, 8), (2, 7)])
r = _rust.hexfield_eq_deep_solve_probe(q, 500, "both", 16, True, False, True, True)
print("quiet ladder status=%s horizon_cut=%s tall=%s kb_death=%s deep_nodes=%s" % (
    r["status"], r["horizon_cut"], r["horizon_cut_tall"], r["deep_kb_death"], r["deep_nodes"]))
print("OK")
