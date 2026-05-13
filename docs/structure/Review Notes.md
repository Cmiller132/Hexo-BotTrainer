Docs Consolidation

README.md (line 4) and Project.md (line 52) still describe the old game_engine / models_common / game_runner / hexo_resnet layout, while PROJECT_STRUCTURE.md (line 3) describes the current five-package design. I’d either archive Project.md as historical design or fold its still-relevant parts into docs/structure. This is probably the highest-value simplification because stale docs make the whole repo feel less coherent than it is.



Review Notes File

Review Notes.md (line 1) is 700+ lines of mostly empty per-file checklist. That creates a lot of apparent structure without much signal. I’d consider replacing it with a short review checklist by package, or generating per-file notes only when there is actual review content.



Training Epoch Modules

hexo_train now has a self-play epoch package instead of the old selectable stages package. Continue watching whether epoch/samples.py should remain one file or split only after real chunk IO makes that worthwhile.



Sample Buffer Scaffolding

hexo_utils.samples now keeps the mechanical sample buffer pieces together in `buffer.py`, keeps schema/record shapes in `records.py`, and keeps reusable target construction in `targets.py`. Revisit splitting `buffer.py` only when real chunk IO, durable indexes, or sampler logic become large enough to justify separate files.



Runner Skeleton

hexo_runner currently has modes/match.py, batch.py, evaluation.py, and selfplay.py, but they are all non-operational stubs, e.g. match.py (line 11). Since the actual shared loop is also still a placeholder in loop.py (line 27), I’d consider flattening runner modes until the loop/session layer exists.



Top-Level Prototype Artifacts

patterns.rs (line 1) looks like a Rust prototype/reference table but is not part of the Cargo workspace and is only mentioned by the review notes. I’d decide whether it belongs in hexo_engine, in docs/archive, or out of the repo. Top-level code-like files that are not build inputs are especially confusing.
