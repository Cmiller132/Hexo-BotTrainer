Docs Consolidation

README.md (line 4) and Project.md (line 52) still describe the old game_engine / models_common / game_runner / hexo_resnet layout, while PROJECT_STRUCTURE.md (line 3) describes the current five-package design. I’d either archive Project.md as historical design or fold its still-relevant parts into docs/structure. This is probably the highest-value simplification because stale docs make the whole repo feel less coherent than it is.



Review Notes File

Review Notes.md (line 1) is 700+ lines of mostly empty per-file checklist. That creates a lot of apparent structure without much signal. I’d consider replacing it with a short review checklist by package, or generating per-file notes only when there is actual review content.



Training Stage Modules

hexo_train has a good central pipeline in pipeline.py (line 59), but the stage implementations are split across many small files, several of which are thin placeholders. The stage package may be worth consolidating into fewer lifecycle modules, for example checkpoint.py, samples.py, and execution.py, or temporarily folding no-op stages into the pipeline until they grow real behavior.



Sample Buffer Scaffolding

hexo_utils.samples is split into store, index, window, writer, sampling, schema, records, and targets. Some files are only placeholder dataclasses, like store.py (line 1), index.py (line 1), and window.py (line 1). I’d investigate combining the mechanical buffer pieces into one buffer.py or storage.py until real chunk/index logic requires separation.



Runner Skeleton

hexo_runner currently has modes/match.py, batch.py, evaluation.py, and selfplay.py, but they are all non-operational stubs, e.g. match.py (line 11). Since the actual shared loop is also still a placeholder in loop.py (line 27), I’d consider flattening runner modes until the loop/session layer exists.



Top-Level Prototype Artifacts

patterns.rs (line 1) looks like a Rust prototype/reference table but is not part of the Cargo workspace and is only mentioned by the review notes. I’d decide whether it belongs in hexo_engine, in docs/archive, or out of the repo. Top-level code-like files that are not build inputs are especially confusing.