# Training Info

## Goal

Training should follow a simple KataGo-style loop:

```text
self-play writes trainable samples
sample buffer stores and shuffles samples
training reads batches from the sample buffer
new checkpoints feed future self-play
```

The runner still writes game records, but those records are detached from
training. They exist for analysis, debugging, audit, and recordkeeping.

## Ownership

- `hexo_runner`: runs games and writes detached game records.
- `hexo_model_*`: decides what a trainable sample means and writes samples
  during self-play.
- `hexo_train`: orchestrates training stages and selects deterministic D6
  symmetries for sampled positions.
- `hexo_utils.samples`: provides shared sample schemas, chunk writing, indexing,
  shuffling, sampling, and default policy/value helpers.
- `hexo_engine`: owns legal moves, transitions, terminal state, and snapshots.

## Self-Play Sample Generation

For each model decision:

1. Runner asks the model-backed player for an action.
2. The model encodes the position, for example a `33x33` crop.
3. Inference/search produces legal-action policy data and a selected action.
4. The model sample writer stores a pending sample.
5. Runner applies the action through the engine.
6. After self-play returns, `hexo_train` calls the model finalizer to attach
   value targets.
7. Finalized samples are appended to the model's sample buffer.

The normal training path should not scan finished runner game records to invent
targets after the fact. Samples are created as self-play runs.

## ResNet Sample Shape

For `hexo_model_resnet`, use compact storage:

```text
binary/int planes: uint8[12, 33, 33]
scalar globals: float32[3]
legal mask: uint8/bool[33, 33]
threat mask: optional uint8/bool[33, 33]
policy target: dense or sparse policy over legal actions
value target: float32, finalized after game end
metadata: compact ids for game, turn, player, model/checkpoint, seed
```

Store scalar globals as three numbers, not full constant planes. During batch
decode, CPU workers can expand them into planes if the model expects that shape.

## Buffer Plan

Samples should be written in chunks rather than kept as millions of Python
objects. A first buffer should support:

- append finalized samples,
- flush chunk files,
- keep a compact index,
- sample or iterate shuffled batches,
- accept deterministic D6 symmetry selections from `hexo_train`,
- decode compact storage into training tensors.

On a 32 GB machine, keep samples compact in storage and use CPU workers to
decode batches. The `7950x` is well suited for expanding `uint8` planes,
applying D6 transforms, building masks, and converting tensors to `float32` or
`float16` while the GPU trains.

## Recommended Defaults

- Store binary planes and masks as `uint8`; bit-pack later if needed.
- Store scalar globals as `float32[3]`.
- Keep policy targets dense at first for simplicity; switch to sparse if memory
  or disk pressure becomes important.
- Decode to model tensors at training time, not during sample writing.
- Keep runner records optional for sample provenance, not as the training
  source of truth.

This mirrors the useful part of KataGo's split: self-play continuously produces
training data, a shuffle/buffer layer prepares it, and training consumes that
prepared data. Exporting and gatekeeping can be added later when the basic loop
is stable.
