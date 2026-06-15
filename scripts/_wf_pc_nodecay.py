"""Static no-decay classification audit (mirrors plugin.py logic, no torch).

Enumerates every named parameter of HexfieldNet with its ndim, then applies the
plugin's rule: decay iff ndim>=2 AND 'bias_table' not in name AND name != 'tokens'.
Checks against spec §6.3 intent: decay = matrix weights only (conv weights +
Linear weights); no_decay = biases, LN params, tokens, bias_table.
"""

C = 96
F = 15
NUM_TOKENS = 8
HEADS = 4
VALUE_BINS = 65
BIAS_ROWS = 237
MLP_RATIO = 2
STV = (2, 6, 16)

# (name, shape) for every parameter, mirroring model.py construction
params = []


def conv(prefix, cin, cout):
    params.append((f"{prefix}.weight", (7, cin, cout)))
    params.append((f"{prefix}.bias", (cout,)))


def lin(prefix, inf, outf):
    params.append((f"{prefix}.weight", (outf, inf)))
    params.append((f"{prefix}.bias", (outf,)))


def ln(prefix, n):
    params.append((f"{prefix}.weight", (n,)))
    params.append((f"{prefix}.bias", (n,)))


conv("stem", F, C)
ln("stem_ln", C)
for i in range(6):
    conv(f"conv_blocks.{i}.conv1", C, C)
    ln(f"conv_blocks.{i}.ln1", C)
    conv(f"conv_blocks.{i}.conv2", C, C)
    ln(f"conv_blocks.{i}.ln2", C)
for i in range(3):
    ln(f"attn_blocks.{i}.ln1", C)
    lin(f"attn_blocks.{i}.attn.q_proj", C, C)
    lin(f"attn_blocks.{i}.attn.k_proj", C, C)
    lin(f"attn_blocks.{i}.attn.v_proj", C, C)
    lin(f"attn_blocks.{i}.attn.out_proj", C, C)
    ln(f"attn_blocks.{i}.ln2", C)
    lin(f"attn_blocks.{i}.fc1", C, MLP_RATIO * C)
    lin(f"attn_blocks.{i}.fc2", MLP_RATIO * C, C)
params.append(("tokens", (NUM_TOKENS, C)))
params.append(("bias_table", (BIAS_ROWS, HEADS)))
ln("ln_final", C)
conv("policy_conv", C, C)
lin("policy_head", C, 1)
conv("opp_policy_conv", C, C)
lin("opp_policy_head", C, 1)
lin("value_reduction", 3 * C, C)
lin("value_head", C, VALUE_BINS)
lin("aux_reduction", 3 * C, C)
for h in STV:
    lin(f"stv_heads.{h}", C, VALUE_BINS)
lin("moves_left_head", C, VALUE_BINS)


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# plugin rule
def is_decay(name, shape):
    ndim = len(shape)
    return ndim >= 2 and "bias_table" not in name and name != "tokens"


# spec intent: decay = matrix weights = conv .weight (ndim3) + Linear .weight (ndim2, not tokens/bias_table)
def spec_decay(name, shape):
    # matrix weights only: a 2-D Linear weight or a 3-D conv weight, excluding tokens/bias_table
    if name in ("tokens", "bias_table"):
        return False
    return name.endswith(".weight") and len(shape) >= 2


decay_total = 0
nodecay_total = 0
disagreements = []
for name, shape in params:
    n = numel(shape)
    pd = is_decay(name, shape)
    sd = spec_decay(name, shape)
    if pd:
        decay_total += n
    else:
        nodecay_total += n
    if pd != sd:
        disagreements.append((name, shape, pd, sd))

print(f"total params : {decay_total + nodecay_total:,d}")
print(f"decay        : {decay_total:,d}")
print(f"no_decay     : {nodecay_total:,d}")
print(f"num params   : {len(params)}")
print()
if disagreements:
    print("DISAGREEMENTS (plugin vs spec-intent):")
    for name, shape, pd, sd in disagreements:
        print(f"  {name} {shape} plugin_decay={pd} spec_decay={sd}")
else:
    print("plugin rule == spec-intent for all parameters: OK")

# Sanity: every no_decay param should be a bias, LN param, tokens, or bias_table
print("\nno_decay members not matching {*.bias, *_ln.*, ln*.*, tokens, bias_table}:")
suspicious = []
for name, shape in params:
    if not is_decay(name, shape):
        ok = (
            name.endswith(".bias")
            or name in ("tokens", "bias_table")
            or ".ln" in name
            or name.startswith("ln_final")
            or name.startswith("stem_ln")
        )
        if not ok:
            suspicious.append((name, shape))
print("  none" if not suspicious else suspicious)
