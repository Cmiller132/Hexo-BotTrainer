"""Static param-count audit for hexfield (no torch needed).

Replicates every module's parameter shapes from model.py / constants.py and
sums them, then compares each line against the spec §9 table.
"""

# constants
C = 96
F = 15
NUM_TOKENS = 8
HEADS = 4
HEAD_DIM = C // HEADS  # 24
MLP_RATIO = 2
VALUE_BINS = 65
BIAS_ROWS = 237
STV_HORIZONS = (2, 6, 16)


def linear(in_f, out_f, bias=True):
    return in_f * out_f + (out_f if bias else 0)


def layernorm(n):
    return 2 * n  # weight + bias


def hexnodeconv(cin, cout):
    # weight (7, cin, cout) + bias (cout,)
    return 7 * cin * cout + cout


report = []


def add(label, value):
    report.append((label, value))
    return value


# --- stem (7-tap 15->96 + LN) ---
stem_conv = hexnodeconv(F, C)             # 7*15*96 + 96
stem_ln = layernorm(C)                    # 2*96
add("stem (conv 7*15*96+96)", stem_conv)
add("stem LN (2*96)", stem_ln)
stem_total = stem_conv + stem_ln

# --- 6 conv blocks ---
# each: conv1 (C->C), ln1, conv2 (C->C), ln2
one_conv_block = hexnodeconv(C, C) + layernorm(C) + hexnodeconv(C, C) + layernorm(C)
conv_blocks = 6 * one_conv_block
add("one conv block", one_conv_block)
add("6 conv blocks", conv_blocks)

# --- 3 attention blocks ---
# each: ln1, q/k/v/out proj (C->C each), ln2, fc1 (C->2C), fc2 (2C->C)
attn_qkvo = 4 * linear(C, C)
attn_mlp = linear(C, MLP_RATIO * C) + linear(MLP_RATIO * C, C)
attn_lns = 2 * layernorm(C)
one_attn_block = attn_qkvo + attn_mlp + attn_lns
attn_blocks = 3 * one_attn_block
add("one attn block (QKVO+MLP+2LN)", one_attn_block)
add("  attn QKVO", attn_qkvo)
add("  attn MLP", attn_mlp)
add("  attn 2 LN", attn_lns)
add("3 attn blocks", attn_blocks)

# --- shared bias table ---
bias_table = BIAS_ROWS * HEADS
add("bias table 237*4", bias_table)

# --- tokens + final LN ---
tokens = NUM_TOKENS * C
final_ln = layernorm(C)
add("tokens 8*96", tokens)
add("final LN 2*96", final_ln)
tokens_finalln = tokens + final_ln

# --- policy + opp heads ---
# policy_conv (C->C), policy_head Linear(C->1); same for opp
policy_one = hexnodeconv(C, C) + linear(C, 1)
policy_heads = 2 * policy_one
add("one policy (conv C->C + Linear C->1)", policy_one)
add("policy + opp heads", policy_heads)

# --- value head ---
# value_reduction Linear(3C->C), value_head Linear(C->65)
value_reduction = linear(3 * C, C)
value_head = linear(C, VALUE_BINS)
value_total = value_reduction + value_head
add("value_reduction (288->96)", value_reduction)
add("value_head (96->65)", value_head)
add("value head total", value_total)

# --- aux reduction + tops ---
# aux_reduction Linear(3C->C); stv heads x3 Linear(C->65); moves_left Linear(C->65)
aux_reduction = linear(3 * C, C)
tops = (len(STV_HORIZONS) + 1) * linear(C, VALUE_BINS)  # 3 stv + 1 moves_left
aux_total = aux_reduction + tops
add("aux_reduction (288->96)", aux_reduction)
add("4 tops (96->65) [3 stv + moves_left]", tops)
add("aux total", aux_total)

grand = (
    stem_total
    + conv_blocks
    + attn_blocks
    + bias_table
    + tokens_finalln
    + policy_heads
    + value_total
    + aux_total
)

print("=== line items ===")
for label, value in report:
    print(f"{label:45s} {value:>10,d}")

print("\n=== spec §9 table comparison ===")
spec = {
    "stem (7-tap 15->96 + LN)": 10368,
    "6 conv blocks": 777600,
    "3 attention blocks (QKVO + MLP + LNs)": 224352,
    "shared bias table 237x4": 948,
    "tokens + final LN": 960,
    "policy + opp heads": 129410,
    "value head (288->96->65)": 34049,
    "aux reduction (288->96) + 4 tops (96->65)": 52964,
}
computed = {
    "stem (7-tap 15->96 + LN)": stem_total,
    "6 conv blocks": conv_blocks,
    "3 attention blocks (QKVO + MLP + LNs)": attn_blocks,
    "shared bias table 237x4": bias_table,
    "tokens + final LN": tokens_finalln,
    "policy + opp heads": policy_heads,
    "value head (288->96->65)": value_total,
    "aux reduction (288->96) + 4 tops (96->65)": aux_total,
}
for k in spec:
    s, cmp = spec[k], computed[k]
    flag = "OK" if s == cmp else "MISMATCH"
    print(f"{k:45s} spec={s:>9,d} computed={cmp:>9,d}  {flag}")

print(f"\nspec total      = {sum(spec.values()):,d}")
print(f"computed total  = {grand:,d}")
print(f"spec headline   = 1,230,651")
print(f"match headline  = {grand == 1230651}")
print(f"match table sum = {sum(spec.values()) == grand}")
