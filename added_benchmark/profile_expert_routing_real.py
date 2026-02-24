#!/usr/bin/env python3
"""
Real expert routing profiling: runs actual Qwen3-30B-A3B-NVFP4 inference
layer-by-layer on wikitext data and captures expert routing decisions.

Processes the full transformer forward pass (attention + MoE) to produce
realistic hidden states at each layer. Weights are NVFP4-dequantized to BF16
on-the-fly, one layer at a time, to fit in GPU memory.
"""
import json, time, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict

# === Model Config (Qwen3-30B-A3B-NVFP4) ===
MODEL_PATH = '/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3'
HIDDEN = 2048
INTER = 768
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48
NUM_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6

# FP4 E2M1 lookup table (low nibble = element 0, high nibble = element 1)
FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


# ═══════════════════════════════════════════════════════════════════
# Weight Loading
# ═══════════════════════════════════════════════════════════════════
class WeightLoader:
    """Load tensors from sharded safetensors with cached file handles."""

    def __init__(self, model_path):
        self.model_path = model_path
        with open(f'{model_path}/model.safetensors.index.json') as f:
            self.weight_map = json.load(f)['weight_map']
        self._handles = {}
        self._lut = None

    def _handle(self, key):
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                f'{self.model_path}/{filename}', framework='pt', device='cuda')
        return self._handles[filename]

    def get(self, key):
        return self._handle(key).get_tensor(key)

    def dequant(self, prefix):
        """Dequantize NVFP4 linear layer to BF16 on GPU."""
        w_u8 = self.get(f'{prefix}.weight')
        w_scale = self.get(f'{prefix}.weight_scale')
        w_scale2 = self.get(f'{prefix}.weight_scale_2')
        if self._lut is None:
            self._lut = FP4_LUT.to(w_u8.device)
        rows, packed = w_u8.shape
        low = (w_u8.to(torch.int32) & 0x0F)
        high = ((w_u8.to(torch.int32) >> 4) & 0x0F)
        unpacked = torch.stack([low, high], dim=-1).reshape(rows, packed * 2).long()
        fp4_vals = self._lut[unpacked]
        sf = w_scale.float().repeat_interleave(16, dim=1)
        return (fp4_vals * sf * w_scale2.float()).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════
# Transformer Components
# ═══════════════════════════════════════════════════════════════════
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    """x: (batch, heads, seq, head_dim), freqs: (seq, head_dim)."""
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def attention_forward(hidden, loader, layer_idx, rope_freqs):
    """GQA attention with QK norms and RoPE."""
    batch, seq, _ = hidden.shape
    prefix = f'model.layers.{layer_idx}.self_attn'

    # Load + dequantize projections
    q_w = loader.dequant(f'{prefix}.q_proj')  # (4096, 2048)
    k_w = loader.dequant(f'{prefix}.k_proj')  # (512, 2048)
    v_w = loader.dequant(f'{prefix}.v_proj')  # (512, 2048)

    q = (hidden @ q_w.T).reshape(batch, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    k = (hidden @ k_w.T).reshape(batch, seq, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = (hidden @ v_w.T).reshape(batch, seq, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    del q_w, k_w, v_w

    # QK norms (per-head RMSNorm)
    q_norm_w = loader.get(f'{prefix}.q_norm.weight')
    k_norm_w = loader.get(f'{prefix}.k_norm.weight')
    q = F.rms_norm(q.float(), (HEAD_DIM,), q_norm_w.float(), RMS_EPS).to(torch.bfloat16)
    k = F.rms_norm(k.float(), (HEAD_DIM,), k_norm_w.float(), RMS_EPS).to(torch.bfloat16)

    # RoPE (cast back to bf16 since cos/sin are float)
    q = apply_rope(q, rope_freqs[:seq]).to(torch.bfloat16)
    k = apply_rope(k, rope_freqs[:seq]).to(torch.bfloat16)

    # GQA: expand KV heads
    if NUM_KV_HEADS < NUM_HEADS:
        rep = NUM_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # Scaled dot-product attention (causal)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out.transpose(1, 2).reshape(batch, seq, NUM_HEADS * HEAD_DIM)
    del q, k, v

    o_w = loader.dequant(f'{prefix}.o_proj')  # (2048, 4096)
    result = attn_out @ o_w.T
    del o_w, attn_out
    return result


def moe_forward(hidden, loader, layer_idx):
    """MoE forward: gate routing + SwiGLU expert computation."""
    batch, seq, dim = hidden.shape
    h = hidden.reshape(-1, dim)  # (n, 2048)
    n = h.shape[0]
    prefix = f'model.layers.{layer_idx}.mlp'

    # ── Routing ──
    gate_w = loader.get(f'{prefix}.gate.weight')  # (128, 2048) BF16
    logits = h.float() @ gate_w.float().T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_idx = torch.topk(scores, TOP_K, dim=-1)  # (n, 8)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # norm_topk_prob

    # Count tokens per expert
    flat_exp = topk_idx.reshape(-1)
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=h.device)
    counts.scatter_add_(0, flat_exp.long(),
                        torch.ones(n * TOP_K, dtype=torch.int32, device=h.device))

    # ── Expert Computation ──
    output = torch.zeros_like(h)
    token_ids = torch.arange(n, device=h.device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    flat_scores = topk_scores.reshape(-1)

    for eid in range(NUM_EXPERTS):
        mask = flat_exp == eid
        if not mask.any():
            continue

        ep = f'{prefix}.experts.{eid}'
        gp = loader.dequant(f'{ep}.gate_proj')  # (768, 2048)
        up = loader.dequant(f'{ep}.up_proj')     # (768, 2048)
        dp = loader.dequant(f'{ep}.down_proj')   # (2048, 768)

        tokens = h[token_ids[mask]]          # (m, 2048)
        gate_out = tokens @ gp.T             # (m, 768)
        up_out = tokens @ up.T               # (m, 768)
        inter = F.silu(gate_out) * up_out    # SwiGLU
        exp_out = inter @ dp.T               # (m, 2048)

        w = flat_scores[mask].unsqueeze(1).to(torch.bfloat16)
        idx = token_ids[mask].unsqueeze(1).expand_as(exp_out).long()
        output.scatter_add_(0, idx, (exp_out * w))

        del gp, up, dp, tokens, gate_out, up_out, inter, exp_out

    return output.reshape(batch, seq, dim), counts


def layer_forward(hidden, loader, layer_idx, rope_freqs):
    """Full transformer layer: LN → Attention → Residual → LN → MoE → Residual."""
    # ── Attention block ──
    ln1 = loader.get(f'model.layers.{layer_idx}.input_layernorm.weight')
    normed = F.rms_norm(hidden.float(), (HIDDEN,), ln1.float(), RMS_EPS).to(torch.bfloat16)
    attn_out = attention_forward(normed, loader, layer_idx, rope_freqs)
    hidden = hidden + attn_out
    del normed, attn_out

    # ── MoE block ──
    ln2 = loader.get(f'model.layers.{layer_idx}.post_attention_layernorm.weight')
    normed = F.rms_norm(hidden.float(), (HIDDEN,), ln2.float(), RMS_EPS).to(torch.bfloat16)
    moe_out, counts = moe_forward(normed, loader, layer_idx)
    hidden = hidden + moe_out
    del normed, moe_out

    torch.cuda.empty_cache()
    return hidden, counts


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Real expert routing profiler')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[64, 128, 256],
                        help='Token batch sizes to profile')
    parser.add_argument('--num-chunks', type=int, default=3,
                        help='Number of text chunks per batch size')
    parser.add_argument('--num-layers', type=int, default=48,
                        help='Number of transformer layers to process (max 48)')
    parser.add_argument('--output', type=str, default=None,
                        help='Write dual-tile config JSON to this path')
    args = parser.parse_args()

    num_layers = min(args.num_layers, NUM_LAYERS)
    print(f"{'='*70}")
    print(f"Real Expert Routing Profiler — Qwen3-30B-A3B-NVFP4")
    print(f"  128 experts, top-8, H=2048, I=768")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Chunks per batch: {args.num_chunks}")
    print(f"  Layers: {num_layers}/{NUM_LAYERS}")
    print(f"{'='*70}")

    loader = WeightLoader(MODEL_PATH)

    # Load embeddings
    print("\nLoading embeddings...")
    embed = loader.get('model.embed_tokens.weight')  # (151936, 2048) BF16
    print(f"  Shape: {embed.shape}, dtype: {embed.dtype}")

    # Precompute RoPE frequencies
    max_seq = max(args.batch_sizes)
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, device='cuda').float() / HEAD_DIM))
    t = torch.arange(max_seq, device='cuda').float()
    freqs = torch.outer(t, inv_freq)
    rope_freqs = torch.cat([freqs, freqs], dim=-1)  # (max_seq, HEAD_DIM)

    # Load tokenizer + wikitext
    print("Loading tokenizer and wikitext-103...")
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    full_text = " ".join([t for t in dataset["text"] if t.strip()])
    all_tokens = tokenizer.encode(full_text)
    print(f"  Total tokens available: {len(all_tokens)}")

    # ── Profile each batch size ──
    all_results = {}  # batch_size -> list of (num_layers, 128) arrays

    for bs in args.batch_sizes:
        print(f"\n{'='*70}")
        print(f"BATCH SIZE: {bs} tokens")
        print(f"{'='*70}")

        chunk_results = []
        for chunk_idx in range(args.num_chunks):
            start = chunk_idx * bs * 2  # stride by 2*bs to avoid overlap
            if start + bs > len(all_tokens):
                print(f"  Chunk {chunk_idx}: not enough tokens, skipping")
                break

            token_ids = torch.tensor(all_tokens[start:start+bs], device='cuda')
            hidden = embed[token_ids].unsqueeze(0)  # (1, bs, 2048)

            layer_counts = []
            t0 = time.time()

            with torch.no_grad():
                for layer_idx in range(num_layers):
                    lt = time.time()
                    hidden, counts = layer_forward(
                        hidden, loader, layer_idx, rope_freqs)
                    layer_counts.append(counts.cpu().numpy())
                    elapsed = time.time() - lt
                    active = int((counts > 0).sum())
                    mx = int(counts.max())
                    med = int(counts[counts > 0].median()) if (counts > 0).any() else 0
                    print(f"  Layer {layer_idx:2d}: {elapsed:5.1f}s | "
                          f"active={active:3d}/{NUM_EXPERTS} "
                          f"max={mx:3d} med={med:2d}", flush=True)

            total = time.time() - t0
            chunk_results.append(np.stack(layer_counts))
            print(f"  Chunk {chunk_idx} done: {total:.1f}s")

        all_results[bs] = chunk_results

    # ── Aggregate Statistics ──
    print(f"\n{'='*70}")
    print("ROUTING STATISTICS (from real model inference)")
    print(f"{'='*70}")

    print(f"\n{'Batch':>6} | {'P30':>4} | {'P50':>4} | {'P70':>4} | {'P90':>4} | "
          f"{'Mean':>5} | {'Max':>4} | {'Idle%':>6}")
    print("-" * 60)

    summary = {}
    for bs in args.batch_sizes:
        if bs not in all_results or not all_results[bs]:
            continue
        all_counts = np.concatenate(all_results[bs], axis=0)  # (chunks*layers, 128)
        flat = all_counts.flatten()
        active = flat[flat > 0]

        if len(active) == 0:
            continue

        p30 = np.percentile(active, 30)
        p50 = np.percentile(active, 50)
        p70 = np.percentile(active, 70)
        p90 = np.percentile(active, 90)
        mean = active.mean()
        mx = flat.max()
        idle = np.mean(flat == 0) * 100

        print(f"{bs:>6} | {p30:>4.0f} | {p50:>4.0f} | {p70:>4.0f} | {p90:>4.0f} | "
              f"{mean:>5.1f} | {mx:>4.0f} | {idle:>5.1f}%")

        summary[bs] = dict(p30=p30, p50=p50, p70=p70, p90=p90,
                           mean=mean, max=mx, idle_pct=idle)

    if args.output and summary:
        json_config = {
            "model": "Qwen3-30B-A3B-NVFP4",
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "profiles": {}
        }
        for bs, s in sorted(summary.items()):
            json_config["profiles"][str(bs)] = {
                "threshold": int(s["p70"]),
                "p30": float(s["p30"]),
                "p50": float(s["p50"]),
                "p70": float(s["p70"]),
                "p90": float(s["p90"]),
                "mean": float(s["mean"]),
                "max": float(s["max"]),
                "idle_pct": float(s["idle_pct"]),
            }
        with open(args.output, 'w') as f:
            json.dump(json_config, f, indent=2)
        print(f"\nWrote dual-tile config to: {args.output}")

    # ── Per-layer breakdown ──
    if all_results:
        largest_bs = max(bs for bs in args.batch_sizes
                         if bs in all_results and all_results[bs])
        first_chunk = all_results[largest_bs][0]
        print(f"\nPer-layer breakdown (batch_size={largest_bs}, chunk 0):")
        print(f"{'Layer':>6} | {'P70':>4} | {'Active':>7} | {'Max':>4} | {'Idle%':>6}")
        print("-" * 40)
        for li in range(len(first_chunk)):
            c = first_chunk[li]
            act = c[c > 0]
            if len(act) == 0:
                continue
            p70 = np.percentile(act, 70)
            print(f"  {li:4d} | {p70:>4.0f} | {len(act):3d}/{NUM_EXPERTS} | "
                  f"{c.max():>4d} | {np.mean(c==0)*100:>5.1f}%")

    # ── Dual-tile recommendations ──
    print(f"\n{'='*70}")
    print("DUAL-TILE THRESHOLD RECOMMENDATIONS")
    print(f"{'='*70}")
    for bs, s in sorted(summary.items()):
        print(f"  batch_size={bs:3d}: threshold={s['p70']:.0f} "
              f"(P70={s['p70']:.0f}, mean={s['mean']:.1f}, idle={s['idle_pct']:.0f}%)")
    print(f"\n  Use: runner.set_dual_tile_profiles([small_g1, small_g2, large_g1, large_g2], "
          f"threshold=<P70>)")


if __name__ == '__main__':
    main()
