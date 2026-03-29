#!/usr/bin/env python3
import psutil, os, gc, torch, json, time
from safetensors import safe_open
from safetensors.torch import save_file

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

snap = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
print(f"Initial RSS: {mem_gb():.1f} GB", flush=True)

with safe_open(f"{snap}/model.safetensors-00001-of-00014.safetensors", framework="pt", device="cpu") as sf:
    for key in list(sf.keys()):
        t = sf.get_tensor(key)
        print(f"Loaded {key}: {t.shape} {t.dtype} | RSS={mem_gb():.1f}GB", flush=True)

        if "gate_up_proj" in key:
            half = t.shape[1] // 2
            out = {}
            for eidx in range(t.shape[0]):
                w = t[eidx, :half, :].contiguous().cuda()
                gs = fp4_global_scale(w).to(torch.float32)
                packed, bs = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
                out[f"e{eidx}.w"] = packed.cpu()
                out[f"e{eidx}.s"] = bs.cpu()
                out[f"e{eidx}.s2"] = gs.cpu().reshape(1)
                del w, packed, bs, gs
                torch.cuda.empty_cache()
                if eidx % 64 == 0:
                    print(f"  Expert {eidx}/256 | RSS={mem_gb():.1f}GB | dict_keys={len(out)}", flush=True)

            print(f"After gate experts: RSS={mem_gb():.1f}GB | {len(out)} tensors", flush=True)
            save_file(out, "/tmp/test_shard.safetensors")
            print(f"After save: RSS={mem_gb():.1f}GB", flush=True)
            del out
            gc.collect()
            print(f"After gc: RSS={mem_gb():.1f}GB", flush=True)

        del t
        gc.collect()
        print(f"After del t + gc: RSS={mem_gb():.1f}GB", flush=True)
