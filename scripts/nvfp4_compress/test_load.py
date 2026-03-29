#!/usr/bin/env python3
import time
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm.sampling_params import SamplingParams


def main():
    t0 = time.time()
    print("Loading Sehyo/Qwen3.5-35B-A3B-NVFP4 with auto_deploy...", flush=True)
    llm = LLM(
        model="Sehyo/Qwen3.5-35B-A3B-NVFP4",
        world_size=1,
        max_seq_len=512,
        max_batch_size=1,
        trust_remote_code=True,
    )
    print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

    out = llm.generate(
        [{"prompt": "What is 2+2? Answer:"}],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    print(f"Output: {out[0].outputs[0].text}", flush=True)
    llm.shutdown()


if __name__ == '__main__':
    main()
