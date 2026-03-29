#!/usr/bin/env python3
import time
from tensorrt_llm import LLM, SamplingParams


def main():
    t0 = time.time()
    print("Loading with tensorrt_llm.LLM...", flush=True)
    llm = LLM(
        model="/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint",
        max_batch_size=1,
        max_seq_len=256,
    )
    print(f"Loaded in {time.time()-t0:.0f}s!", flush=True)

    out = llm.generate(
        ["What is 2+2? Answer:"],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    print(f"Output: {out[0].outputs[0].text}", flush=True)


if __name__ == "__main__":
    main()
