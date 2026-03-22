#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/channel_quant_new')
sys.path.insert(0, '/workspace/channel_quant')

import exact_docker_eval as ede
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(ede.MODEL_ID, trust_remote_code=True)
    eval_ids, nsamples, seqlen = ede.load_eval_data(tokenizer, 2048)
    nsamples = 4  # quick exact sanity on first 4 chunks only

    snapshot_dir, root_config, weight_map = ede.load_root_config(ede.MODEL_ID)
    config = ede.build_text_config(root_config)
    device = ede.torch.device('cuda')
    dtype = ede.torch.float16

    results = {}
    for run_config in ede.default_run_configs():
        ppl = ede.evaluate_ppl(eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, dtype, run_config)
        results[run_config.label] = {
            'ppl': round(ppl, 4),
            'nsamples': nsamples,
            'seqlen': seqlen,
        }
        print(f'{run_config.label}: PPL={ppl:.4f}')

    out = {'metadata': {'nsamples': nsamples, 'seqlen': seqlen}, 'results': results}
    Path('/workspace/exact_docker_eval_small.json').write_text(json.dumps(out, indent=2))
    print('Saved -> /workspace/exact_docker_eval_small.json')

if __name__ == '__main__':
    main()
