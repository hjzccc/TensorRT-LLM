#!/usr/bin/env python3
import sys

sys.path.insert(0, "/code/tensorrt_llm/scripts/nvfp4_compress")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")

from direct_weight_store import DirectWeightStore
import spike1_ground_truth

spike1_ground_truth.WeightStore = DirectWeightStore

import exact_docker_eval

sys.argv = [exact_docker_eval.__file__, "--configs", "uniform_nvfp4", "--output", "/tmp/eval_direct.json"]
exact_docker_eval.main()
