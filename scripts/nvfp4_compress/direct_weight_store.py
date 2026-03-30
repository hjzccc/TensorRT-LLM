"""Drop-in replacement for WeightStore that uses direct file IO instead of safetensors mmap.

Prevents page cache accumulation that triggers systemd-oomd on low-RAM systems.
"""
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

DTYPE_MAP = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "U8": (torch.uint8, 1),
}


def _parse_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size
    return header, data_offset


def _read_tensor(path, header, data_offset, key):
    meta = header[key]
    shape = meta["shape"]
    dtype_str = meta["dtype"]
    torch_dtype, bpe = DTYPE_MAP[dtype_str]
    start = data_offset + meta["data_offsets"][0]
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]

    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(nbytes)

    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(torch_dtype).reshape(shape)


class DirectWeightStore:
    def __init__(self, model_id, snapshot_dir, weight_map):
        self.model_id = model_id
        self.snapshot_dir = snapshot_dir
        self.weight_map = weight_map
        self._downloaded = {}
        self._headers = {}

    def _ensure_file(self, filename):
        if filename not in self._downloaded:
            path = hf_hub_download(repo_id=self.model_id, filename=filename, repo_type="model")
            self._downloaded[filename] = Path(path)
        return self._downloaded[filename]

    def _get_header(self, shard_path):
        shard_path = str(shard_path)
        if shard_path not in self._headers:
            self._headers[shard_path] = _parse_header(shard_path)
        return self._headers[shard_path]

    def load_tensors(self, keys):
        grouped = defaultdict(list)
        for key in keys:
            grouped[self.weight_map[key]].append(key)

        tensors = {}
        for filename, shard_keys in grouped.items():
            shard_path = self._ensure_file(filename)
            header, data_offset = self._get_header(shard_path)
            print(f"Loading shard {filename} for {len(shard_keys)} tensor(s)", flush=True)
            for key in shard_keys:
                tensors[key] = _read_tensor(str(shard_path), header, data_offset, key)
        return tensors
