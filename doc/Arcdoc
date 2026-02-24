ModelEngine.forward()
  └─ _set_up_attn_metadata()
       └─ Creates AttentionMetadata with:
            - kv_cache_manager: manages paged KV cache blocks
            - kv_cache_params: {use_cache=True, num_cached_tokens_per_seq=[...]}
            - kv_cache_block_offsets: maps sequences → GPU memory blocks
  └─ model.forward(input_ids, attn_metadata)
       └─ for each DecoderLayer:
            ├─ self.self_attn(hidden_states, attn_metadata)  ← KV cache read/write HERE
            │    └─ Prefill: compute K,V for all tokens → WRITE to KV cache pages
            │    └─ Decode:  compute K,V for 1 new token → APPEND to KV cache,
            │                READ cached K,V for attention
            │
            └─ self.mlp(hidden_states, attn_metadata)        ← MoE runs here
                 └─ run_moe(x)  ← just processes x, no KV cache involvement