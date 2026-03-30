# Safe Batching Practices for `_loglikelihood_tokens` in lm-eval 0.4.x

## Overview

This document provides guidance on implementing safe batching for the `_loglikelihood_tokens` method in custom lm-eval model wrappers (e.g., `NVFP4LM`). The current implementation in `lm_eval_nvfp4.py` processes requests sequentially; this guide explains how to batch them safely while maintaining compatibility with lm-eval's evaluation framework.

---

## 1. Request Structure & Interface Contract

### Input Format
The `_loglikelihood_tokens` method receives a list of requests, each containing:

```python
(
    (context_str, continuation_str),  # Tuple of strings (for caching/logging)
    context_tokens,                    # list[int] - pre-tokenized context
    continuation_tokens                # list[int] - pre-tokenized continuation
)
```

**Key Point**: Tokens are **already encoded** by `TemplateLM.loglikelihood()` before reaching your method. You receive token IDs, not raw text.

### Output Format
Must return a list of tuples matching input length exactly:

```python
list[tuple[float, bool]]  # [(log_prob: float, is_greedy: bool), ...]
```

- **log_prob**: Sum of log-probabilities for continuation tokens
- **is_greedy**: Whether argmax predictions match actual continuation tokens

---

## 2. Request Ordering Requirement

**CRITICAL**: You **MUST preserve input request order** in the output list.

### Why?
The lm-eval evaluator (`simple_evaluate()`) relies on result ordering to match outputs back to original requests. Reordering results will corrupt evaluation metrics.

### Current Pattern (Sequential)
```python
def _loglikelihood_tokens(self, requests, disable_tqdm=False, **_):
    results = []
    for request_key, context_enc, continuation_enc in requests:
        results.append(self._score_context_continuation(...))
    return results  # Order preserved by iteration
```

### Safe Batching Pattern
```python
def _loglikelihood_tokens(self, requests, disable_tqdm=False, **_):
    results = [None] * len(requests)  # Pre-allocate with correct length
    
    # Process in batches
    for batch_start in range(0, len(requests), self.batch_size):
        batch_end = min(batch_start + self.batch_size, len(requests))
        batch_indices = range(batch_start, batch_end)
        batch_requests = [requests[i] for i in batch_indices]
        
        # Process batch
        batch_results = self._score_batch(batch_requests)
        
        # Store results in original positions
        for idx, result in zip(batch_indices, batch_results):
            results[idx] = result
    
    return results  # Order preserved by index mapping
```

---

## 3. Padding & Grouping Strategy

### Can You Avoid Padding?
**Yes, safely.** You can group requests by exact input length to avoid padding masks.

### Why This Works
- Each request is independent (no cross-request dependencies)
- `_score_context_continuation()` already handles variable-length inputs
- Grouping by length reduces wasted computation on padding tokens

### Safe Grouping Pattern
```python
from collections import defaultdict

def _loglikelihood_tokens(self, requests, disable_tqdm=False, **_):
    results = [None] * len(requests)
    
    # Group by total input length
    length_groups = defaultdict(list)
    for idx, (_, context_enc, continuation_enc) in enumerate(requests):
        total_len = len(context_enc) + len(continuation_enc)
        length_groups[total_len].append((idx, requests[idx]))
    
    # Process each length group
    for total_len, group in length_groups.items():
        indices = [idx for idx, _ in group]
        group_requests = [req for _, req in group]
        
        # Process group (can batch within same length)
        group_results = self._score_batch(group_requests)
        
        # Store in original positions
        for idx, result in zip(indices, group_results):
            results[idx] = result
    
    return results
```

### Trade-offs
| Approach | Pros | Cons |
|----------|------|------|
| **No grouping** | Simplest, minimal code changes | Padding overhead if lengths vary |
| **Group by length** | Eliminates padding, faster compute | Slightly more complex, memory overhead for groups |
| **Adaptive batching** | Optimal throughput | Complex, requires profiling |

**Recommendation**: Start with no grouping (simplest), profile, then add grouping if padding is a bottleneck.

---

## 4. Batching Implementation Pattern

### Minimal Batching (No Grouping)
```python
def _loglikelihood_tokens(self, requests, disable_tqdm=False, **_):
    results = [None] * len(requests)
    
    for batch_start in range(0, len(requests), self.batch_size):
        batch_end = min(batch_start + self.batch_size, len(requests))
        batch_indices = range(batch_start, batch_end)
        
        # Collect batch
        batch_data = []
        for idx in batch_indices:
            request_key, context_enc, continuation_enc = requests[idx]
            batch_data.append((request_key, context_enc, continuation_enc))
        
        # Process batch (vectorized or loop)
        for i, (request_key, context_enc, continuation_enc) in enumerate(batch_data):
            result = self._score_context_continuation(
                request_key, context_enc, continuation_enc
            )
            results[batch_start + i] = result
    
    return results
```

### Vectorized Batching (Advanced)
If you want to vectorize `_score_context_continuation()` for true batching:

```python
def _score_batch(self, batch_requests):
    """Process multiple requests in a single forward pass."""
    batch_size = len(batch_requests)
    max_len = max(
        len(ctx) + len(cont)
        for _, ctx, cont in batch_requests
    )
    
    # Pad all sequences to max_len
    padded_inputs = []
    for _, context_enc, continuation_enc in batch_requests:
        total = context_enc + continuation_enc
        padded = total + [self.pad_token_id] * (max_len - len(total))
        padded_inputs.append(padded)
    
    # Single forward pass
    input_ids = torch.tensor(padded_inputs, device=self._torch_device)
    logits = self._model_call(input_ids)  # Shape: [batch_size, seq_len, vocab_size]
    
    # Extract results per request
    results = []
    for i, (request_key, context_enc, continuation_enc) in enumerate(batch_requests):
        # Extract continuation logits for this request
        cont_start = len(context_enc)
        cont_logits = logits[i, cont_start:cont_start+len(continuation_enc)]
        
        # Compute log-probs and greedy match
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_logprobs = log_probs[range(len(continuation_enc)), continuation_enc]
        
        answer = (
            float(token_logprobs.sum().item()),
            bool(torch.all(cont_logits.argmax(dim=-1) == torch.tensor(continuation_enc)))
        )
        
        if request_key is not None:
            self.cache_hook.add_partial("loglikelihood", request_key, answer)
        
        results.append(answer)
    
    return results
```

---

## 5. Cache Hook Compatibility

### Current Behavior
The `cache_hook.add_partial()` call in `_score_context_continuation()` caches results by `request_key`:

```python
if request_key is not None:
    self.cache_hook.add_partial("loglikelihood", request_key, answer)
```

### Batching Considerations
- **Order-independent**: Cache uses `request_key` (the string tuple), not position
- **Safe to batch**: Caching works correctly regardless of processing order
- **No changes needed**: Keep the cache call as-is in batched code

---

## 6. Gotchas & Common Mistakes

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Result reordering** | Evaluation metrics don't match baseline | Use index mapping to preserve order |
| **Padding token mismatch** | NaN logits or crashes | Ensure `pad_token_id` matches model's tokenizer |
| **Batch size too large** | OOM errors | Reduce `self.batch_size` or use gradient checkpointing |
| **Forgetting cache hook** | Cache misses, slower subsequent runs | Call `cache_hook.add_partial()` for every result |
| **Mixing string & token inputs** | Type errors | Remember: inputs are **already tokenized** |

---

## 7. Recommended Implementation for NVFP4LM

### Step 1: Add Batch Size Parameter
```python
class NVFP4LM(LmEvalWrapper):
    def __init__(self, ...):
        super().__init__(...)
        self.batch_size = 128  # Tune based on GPU memory
```

### Step 2: Implement Batched Method
```python
def _loglikelihood_tokens(self, requests, disable_tqdm=False, **_):
    results = [None] * len(requests)
    iterator = tqdm(
        range(0, len(requests), self.batch_size),
        total=(len(requests) + self.batch_size - 1) // self.batch_size,
        desc="Running loglikelihood requests",
        disable=(disable_tqdm or self.rank != 0),
    )
    
    for batch_start in iterator:
        batch_end = min(batch_start + self.batch_size, len(requests))
        
        for idx in range(batch_start, batch_end):
            request_key, context_enc, continuation_enc = requests[idx]
            results[idx] = self._score_context_continuation(
                request_key, context_enc, continuation_enc
            )
    
    return results
```

### Step 3: Verify Order Preservation
```python
# Test: Process same requests in different batch sizes
# Results should be identical regardless of batch_size
```

---

## 8. Verification Checklist

Before deploying batched code:

- [ ] Output list length matches input list length
- [ ] Results are in same order as input requests
- [ ] Cache hook is called for every request
- [ ] No NaN or inf values in log-probabilities
- [ ] Evaluation metrics match sequential baseline
- [ ] Memory usage is acceptable for your batch size
- [ ] Throughput improvement justifies added complexity

---

## Summary

**Safe batching for `_loglikelihood_tokens` requires:**

1. **Preserve request order** via index mapping, not reordering
2. **Optional grouping by length** to reduce padding overhead
3. **Keep cache hook calls** for every result
4. **Test against sequential baseline** to verify correctness

The current sequential implementation in `NVFP4LM` is correct and safe. Batching is an optimization that requires careful order preservation but is compatible with lm-eval's evaluation framework.
