#!/usr/bin/env python3
"""Check if loss accumulation is causing NaN/Inf."""
import math

# Simulate loss accumulation
losses = []

# Add some reasonable losses
for i in range(144):  # 145 chunks - 1
    losses.append(10.5)  # Reasonable loss value

# Check if any are NaN or Inf
print(f"Total losses: {len(losses)}")
print(f"Sum: {sum(losses)}")
print(f"Mean: {sum(losses) / len(losses)}")

# Compute PPL
mean_loss = sum(losses) / len(losses)
ppl = math.exp(mean_loss)
print(f"PPL: {ppl:.4f}")

# Now add a NaN
losses_with_nan = losses.copy()
losses_with_nan.append(float('nan'))

print(f"\nWith NaN:")
print(f"Sum: {sum(losses_with_nan)}")
print(f"Mean: {sum(losses_with_nan) / len(losses_with_nan)}")
ppl_nan = math.exp(sum(losses_with_nan) / len(losses_with_nan))
print(f"PPL: {ppl_nan}")

# Check if this matches the observed PPL
observed_ppl = 1817424.66
print(f"\nObserved PPL: {observed_ppl}")
print(f"log(observed): {math.log(observed_ppl):.4f}")

# What loss would give this PPL?
required_loss = math.log(observed_ppl)
print(f"Required mean loss: {required_loss:.4f}")

# If we have 145 chunks and one is NaN, what would the sum be?
# sum([10.5]*144 + [nan]) = nan
# mean = nan / 145 = nan
# exp(nan) = nan

# But the result is a number, not nan. So the issue is different.
# Maybe the loss is being computed incorrectly?

# Let's check: if the loss is computed as sum of all losses / num_losses
# and one loss is very large, what would happen?

large_loss = math.log(observed_ppl) * 145  # This would give the observed PPL
print(f"\nIf one loss is {large_loss:.4f}, mean would be {large_loss/145:.4f}, PPL would be {math.exp(large_loss/145):.4f}")

# Or if the loss is computed incorrectly as sum(losses) instead of mean(losses)?
print(f"\nIf PPL = exp(sum(losses)) instead of exp(mean(losses)):")
print(f"  sum([10.5]*145) = {10.5*145}")
print(f"  exp(1522.5) = {math.exp(1522.5)}")

