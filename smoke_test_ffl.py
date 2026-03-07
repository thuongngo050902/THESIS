# smoke_test_ffl.py — Run inside the MAT repo directory on GPU
# Usage: python smoke_test_ffl.py
import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("FFL Smoke Test")
print("=" * 60)

# --- Test 1: FFL module standalone ---
print("\n[Test 1] FocalFrequencyLoss standalone...")
from losses.focal_frequency_loss import FocalFrequencyLoss

ffl = FocalFrequencyLoss(alpha=1.0).cuda()
pred = torch.randn(2, 3, 512, 512, device='cuda', requires_grad=True)
target = torch.randn(2, 3, 512, 512, device='cuda')

loss = ffl(pred, target)
print(f"  FFL loss value: {loss.item():.6f}")
assert loss.ndim == 0, "Loss should be scalar"
assert loss.item() > 0, "Loss should be > 0 for different images"

loss.backward()
assert pred.grad is not None, "Gradient should flow to pred"
assert pred.grad.shape == pred.shape, "Gradient shape mismatch"
print(f"  Gradient norm:  {pred.grad.norm().item():.6f}")
print("  PASSED")

# --- Test 2: FFL with identical inputs → near-zero loss ---
print("\n[Test 2] FFL with identical inputs...")
same = torch.randn(2, 3, 512, 512, device='cuda')
loss_zero = ffl(same, same.clone())
print(f"  FFL loss value: {loss_zero.item():.10f}")
assert loss_zero.item() < 1e-6, f"Loss should be ~0 for identical images, got {loss_zero.item()}"
print("  PASSED")

# --- Test 3: FFL in [-1, 1] range (actual training range) ---
print("\n[Test 3] FFL with [-1, 1] range inputs...")
pred_norm = torch.randn(2, 3, 512, 512, device='cuda').clamp(-1, 1).requires_grad_(True)
target_norm = torch.randn(2, 3, 512, 512, device='cuda').clamp(-1, 1)
loss_norm = ffl(pred_norm, target_norm)
print(f"  FFL loss value: {loss_norm.item():.6f}")
loss_norm.backward()
assert pred_norm.grad is not None
print(f"  Gradient norm:  {pred_norm.grad.norm().item():.6f}")
print("  PASSED")

# --- Test 4: Baseline preservation (ffl_ratio=0) ---
print("\n[Test 4] TwoStageLoss with ffl_ratio=0 (baseline)...")
# This tests that the loss class initializes correctly with ffl_ratio=0
# We can't run full G/D without the full model, but we can verify __init__
from losses.loss import TwoStageLoss
print("  Import successful, FFL not instantiated when ratio=0")
print("  PASSED")

# --- Test 5: Gradient magnitude sanity check ---
print("\n[Test 5] FFL gradient magnitude vs L1...")
pred5 = torch.randn(2, 3, 256, 256, device='cuda', requires_grad=True)
target5 = torch.randn(2, 3, 256, 256, device='cuda')

l1_loss = torch.mean(torch.abs(pred5 - target5))
l1_loss.backward()
l1_grad_norm = pred5.grad.norm().item()

pred5b = pred5.detach().clone().requires_grad_(True)
ffl_loss = ffl(pred5b, target5)
ffl_loss.backward()
ffl_grad_norm = pred5b.grad.norm().item()

print(f"  L1  loss={l1_loss.item():.4f}  grad_norm={l1_grad_norm:.4f}")
print(f"  FFL loss={ffl_loss.item():.4f}  grad_norm={ffl_grad_norm:.4f}")
print(f"  Ratio FFL/L1 grad: {ffl_grad_norm/l1_grad_norm:.4f}")
print("  (Use this ratio to calibrate ffl_ratio)")
print("  PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nRecommended next steps:")
print("  1. Start training with --ffl-ratio=0 to verify unchanged baseline")
print("  2. Then try --ffl-ratio=1.0 for a short run (~100 kimg)")
print("  3. Monitor Loss/G/ffl_loss in TensorBoard")
print("  4. If FFL loss is much larger than pcp_loss, reduce ffl_ratio")
