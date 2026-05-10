# Phase 3 Thesis Final Updated From Qualitative Review Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize the thesis-winning Phase 3 architecture by building on Phase 2 robustness while restoring the facial landmark clarity that sometimes made Phase 1 look visually stronger.

**Architecture:** Use the Phase 2 codebase and default to the best Phase 2 checkpoint as the Phase 3 parent, because the newest qualitative review shows Phase 2 is more reliable on the hardest masks and broad face reconstruction. Update Phase 3 so it no longer means only static structure fusion; instead it becomes an adaptive structure-aware transformer refinement stage with three parts: lightweight structure encoding, deep-transformer structure injection at `16x16`, and mask-severity-conditioned gating that injects more structural guidance on hard masks while staying lighter on easier masks so the model preserves the crisper Phase 1 feeling.

**Tech Stack:** Python, PyTorch, MAT `networks/mat.py`, new `networks/structure_guidance.py`, Click CLI in `train.py`, Colab entrypoint `collab/train_mat_real_(2).py`, `unittest`, resume-safe checkpoint loading in `training/training_loop.py`

---

## Updated Thesis Read From The New Qualitative Review

### What the new images imply
- The new irregular-mask female example strongly favors `Phase2 ckpt68` over `Phase1 Arch`. Phase 2 is more coherent, more symmetric, and more globally face-plausible.
- The new dark-background portrait still shows that Phase 2 can look slightly softer than Phase 1, but it is not actually less structurally plausible. The main difference is that Phase 1 sometimes appears sharper, not necessarily more correct.
- The male central-mask example shows the same pattern: Phase 2 is competitive or better in whole-face plausibility, but Phase 1 can feel more visually "clear" because some anchor details remain higher-contrast.

### Updated conclusion
- `Phase 2` should now be treated as the stronger base for Phase 3.
- The remaining weakness is not "Phase 2 is wrong"; it is that Phase 2 can become slightly over-smooth or under-emphasize local anchor clarity.
- Therefore the thesis-final Phase 3 should not abandon Phase 2. It should build on Phase 2 and add **adaptive structure-aware transformer guidance** that improves geometry on hard masks without washing out facial anchors on easier masks.

## Updated Phase 3 Thesis Claim

The Phase 3 contribution is no longer just "lightweight structure guidance." It is now:

**Adaptive Structure-Aware Transformer Refinement for Resume-Safe Face Artwork Inpainting**

This updated name matches the evidence better because Phase 3 now has to do two things at once:
- keep the robustness and global plausibility already visible in Phase 2,
- recover the facial anchor clarity that made some Phase 1 outputs subjectively look stronger.

## Parent And Branch Strategy

### Default parent
- Use the best `Phase 2` checkpoint as the Phase 3 training parent.

### Regression guard
- Keep the best `Phase 1` checkpoint as a mandatory evaluation reference.
- Only fall back to a Phase 1 parent if the first Phase 3 pilot from the Phase 2 parent is clearly worse on both clarity and geometry across the fixed hard-case set.

### Why this changed
The new images reduce the earlier uncertainty. Phase 2 now looks clearly more valuable on the masks that matter most for a thesis-level face inpainting claim: broad corruption, irregular corruption, and full-center corruption.

## Design Principles

- Do not replace the MAT backbone.
- Do not change `w_dim`, `z_dim`, `num_ws`, transformer stage counts, or decoder depth.
- Keep all new modules optional and zero-init.
- Add stronger transformer modification than Phase 2, but only through residual, bias, and gated side branches.
- Let the structure signal be **adaptive**, not always-on at full strength.

## File Map

- Create: `networks/structure_guidance.py`
- Modify: `networks/mat.py`
- Modify: `networks/__init__.py`
- Modify: `train.py`
- Modify: `collab/train_mat_real_(2).py`
- Modify: `training/training_loop.py`
- Create: `test_1/test_phase3_structure_guidance_contract.py`
- Modify: `test_1/test_train_architecture_flags.py`
- Modify or Create: `test_1/test_training_loop_lrt_grouping.py`

## Chunk 1: Lock The New Adaptive Phase 3 Surface In Tests

### Task 1: Add failing tests for adaptive structure-aware Phase 3 contracts

**Files:**
- Create: `test_1/test_phase3_structure_guidance_contract.py`
- Modify: `test_1/test_train_architecture_flags.py`
- Modify or Create: `test_1/test_training_loop_lrt_grouping.py`

- [ ] **Step 1: Add a failing contract for the new structure module names**

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Phase3StructureGuidanceContractTest(unittest.TestCase):
    def test_structure_module_declares_adaptive_components(self):
        text = (ROOT / "networks/structure_guidance.py").read_text(encoding="utf-8")
        self.assertIn("class StructureInputBuilder", text)
        self.assertIn("class StructureEncoder", text)
        self.assertIn("class StructureAwareAttentionBias", text)
        self.assertIn("class StructureResidualAdapter", text)
        self.assertIn("class MaskSeverityGate", text)
```

- [ ] **Step 2: Extend the train flag contract for adaptive gating controls**

```python
def test_train_exposes_updated_phase3_flags(self):
    train_text = self.read_text("train.py")
    self.assertIn("--enable-structure-guidance", train_text)
    self.assertIn("--enable-structure-fuse-16", train_text)
    self.assertIn("--enable-structure-fuse-stage2", train_text)
    self.assertIn("--enable-structure-fuse-32", train_text)
    self.assertIn("--enable-adaptive-structure-gate", train_text)
```

- [ ] **Step 3: Add a failing grouping test for structure-aware transformer params**

```python
def test_training_loop_mentions_structure_specific_markers(self):
    text = (ROOT / "training/training_loop.py").read_text(encoding="utf-8")
    self.assertIn("tran_struct", text)
    self.assertIn("struct", text)
    self.assertIn("bias", text)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract test_1.test_train_architecture_flags test_1.test_training_loop_lrt_grouping -v`

Expected: FAIL because the adaptive Phase 3 components and flags do not exist yet.

- [ ] **Step 5: Commit the failing contracts**

```bash
git add test_1/test_phase3_structure_guidance_contract.py test_1/test_train_architecture_flags.py test_1/test_training_loop_lrt_grouping.py
git commit -m "test: add updated phase3 adaptive structure contracts"
```

## Chunk 2: Build The Lightweight Structure Stack And Adaptive Gate

### Task 2: Create `networks/structure_guidance.py`

**Files:**
- Create: `networks/structure_guidance.py`
- Modify: `networks/__init__.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Implement `StructureInputBuilder` from visible RGB, mask, and boundary**

```python
class StructureInputBuilder(nn.Module):
    def forward(self, image, mask):
        visible = image * mask
        gray = visible.mean(dim=1, keepdim=True)
        grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        boundary = torch.abs(mask - F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1))
        return torch.cat([visible, gray, grad_x, grad_y, boundary], dim=1), boundary
```

- [ ] **Step 2: Implement `StructureEncoder` with `32x32` and `16x16` outputs**

```python
class StructureEncoder(nn.Module):
    def __init__(self, in_channels=6, base_channels=64, out_dim=180):
        ...
        self.to_32 = Conv2dLayer(in_channels=base_channels * 2, out_channels=out_dim, kernel_size=3)
        self.to_16 = Conv2dLayer(in_channels=base_channels * 4, out_channels=out_dim, kernel_size=3)
```

- [ ] **Step 3: Implement zero-init transformer-side helpers**

```python
class StructureAwareAttentionBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.tran_struct_bias_proj = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.tran_struct_bias_proj.weight)
        nn.init.zeros_(self.tran_struct_bias_proj.bias)
```

```python
class StructureResidualAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.tran_struct_down = nn.Linear(dim, hidden_dim)
        self.tran_struct_up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.tran_struct_up.weight)
        nn.init.zeros_(self.tran_struct_up.bias)
```

- [ ] **Step 4: Implement `MaskSeverityGate` so Phase 3 can be strong on hard masks and light on easy masks**

```python
class MaskSeverityGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, pooled_struct, pooled_boundary, pooled_mask):
        score = torch.cat([pooled_struct, pooled_boundary, pooled_mask], dim=1)
        return torch.sigmoid(self.fc(score))
```

- [ ] **Step 5: Export the new classes from `networks/__init__.py`**

- [ ] **Step 6: Run the structure contract test**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS for the module-level existence checks.

- [ ] **Step 7: Commit the structure stack**

```bash
git add networks/structure_guidance.py networks/__init__.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add adaptive structure guidance stack"
```

## Chunk 3: Modify The Deep Transformer Path In A Stronger But Still Safe Way

### Task 3: Add adaptive structure-aware injection to the `16x16` transformer path

**Files:**
- Modify: `networks/mat.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Add a failing contract for adaptive transformer hooks**

```python
def test_mat_declares_adaptive_structure_transformer_hooks(self):
    text = (ROOT / "networks/mat.py").read_text(encoding="utf-8")
    self.assertIn("enable_structure_guidance", text)
    self.assertIn("enable_adaptive_structure_gate", text)
    self.assertIn("tran_struct_bias", text)
    self.assertIn("tran_struct_adapter", text)
    self.assertIn("struct_alpha", text)
```

- [ ] **Step 2: Extend `WindowAttention` to accept structure windows and adaptive gate values**

```python
def forward(self, x, mask_windows=None, structure_windows=None, structure_gate=None, mask=None):
    ...
    if self.tran_struct_bias is not None and structure_windows is not None:
        struct_logits = self.tran_struct_bias(structure_windows)
        if structure_gate is not None:
            struct_logits = struct_logits * structure_gate.unsqueeze(-1)
        attn = attn + struct_logits.permute(0, 2, 1).unsqueeze(-1)
```

- [ ] **Step 3: Add an adaptive structure residual path in `SwinTransformerBlock`**

```python
if self.tran_struct_adapter is not None and structure_tokens is not None:
    struct_delta = self.tran_struct_adapter(structure_tokens)
    if structure_gate is not None:
        struct_delta = struct_delta * structure_gate.unsqueeze(-1)
    x = x + self.struct_alpha.to(x.dtype) * struct_delta
```

- [ ] **Step 4: Restrict the thesis-critical first implementation to `16x16`**

```python
enable_struct_here = (
    enable_structure_guidance
    and input_resolution[0] == 16
    and enable_structure_fuse_16
)
```

- [ ] **Step 5: Keep all structure hooks optional and zero-init**

Rules:
- old checkpoints must still load with `require_all=False`
- new parameters must produce zero effect at initialization
- when `enable_structure_guidance=False`, the forward path must behave exactly like current Phase 2

- [ ] **Step 6: Run focused tests**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS.

- [ ] **Step 7: Commit the deep-transformer update**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add adaptive structure hooks to 16x16 transformer path"
```

### Task 4: Fuse the same adaptive structure signal into the stage-2 bottleneck

**Files:**
- Modify: `networks/mat.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Add a failing contract for adaptive stage-2 structure fusion**

```python
def test_mat_declares_adaptive_stage2_structure_fusion(self):
    text = (ROOT / "networks/mat.py").read_text(encoding="utf-8")
    self.assertIn("enable_structure_fuse_stage2", text)
    self.assertIn("stage2_struct_proj", text)
    self.assertIn("stage2_struct_gate", text)
```

- [ ] **Step 2: Inject the structure feature into `fea_16` before style generation**

```python
struct_32, struct_16, struct_gate = self.structure_encoder(...)
...
if self.enable_structure_fuse_stage2:
    stage2_delta = self.stage2_struct_proj(struct_16)
    if self.enable_adaptive_structure_gate:
        stage2_delta = stage2_delta * struct_gate.view(-1, 1, 1, 1)
    fea_16 = fea_16 + self.stage2_struct_gate.to(fea_16.dtype) * stage2_delta
```

- [ ] **Step 3: Keep the stage-2 projection and gate zero-init**

```python
nn.init.zeros_(self.stage2_struct_proj.weight)
nn.init.zeros_(self.stage2_struct_proj.bias)
self.stage2_struct_gate = nn.Parameter(torch.zeros([]))
```

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS.

- [ ] **Step 5: Commit the stage-2 fusion update**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add adaptive stage2 structure bottleneck fusion"
```

## Chunk 4: Expose The Updated Thesis Controls In The Training Surface

### Task 5: Add the updated Phase 3 flags to CLI and Colab

**Files:**
- Modify: `train.py`
- Modify: `collab/train_mat_real_(2).py`
- Modify: `test_1/test_train_architecture_flags.py`

- [ ] **Step 1: Add the new kwargs to `setup_training_loop_kwargs`**

```python
enable_structure_guidance = False,
enable_structure_fuse_16 = False,
enable_structure_fuse_stage2 = False,
enable_structure_fuse_32 = False,
enable_adaptive_structure_gate = False,
```

- [ ] **Step 2: Thread them into `args.G_kwargs.synthesis_kwargs`**

```python
args.G_kwargs.synthesis_kwargs.enable_structure_guidance = enable_structure_guidance
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_16 = enable_structure_fuse_16
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_stage2 = enable_structure_fuse_stage2
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_32 = enable_structure_fuse_32
args.G_kwargs.synthesis_kwargs.enable_adaptive_structure_gate = enable_adaptive_structure_gate
```

- [ ] **Step 3: Add the Click options**

```python
@click.option('--enable-structure-guidance', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-16', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-stage2', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-32', type=bool, metavar='BOOL')
@click.option('--enable-adaptive-structure-gate', type=bool, metavar='BOOL')
```

- [ ] **Step 4: Mirror the same controls in `collab/train_mat_real_(2).py`**

```python
ENABLE_STRUCTURE_GUIDANCE = False
ENABLE_STRUCTURE_FUSE_16 = False
ENABLE_STRUCTURE_FUSE_STAGE2 = False
ENABLE_STRUCTURE_FUSE_32 = False
ENABLE_ADAPTIVE_STRUCTURE_GATE = False
```

- [ ] **Step 5: Run the flag contract tests**

Run: `python -m unittest test_1.test_train_architecture_flags -v`

Expected: PASS.

- [ ] **Step 6: Commit the training surface**

```bash
git add train.py collab/train_mat_real_(2).py test_1/test_train_architecture_flags.py
git commit -m "feat: add updated phase3 adaptive structure flags"
```

### Task 6: Update optimizer grouping so new structure-aware transformer params get `--lrt`

**Files:**
- Modify: `training/training_loop.py`
- Modify: `test_1/test_training_loop_lrt_grouping.py`

- [ ] **Step 1: Extend the grouping markers**

```python
transformer_markers = ['tran', 'Tran', 'adapter', 'Adapter', 'struct', 'Struct', 'bias', 'gate', 'Gate']
```

- [ ] **Step 2: Add a short code comment explaining the naming contract**

Document that structure-aware transformer modules must keep stable marker names such as `tran_struct_bias`, `tran_struct_adapter`, `stage2_struct_proj`, or `struct_gate` so the high-LR group remains explicit and reviewable.

- [ ] **Step 3: Run the grouping test**

Run: `python -m unittest test_1.test_training_loop_lrt_grouping -v`

Expected: PASS.

- [ ] **Step 4: Commit the grouping update**

```bash
git add training/training_loop.py test_1/test_training_loop_lrt_grouping.py
git commit -m "feat: extend lrt grouping for adaptive phase3 structure modules"
```

## Chunk 5: Change The Evaluation Protocol So The Thesis Matches What The Images Actually Show

### Task 7: Split qualitative evaluation into clarity and corruption regimes

**Files:**
- Create or Update: your evaluation notes or comparison script inputs
- Modify if needed: `collab/eval_phase1_helpers/make_comparisons.py`

- [ ] **Step 1: Define three required evaluation buckets**

Bucket A: small-to-medium masks where face anchor clarity matters  
Bucket B: irregular large masks  
Bucket C: central vertical masks over eyes, nose, and mouth

- [ ] **Step 2: Build a fixed validation subset with at least 5 examples per bucket**

The goal is to stop mixing "looks sharper" with "is actually more correct" and judge the Phase 3 thesis claim on consistent mask regimes.

- [ ] **Step 3: Make the side-by-side sheet always include**

- `Finetune`
- `Phase1 Arch`
- `Phase2 ckpt68`
- `Phase3 pilot`

- [ ] **Step 4: Add written review criteria per image**

Score each image on:
- global face plausibility
- eye placement
- nose-mouth alignment
- contour continuity
- local clarity

- [ ] **Step 5: Commit any evaluation-script or notes update**

```bash
git add collab/eval_phase1_helpers/make_comparisons.py
git commit -m "chore: update qualitative evaluation buckets for phase3 thesis review"
```

## Chunk 6: Use Phase 2 As Parent, But Make The First Phase 3 Pilot Deliberately Conservative

### Task 8: Run the thesis-critical Phase 3 pilot from the best Phase 2 checkpoint

**Files:**
- Modify: `collab/train_mat_real_(2).py`
- Create or Update: experiment notes

- [ ] **Step 1: Freeze the best Phase 2 parent checkpoint**

Selection rule:
- prioritize robustness on irregular and central masks
- use Phase 1 only as regression guard, not as default parent

- [ ] **Step 2: Define the first Phase 3 pilot config**

```bash
python train.py \
  --data /path/to/FACEART/train \
  --data_val /path/to/FACEART/val \
  --cfg places512 \
  --resume /path/to/best_phase2_parent.pkl \
  --epochs 10 \
  --batch 4 \
  --workers 2 \
  --snap 2 \
  --metrics none \
  --pr 0.1 \
  --pl False \
  --truncation 0.5 \
  --style_mix 0.5 \
  --ema 10 \
  --lr 5e-5 \
  --lrt 1e-4 \
  --lambda-ffl 0.02 \
  --aug noaug \
  --enable-rel-pos-bias True \
  --enable-mask-bias True \
  --enable-deterministic-latent-gate True \
  --enable-tran-adapter-32 True \
  --enable-tran-adapter-16 True \
  --enable-structure-guidance True \
  --enable-structure-fuse-16 True \
  --enable-structure-fuse-stage2 True \
  --enable-structure-fuse-32 False \
  --enable-adaptive-structure-gate True
```

- [ ] **Step 3: Require a resume check on the first pilot**

Expected:
- only missing keys from Phase 3 modules
- no shape mismatch
- no immediate divergence or OOM

- [ ] **Step 4: Compare the first pilot directly against Phase 1 and Phase 2 on the fixed buckets**

Success rule:
- Bucket B and Bucket C must stay at least as good as Phase 2
- Bucket A must recover some of the local anchor clarity that made Phase 1 look sharper

- [ ] **Step 5: Commit any pilot-config change**

```bash
git add collab/train_mat_real_(2).py
git commit -m "chore: add updated phase3 thesis pilot configuration"
```

## Chunk 7: Keep One Controlled Extension, But Only After The Core Thesis Claim Is Stable

### Task 9: Add `32x32` structure fusion only as a Phase 3B extension

**Files:**
- Modify: `networks/mat.py`
- Modify: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Gate `32x32` structure fusion separately**

```python
enable_struct_here = (
    enable_structure_guidance
    and (
        (input_resolution[0] == 16 and enable_structure_fuse_16)
        or (input_resolution[0] == 32 and enable_structure_fuse_32)
    )
)
```

- [ ] **Step 2: Keep `32x32` off for the first thesis-critical run**

This avoids destabilizing a branch that is already trying to prove two things at once: strong geometry and restored clarity.

- [ ] **Step 3: Launch the extension only after the `16 + stage2 + adaptive gate` branch is stable**

Suggested run names:
- `faceart_phase3_adaptive_struct16_stage2`
- `faceart_phase3_adaptive_struct32_16_stage2`

- [ ] **Step 4: Commit the extension path**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add optional 32x32 adaptive structure fusion"
```

## Verification Checklist

- [ ] Confirm the Phase 3 branch starts from the Phase 2 codebase.
- [ ] Confirm the best Phase 2 checkpoint is frozen as the default parent.
- [ ] Confirm the best Phase 1 checkpoint is retained as a regression reference.
- [ ] Confirm `StructureInputBuilder`, `StructureEncoder`, `StructureAwareAttentionBias`, `StructureResidualAdapter`, and `MaskSeverityGate` exist.
- [ ] Confirm `16x16` structure-aware transformer hooks are optional and zero-init.
- [ ] Confirm the adaptive structure gate changes injection strength based on mask severity.
- [ ] Confirm stage-2 bottleneck fusion is optional and zero-init.
- [ ] Confirm new flags exist in `train.py` and `collab/train_mat_real_(2).py`.
- [ ] Confirm `training/training_loop.py` routes structure-aware transformer params into the `--lrt` group intentionally.
- [ ] Confirm the first Phase 3 pilot preserves or improves Phase 2 on hard masks while improving clarity relative to Phase 2 on at least part of Bucket A.

## Definition Of Done

### Phase 3 core is done when
- The model resumes cleanly from the selected Phase 2 parent checkpoint.
- The first pilot is stable for multiple snapshots.
- On Bucket B and Bucket C, Phase 3 is at least as good as Phase 2 in geometry and plausibility.
- On Bucket A, Phase 3 visibly reduces the "softening" problem and recovers some of the local facial anchor clarity that made Phase 1 look stronger.
- Across the full fixed set, Phase 3 becomes the most balanced branch overall, not necessarily the sharpest on every image.

### Phase 3B is done when
- `32x32` fusion gives further gains without erasing the clarity gains recovered by the adaptive gate.
- The extension keeps training stable and remains resume-safe.

## Final Recommendation

The updated qualitative evidence changes the plan in one important way: `Phase 2` should now be treated as the right base, not as a failed detour. The correct thesis-final move is to **keep Phase 2 robustness and repair its softness**, not to backtrack to Phase 1. Therefore the most reasonable Phase 3 is an **adaptive structure-aware transformer refinement** branch that modifies the deep transformer path more strongly than before, but only through zero-init side modules and mask-severity-conditioned gates so the model stays resume-safe and gains clarity without giving up the broader improvements already visible in Phase 2.