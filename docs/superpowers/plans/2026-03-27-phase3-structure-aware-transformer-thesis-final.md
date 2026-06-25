# Phase 3 Structure-Aware Transformer Thesis Final Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the thesis-final Phase 3 architecture that improves facial geometry under large central masks by modifying the deep MAT transformer path with lightweight, zero-init structure-aware modules while remaining resume-safe from earlier checkpoints.

**Architecture:** Start from the `codex/phase2` codebase because it already contains the Phase 1 stabilization path and Phase 2 deep transformer adapters behind flags. Add a new structure branch that extracts lightweight geometry cues from visible pixels and mask boundaries, injects them directly into the `16x16` transformer reasoning path through zero-init attention bias and residual adapter hooks, and fuses the same structure signal into the stage-2 bottleneck. Keep `32x32` structure fusion as an optional Phase 3B extension, not part of the first thesis-critical run.

**Tech Stack:** Python, PyTorch, MAT `networks/mat.py`, new `networks/structure_guidance.py`, Click CLI in `train.py`, Colab entrypoint `collab/train_mat_real_(2).py`, `unittest`, resume-safe checkpoint loading in `training/training_loop.py`

---

## Preflight

- Use a dedicated worktree branched from `codex/phase2`, for example `codex/phase3-thesis-final`.
- Treat the best Phase 1 checkpoint and the best Phase 2 checkpoint as two candidate parents for Phase 3. Do not assume the Phase 2 checkpoint is automatically superior just because the code branch is newer.
- Thesis success criterion: Phase 3 must beat the best earlier branch on facial geometry under center-face masks, not merely produce smoother textures.

## File Map

- Create: `networks/structure_guidance.py`
- Modify: `networks/mat.py`
- Modify: `networks/__init__.py`
- Modify: `train.py`
- Modify: `collab/train_mat_real_(2).py`
- Modify: `training/training_loop.py`
- Create: `test_1/test_phase3_structure_guidance_contract.py`
- Modify: `test_1/test_train_architecture_flags.py`
- Create or Modify: `test_1/test_training_loop_lrt_grouping.py`

## Chunk 1: Lock The Phase 3 Interface Before Any Runtime Work

### Task 1: Add failing contract tests for the Phase 3 public surface

**Files:**
- Create: `test_1/test_phase3_structure_guidance_contract.py`
- Modify: `test_1/test_train_architecture_flags.py`
- Create or Modify: `test_1/test_training_loop_lrt_grouping.py`

- [ ] **Step 1: Write the failing contract test for the new structure module**

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Phase3StructureGuidanceContractTest(unittest.TestCase):
    def test_structure_guidance_module_declares_core_components(self):
        text = (ROOT / "networks/structure_guidance.py").read_text(encoding="utf-8")
        self.assertIn("class StructureInputBuilder", text)
        self.assertIn("class StructureEncoder", text)
        self.assertIn("class StructureAwareAttentionBias", text)
        self.assertIn("class StructureResidualAdapter", text)
```

- [ ] **Step 2: Extend the train flag contract test with the Phase 3 CLI**

```python
def test_train_exposes_phase3_structure_flags(self):
    train_text = self.read_text("train.py")
    self.assertIn("--enable-structure-guidance", train_text)
    self.assertIn("--enable-structure-fuse-16", train_text)
    self.assertIn("--enable-structure-fuse-stage2", train_text)
    self.assertIn("--enable-structure-fuse-32", train_text)
```

- [ ] **Step 3: Add a failing optimizer-grouping test for structure-aware transformer params**

```python
class TrainingLoopLrtGroupingTest(unittest.TestCase):
    def test_training_loop_mentions_structure_transformer_markers(self):
        text = (ROOT / "training/training_loop.py").read_text(encoding="utf-8")
        self.assertIn("tran_struct", text)
        self.assertIn("adapter", text)
        self.assertIn("bias", text)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract test_1.test_train_architecture_flags test_1.test_training_loop_lrt_grouping -v`

Expected: FAIL because the module, CLI flags, and grouping markers do not exist yet.

- [ ] **Step 5: Commit the failing-test scaffold**

```bash
git add test_1/test_phase3_structure_guidance_contract.py test_1/test_train_architecture_flags.py test_1/test_training_loop_lrt_grouping.py
git commit -m "test: add phase3 structure guidance contracts"
```

## Chunk 2: Build A Lightweight Structure Path That Stays Resume-Safe

### Task 2: Create the structure guidance module with zero-init outputs

**Files:**
- Create: `networks/structure_guidance.py`
- Modify: `networks/__init__.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Implement `StructureInputBuilder` for visible-image geometry cues**

```python
class StructureInputBuilder(nn.Module):
    def forward(self, image, mask):
        visible = image * mask
        gray = visible.mean(dim=1, keepdim=True)
        grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        boundary = torch.abs(mask - F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1))
        return visible, gray, boundary, grad_x, grad_y
```

- [ ] **Step 2: Implement `StructureEncoder` that emits `32x32` and `16x16` features**

```python
class StructureEncoder(nn.Module):
    def __init__(self, in_channels=7, base_channels=64, out_dim=180):
        ...
        self.to_32 = Conv2dLayer(in_channels=base_channels * 2, out_channels=out_dim, kernel_size=3)
        self.to_16 = Conv2dLayer(in_channels=base_channels * 4, out_channels=out_dim, kernel_size=3)
```

- [ ] **Step 3: Add zero-init projection helpers for transformer use**

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

- [ ] **Step 4: Export the new module from `networks/__init__.py`**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS for module-existence checks.

- [ ] **Step 5: Commit the structure module scaffold**

```bash
git add networks/structure_guidance.py networks/__init__.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add lightweight structure guidance module scaffold"
```

## Chunk 3: Modify The Deep Transformer Path, But Only Where It Can Win The Thesis

### Task 3: Thread structure features into the `16x16` transformer path first

**Files:**
- Modify: `networks/mat.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Add a failing transformer-hook contract**

```python
def test_mat_declares_structure_hooks_for_deep_transformer(self):
    text = (ROOT / "networks/mat.py").read_text(encoding="utf-8")
    self.assertIn("enable_structure_guidance", text)
    self.assertIn("enable_structure_fuse_16", text)
    self.assertIn("tran_struct_bias", text)
    self.assertIn("tran_struct_adapter", text)
```

- [ ] **Step 2: Extend `WindowAttention` to accept structure windows**

```python
def forward(self, x, mask_windows=None, structure_windows=None, mask=None):
    ...
    if self.tran_struct_bias is not None and structure_windows is not None:
        struct_logits = self.tran_struct_bias(structure_windows)
        struct_logits = struct_logits.permute(0, 2, 1)
        attn = attn + struct_logits.unsqueeze(-1) + struct_logits.unsqueeze(-2)
```

- [ ] **Step 3: Extend `SwinTransformerBlock` with a zero-init structure residual path**

```python
if self.tran_struct_adapter is not None and structure_tokens is not None:
    struct_delta = self.tran_struct_adapter(structure_tokens)
    x = x + self.struct_alpha.to(x.dtype) * struct_delta
```

- [ ] **Step 4: Restrict the first implementation to `16x16` only**

Implement the resolution gate exactly where Phase 2 already gates adapters:

```python
enable_struct_here = (
    enable_structure_guidance
    and input_resolution[0] == 16
    and enable_structure_fuse_16
)
```

- [ ] **Step 5: Keep the old path unchanged when flags are off**

Resume-safety rule:
- New modules must be optional.
- Old checkpoints must load with `require_all=False`.
- New parameters must start at zero effect.

- [ ] **Step 6: Run focused tests**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS.

- [ ] **Step 7: Commit the deep-transformer integration**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add structure-aware hooks to 16x16 transformer path"
```

### Task 4: Fuse the same structure signal into the stage-2 bottleneck

**Files:**
- Modify: `networks/mat.py`
- Test: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Add a failing contract for stage-2 structure fusion**

```python
def test_mat_declares_stage2_structure_fusion(self):
    text = (ROOT / "networks/mat.py").read_text(encoding="utf-8")
    self.assertIn("enable_structure_fuse_stage2", text)
    self.assertIn("stage2_struct_gate", text)
```

- [ ] **Step 2: Build the structure path in `SynthesisNet` from the same encoder output**

```python
struct_32, struct_16 = self.structure_encoder(images_in, masks_in)
...
if self.enable_structure_fuse_stage2:
    fea_16 = fea_16 + self.stage2_struct_gate.to(fea_16.dtype) * self.stage2_struct_proj(struct_16)
```

- [ ] **Step 3: Keep the stage-2 projection zero-init**

```python
nn.init.zeros_(self.stage2_struct_proj.weight)
nn.init.zeros_(self.stage2_struct_proj.bias)
self.stage2_struct_gate = nn.Parameter(torch.zeros([]))
```

- [ ] **Step 4: Run the focused contract tests**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS.

- [ ] **Step 5: Commit the stage-2 fusion path**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add stage2 structure bottleneck fusion"
```

## Chunk 4: Add The Training Surface Without Breaking Old Runs

### Task 5: Thread the Phase 3 flags through CLI and notebook entrypoint

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
```

- [ ] **Step 2: Thread them into `args.G_kwargs.synthesis_kwargs`**

```python
args.G_kwargs.synthesis_kwargs.enable_structure_guidance = enable_structure_guidance
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_16 = enable_structure_fuse_16
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_stage2 = enable_structure_fuse_stage2
args.G_kwargs.synthesis_kwargs.enable_structure_fuse_32 = enable_structure_fuse_32
```

- [ ] **Step 3: Add Click options and Colab variables**

```python
@click.option('--enable-structure-guidance', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-16', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-stage2', type=bool, metavar='BOOL')
@click.option('--enable-structure-fuse-32', type=bool, metavar='BOOL')
```

- [ ] **Step 4: Mirror the same defaults in `collab/train_mat_real_(2).py`**

```python
ENABLE_STRUCTURE_GUIDANCE = False
ENABLE_STRUCTURE_FUSE_16 = False
ENABLE_STRUCTURE_FUSE_STAGE2 = False
ENABLE_STRUCTURE_FUSE_32 = False
```

- [ ] **Step 5: Run the flag contract test**

Run: `python -m unittest test_1.test_train_architecture_flags -v`

Expected: PASS.

- [ ] **Step 6: Commit the training surface**

```bash
git add train.py collab/train_mat_real_(2).py test_1/test_train_architecture_flags.py
git commit -m "feat: add phase3 structure guidance flags"
```

### Task 6: Make the optimizer grouping explicitly structure-aware

**Files:**
- Modify: `training/training_loop.py`
- Modify: `test_1/test_training_loop_lrt_grouping.py`

- [ ] **Step 1: Extend the grouping markers to include structure-specific names**

```python
transformer_markers = ['tran', 'Tran', 'adapter', 'Adapter', 'struct', 'Struct', 'bias']
```

- [ ] **Step 2: Document the naming contract in code comments**

Add a short comment explaining that new structure-aware transformer modules should keep stable marker names like `tran_struct_bias`, `tran_struct_adapter`, or `stage2_struct_proj` so `--lrt` remains predictable.

- [ ] **Step 3: Run the grouping test**

Run: `python -m unittest test_1.test_training_loop_lrt_grouping -v`

Expected: PASS.

- [ ] **Step 4: Commit the optimizer update**

```bash
git add training/training_loop.py test_1/test_training_loop_lrt_grouping.py
git commit -m "feat: extend lrt grouping for phase3 structure modules"
```

## Chunk 5: Choose The Right Parent Checkpoint Before The Full Thesis Run

### Task 7: Run a dual-parent Phase 3 pilot instead of blindly trusting Phase 2

**Files:**
- Modify: `collab/train_mat_real_(2).py`
- Create or Update: run notes in the experiment tracker you use for checkpoints

- [ ] **Step 1: Freeze two candidate parent checkpoints**

Candidate A: best Phase 1 checkpoint  
Candidate B: best Phase 2 checkpoint

Do not proceed with only one parent unless one branch has already won clearly in side-by-side qualitative review.

- [ ] **Step 2: Define the first thesis-critical Phase 3 run**

```bash
python train.py \
  --data /path/to/FACEART/train \
  --data_val /path/to/FACEART/val \
  --cfg places512 \
  --resume /path/to/candidate_parent.pkl \
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
  --enable-structure-fuse-32 False
```

- [ ] **Step 3: Run two short pilots with identical config**

Run:
- Pilot A from the best Phase 1 checkpoint
- Pilot B from the best Phase 2 checkpoint

Expected:
- Both runs resume with only missing keys for new Phase 3 modules.
- No shape mismatch.
- No immediate OOM or divergence.

- [ ] **Step 4: Compare geometry-first cases before choosing the full run**

Use the same hard cases:
- center-face masks
- eyes hidden
- nose + mouth overlap
- strong asymmetry risk

Selection rule:
- If Phase 2 parent is not visibly better than Phase 1 parent on these cases, use the Phase 1 parent for the full Phase 3 run.

- [ ] **Step 5: Commit the experiment config changes if needed**

```bash
git add collab/train_mat_real_(2).py
git commit -m "chore: add phase3 thesis pilot configuration"
```

## Chunk 6: Keep One Controlled Extension, But Do Not Let It Block Thesis Closure

### Task 8: Add `32x32` structure fusion only as a controlled Phase 3B extension

**Files:**
- Modify: `networks/mat.py`
- Modify: `test_1/test_phase3_structure_guidance_contract.py`

- [ ] **Step 1: Gate `32x32` fusion behind a separate flag**

```python
enable_struct_here = (
    enable_structure_guidance
    and (
        (input_resolution[0] == 16 and enable_structure_fuse_16)
        or (input_resolution[0] == 32 and enable_structure_fuse_32)
    )
)
```

- [ ] **Step 2: Run the same contract tests again**

Run: `python -m unittest test_1.test_phase3_structure_guidance_contract -v`

Expected: PASS.

- [ ] **Step 3: Launch `32x32` only after the thesis-critical `16 + stage2` run is stable**

Run label suggestion:
- `faceart_phase3_struct16_stage2`
- `faceart_phase3_struct32_16_stage2`

- [ ] **Step 4: Commit the extension path**

```bash
git add networks/mat.py test_1/test_phase3_structure_guidance_contract.py
git commit -m "feat: add optional 32x32 structure fusion extension"
```

## Verification Checklist

- [ ] Confirm the Phase 3 code is implemented on a dedicated branch from `codex/phase2`.
- [ ] Confirm `StructureInputBuilder`, `StructureEncoder`, `StructureAwareAttentionBias`, and `StructureResidualAdapter` exist in `networks/structure_guidance.py`.
- [ ] Confirm `16x16` deep transformer structure hooks are optional and zero-init.
- [ ] Confirm stage-2 bottleneck fusion is optional and zero-init.
- [ ] Confirm new flags exist in `train.py` and `collab/train_mat_real_(2).py`.
- [ ] Confirm `training/training_loop.py` routes structure-aware transformer params into the `--lrt` group intentionally.
- [ ] Confirm resume from both Phase 1 and Phase 2 parents yields only missing keys for new Phase 3 modules.
- [ ] Confirm the first full Phase 3 run is chosen from evidence, not from branch recency.

## Definition Of Done

### Phase 3 core is done when
- The model resumes cleanly from the selected parent checkpoint.
- `16x16` transformer structure hooks and stage-2 structure fusion train stably for multiple snapshots.
- On the hard center-face cases, the chosen Phase 3 run looks more geometrically correct than both the best Phase 1 and best Phase 2 outputs.
- Improvements are visible in eyes, nose, mouth alignment, and jawline continuity, not only in texture smoothness.

### Phase 3B is done when
- `32x32` structure fusion improves or at least preserves the Phase 3 core geometry gains.
- The extension does not destabilize training or erase the clarity advantage seen in Phase 1 hard cases.

## Final Recommendation

For the thesis-final branch, do **not** choose between "too safe" and "too heavy." The most reasonable path is to branch from the Phase 2 codebase, but let the data choose whether the best Phase 1 or Phase 2 checkpoint should be the training parent. Then implement a deeper yet resume-safe Phase 3 by modifying the deep transformer path itself with zero-init structure-aware attention bias and residual structure adapters at `16x16`, plus a stage-2 bottleneck fuse. This is the smallest change that is still strong enough to plausibly beat Phase 1 on the geometry failures that matter for the thesis.