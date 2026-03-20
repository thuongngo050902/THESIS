# FACEART MAT Colab Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MAT repo and Colab notebook reliably fine-tune the original MAT checkpoint on FACEART with an added configurable FFL term.

**Architecture:** Keep MAT training logic inside the existing repo and use the Colab notebook as an orchestration layer. Harden the repo for Colab imports and modern PyTorch, expose `lambda_ffl` cleanly in the CLI, and rewrite the notebook around a single reproducible fine-tuning path.

**Tech Stack:** Python, Click CLI, PyTorch-style training loop, Google Colab, Google Drive, unittest

---

## Chunk 1: Contract Tests And Compatibility Fixes

### Task 1: Add failing contract tests for the Colab-safe interface

**Files:**
- Create: `test_1/test_colab_entrypoint_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_exposes_lambda_ffl_cli_flag():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_1.test_colab_entrypoint_contract -v`
Expected: FAIL because the current repo still lacks some of the required source-level compatibility changes.

- [ ] **Step 3: Write minimal implementation**

Patch the repo files needed by the tests without redesigning MAT.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_1.test_colab_entrypoint_contract -v`
Expected: PASS

## Chunk 2: Repo Runtime Hardening

### Task 2: Make the repo Colab-safe without changing MAT architecture

**Files:**
- Create: `datasets/__init__.py`
- Create: `losses/__init__.py`
- Create: `networks/__init__.py`
- Modify: `torch_utils/misc.py`
- Modify: `generate_image.py`
- Modify: `train.py`

- [ ] **Step 1: Fix import/package issues**
- [ ] **Step 2: Add `--lambda-ffl` while preserving `--ffl-ratio`**
- [ ] **Step 3: Remove notebook-only need for the `InfiniteSampler` patch**
- [ ] **Step 4: Keep baseline behavior unchanged when `lambda_ffl=0`**
- [ ] **Step 5: Run the contract tests**

## Chunk 3: Colab Notebook Rewrite

### Task 3: Convert the uploaded notebook script into a clean fine-tuning entrypoint

**Files:**
- Modify: `collab/train_mat_real_(2).py`

- [ ] **Step 1: Remove token-based git flow and dataset scraping sections**
- [ ] **Step 2: Add explicit FACEART fine-tuning configuration cells**
- [ ] **Step 3: Add dependency install, `PYTHONPATH`, and path validation cells**
- [ ] **Step 4: Add the final `train.py` command using the original MAT pretrained checkpoint and `LAMBDA_FFL`**
- [ ] **Step 5: Add optional post-train inference/comparison cells**
- [ ] **Step 6: Run the contract tests again**

## Chunk 4: Final Verification

### Task 4: Verify the minimal local guarantees

**Files:**
- Test: `test_1/test_colab_entrypoint_contract.py`
- Test: `test_1/test_schedule_utils.py`

- [ ] **Step 1: Run `python -m unittest test_1.test_colab_entrypoint_contract -v`**
- [ ] **Step 2: Run `python -m unittest test_1.test_schedule_utils -v`**
- [ ] **Step 3: Summarize what was verified locally and what still requires a real Colab GPU run**
