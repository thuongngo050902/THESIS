# PARF Redesign — Steps 2 (Input) & 3 (Output) Design

**Date:** 2026-06-09
**Branch:** `feat/parf-3step-redesign`
**Status:** Approved, ready for implementation plan

## Context

The PARF (Portrait Artwork Reconstruction Framework) demo is being reorganized from
the legacy 4-tab Streamlit app into a 3-step flow per
`design_handoff_parf_redesign/README.md` (the low-fidelity HTML wireframe handoff).

**Step 1 — Create Input** is complete and working (`parf_render_create_input` and the
`parf_*` helpers in `streamlit_app.py`): upload → mark config (shape / position / scale /
nudge) → live binary-mask + damaged-portrait preview → Confirm locks the package and
navigates to Step 2.

This spec covers **Step 2 (Input — run inference)** and **Step 3 (Output — result +
compare + zoom lightbox)**, which are currently stubs:

- `parf_render_input_step` (`streamlit_app.py:1417`) — shows the confirmed package but the
  **Run inference** button is `disabled=True`.
- `parf_render_output_step` (`streamlit_app.py:1447`) — only an info message.

## Goals

- Wire **Run inference** end-to-end on Step 2 and navigate to Output on success.
- Build the Step 3 Output view: persisted result image, Zoom, Advanced compare disclosure,
  compare section (chips + reorder/remove grid), and a synced zoom lightbox.
- Reuse the existing inference + caching machinery; do not change Step 1 or the inference
  adapter contract.

## Non-Goals (out of scope)

- `localStorage` cross-refresh persistence (YAGNI — `st.session_state` covers in-session
  reruns; full refresh-safety is explicitly optional in the handoff).
- Any change to Step 1, the `demo_ui/inference_adapter.py` contract, or
  `colab_inference_api.py` endpoints.
- New backend checkpoints or model changes.

## Decisions (from brainstorming)

1. **Zoom lightbox: Hybrid.** Native Streamlit owns compare/order/remove state
   (server-authoritative, fully synced). The lightbox is an embedded `components.html`
   JS stage for the genuinely client-side interactions (scroll-zoom, drag-pan, synced
   transform, row/grid toggle).
2. **Compare references: wire all, load lazily.** Run inference produces the Output
   (`final`). MAT (`mat_baseline`) and Coarse (`stage1`) are fetched lazily the first time
   the compare view needs them, reusing the existing per-checkpoint caches. Origin and
   Damaged come from the confirmed package (no inference).
3. **Backend defaults to remote** (Colab API). Local checkpoint paths point at a server,
   not the dev Mac, so remote is the working default; local stays available via Advanced.

## Reused machinery

| Compare item | Source (already implemented) |
|---|---|
| **Output** (always present) | `ensure_final_output_result` / remote `checkpoint=final` |
| **Coarse Restoration** | `ensure_stage1_result` / remote `checkpoint=stage1` |
| **MAT** | `ensure_mat_original_result` / remote `checkpoint=mat_baseline` |
| **Origin** | `st.session_state["confirmed_input_image"]` |
| **Damaged Portrait** | `st.session_state["confirmed_masked_input_image"]` |

The remote API (`colab_inference_api.py`) supports `final`, `stage1`, and `mat_baseline`,
each returning one image. The `final_output_result is not None` gate that the step bar
already uses to unlock Output is preserved.

## Design

### Step 2 — Input (run inference)

- Keep the current two-column layout (action column / package preview column).
- Replace the disabled button with an active **Run inference**:
  - On click: show `Running reconstruction…` via `st.status`/spinner.
  - Resolve the confirmed image + mask, call the existing run path with `checkpoint=final`
    (remote by default), store the `StageArtifact` in `final_output_result`.
  - **Success:** show `Reconstruction complete — opening Output.`, set
    `parf_step = "output"`, `st.rerun()` (step bar unlocks Output).
  - **Error:** surface the error message, stay on Input, do not navigate.
- Keep **← Back to Create Input**.
- Add a **collapsed Advanced expander** exposing backend mode, remote endpoint, and
  checkpoint paths (defaults: remote + env endpoint) so existing configurability survives
  without cluttering the clean flow.
- Re-confirmation safety is already handled: editing the mark in Step 1 invalidates the
  confirmation (`parf_compute_live_mask` → `invalidate_confirmation_and_results`), which
  clears `final_output_result` and re-locks Output.

### Step 3 — Output (result + compare)

Single centered column (max width ~760px via CSS in `inject_parf_theme`).

- **Result block:** large Output image with a **Zoom** button anchored top-right, plus the
  note about session persistence and zoom behavior.
- **Advanced — compare** disclosure (`st.expander`): helper text + 4 checkboxes
  (Damaged Portrait, MAT, Coarse Restoration, Origin; all checked by default) bound to
  `parf_cmp`, plus a **Compare →** button that sets `parf_compare_open = True`.
- **Compare section** (rendered when `parf_compare_open`):
  - Heading + **Zoom all (synced)** button + helper text.
  - **Item list** computed by a pure helper `parf_compare_items()`:
    *Output (always) + checked references − removed, in `parf_order`*.
  - **Chips row:** one chip per item; every chip except Output has a × button that adds the
    key to `parf_removed`.
  - **Comparison grid:** responsive `st.columns` of cards; each card = image + title +
    sub-label + ◀ ▶ reorder buttons (◀ disabled on the first item, ▶ on the last).
    Reorder swaps an item with its neighbor in `parf_order`.
  - MAT/Coarse images are fetched lazily via the `ensure_*_result` helpers when their tile
    renders; results are cached so repeat renders are free.

### Zoom lightbox (hybrid)

- Native Streamlit owns order/remove/compare state. The lightbox is an embedded
  `components.html` JS stage seeded with the selected images (data URLs) + current order +
  layout.
- **Interactions inside the component:** scroll = synced zoom (clamp ~0.4×–6×), drag =
  synced pan (suppressed when pointer-down lands on a button), **Row / Grid** toggle (shown
  when 3+ images), per-image ◀ ▶ that reorder *within the open lightbox* (client-side only;
  authoritative reorder remains the grid).
- **Opened two ways:** single image (Output **Zoom** button) and all images
  (**Zoom all**). A `parf_lb_open` flag + `parf_lb_scope` decide what renders; a
  server-side **Close** button dismisses it.
- **Honest constraint:** Streamlit's iframe sandbox means the lightbox renders as a large
  dark full-content-width *stage*, not an OS-level full-screen overlay. All described
  interactions work within it. Esc/backdrop-close work inside the component for its own
  area; the Streamlit **Close** button is the authoritative dismiss.

## State (added to `init_session_state`)

| Key | Type | Default | Purpose |
|---|---|---|---|
| `parf_cmp` | dict{masked,mat,coarse,origin: bool} | all `True` | Advanced compare checkboxes |
| `parf_compare_open` | bool | `False` | Whether the compare section is shown |
| `parf_order` | list[str] | derived | Comparison order, shared by grid + lightbox |
| `parf_removed` | set[str] | empty | Items removed from the comparison via chip × |
| `parf_lb_open` | bool | `False` | Whether the lightbox stage is shown |
| `parf_lb_layout` | {row, grid} | `row` | Lightbox layout |
| `parf_lb_scope` | {single, all} | `single` | What the lightbox renders |

Backend keys (`backend_mode`, `remote_endpoint`, checkpoint paths) already exist in state.
`localStorage` persistence is intentionally omitted.

## Testing

**Pytest unit tests** (next to `tests/test_shape_masks.py`) for the pure logic:

- `parf_compare_items()` — Output always present and first; checked references included;
  removed items excluded; order follows `parf_order`.
- Reorder swap helper — swaps neighbors; clamps at the ends (no-op on first ◀ / last ▶).
- Checkbox → order derivation — toggling a checkbox adds/removes the key consistently.
- Item-key → source mapping — each key resolves to the correct `ensure_*` / `confirmed_*`
  source.

**Manual verification** (lightbox JS + rendering): run Streamlit and click the full flow —
Step 1 → Confirm → Run inference → Output → compare → reorder/remove → Zoom / Zoom all →
Close. Verify scroll-zoom, drag-pan, synced transform, and Row/Grid in the lightbox.

## Risks / open points

- Remote inference latency/availability affects Run inference and lazy compare loads;
  errors are surfaced, not swallowed.
- The embedded-iframe lightbox cannot cover the full OS viewport; mitigated by a large
  dark stage and documented as a known constraint.
