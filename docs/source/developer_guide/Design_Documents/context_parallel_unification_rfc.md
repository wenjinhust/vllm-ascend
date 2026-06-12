# RFC: Context Parallel (PCP/DCP) Unification

**Status:** Draft  
**Authors:** vLLM Ascend Team  
**Created:** 2026-06-12  
**Related:** [Context Parallel Design](./context_parallel.md), [vLLM RFC #25749](https://github.com/vllm-project/vllm/issues/25749)

---

## TL;DR

vLLM Ascend's PCP (Prefill Context Parallel) and DCP (Decode Context Parallel) share a unified KV-sharding model (`cp_size = pcp_size × dcp_size`), but the **implementation is fragmented**: token partitioning, metadata preparation, and attention-backend integration are duplicated across model-type branches and per-backend CP modules. This RFC proposes a **strategy-based `ContextParallelManager`** with capability-driven model registration, unified metadata contracts, and phased migration—so new models can be adapted by declaring a CP profile instead of editing scattered `if pcp_use_hybrid_attn` branches.

---

## 1. Motivation

### 1.1 Problem Statement

Context Parallel is critical for long-sequence inference on Ascend NPUs. The existing [context parallel design doc](./context_parallel.md) describes PCP and DCP conceptually as two facets of a single CP abstraction. In practice, the codebase diverges:

| Layer | Intended abstraction | Actual state |
|-------|-------------------|--------------|
| KV sharding | `cp_rank = pcp_rank × dcp_size + dcp_rank` | Unified in `block_table.py` |
| Token partitioning | One CP split policy per model class | **Two** policies in `PCPManager.update_tokens_for_pcp()` |
| Metadata | Shared prefill/decode CP metadata | Hybrid-only fields mixed with standard fields in `AscendPrefillContextParallelMetadata` |
| Attention | Common CP communication primitives | **Four** separate CP backends (GQA / MLA / DSA / SFA) |
| Model adaptation | Declarative capability flags | **Hardcoded** `model_type` whitelist |

**Impact:** Adapting a new model (especially hybrid FLA+Full-Attention architectures like Qwen3-Next/3.5) requires touching `pcp_utils.py`, `model_runner_v1.py`, one or more `*_cp.py` files, ACL Graph paths, and e2e tests—without a single registration point.

### 1.2 Concrete Pain Points

1. **`PCPManager` is misnamed and overloaded** — it owns DCP MTP masks, KV local seq-len calculation, MM preprocessing localization, and hybrid FLA/FA reorder indices, yet is only associated with PCP by name.

2. **`pcp_use_hybrid_attn` is a model-type whitelist** (`qwen3_next`, `qwen3_5`, `qwen3_5_moe`). A new hybrid model fails silently or produces wrong logits/KV unless this list is manually extended.

3. **Dual code paths** in `update_tokens_for_pcp`, `get_logits_indices`, `get_restore_hidden_states`, `get_padded_slot_mapping`, and `model_runner_v1._prepare_inputs` double maintenance cost and block ACL Graph support for hybrid models (`FIXME: support hybrid attn backend` in `aclgraph_utils.py`).

4. **Upstream vLLM capability checks are unused** — `AttentionImpl.supports_pcp`, `can_return_lse_for_decode`, and `check_attention_cp_compatibility()` exist in upstream `vllm/v1/worker/cp_utils.py`, but vLLM Ascend attention impls do not declare these flags (grep shows zero matches in `vllm_ascend/`).

5. **v2 ModelRunner gap** — `worker/v2/model_runner.py` has `dcp_local_seq_lens=None  # TODO: support cp`, indicating the fragmentation will worsen across runner generations without intervention.

6. **Technical debt markers** — `# TODO: To be refactored` on `attn_chunk_seqlens` in `attention/utils.py`; PCP-only restore logic explicitly disabled for DCP-only in `attention_cp.py` comments.

---

## 2. Current Architecture Analysis

### 2.1 Component Map

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         model_runner_v1.py                              │
│  use_cp = pcp_size * dcp_size > 1                                       │
│  branches on pcp_manager.pcp_use_hybrid_attn (15+ call sites)           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PCPManager (pcp_utils.py, ~1450 LOC)                 │
├─────────────────────────────────────────────────────────────────────────┤
│ PCP token split     │ update_tokens_for_pcp()                           │
│                     │   ├─ Standard: DualChunkSwap head-tail            │
│                     │   └─ Hybrid:  FLA-aligned via _get_cp_local_seq_lens│
│ DCP KV lengths      │ _get_cp_local_seq_lens()                          │
│ DCP MTP mask        │ generate_mtp_attention_mask_for_decode()            │
│ PCP long-seq meta   │ generate_pcp_metadata() [gated: pcp_size > 1]    │
│ MM under CP         │ gather_mm_embeddings_for_pcp(), localize scheduler  │
│ Spec decode / MTP   │ generate_pcp_mtp_input()                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  block_table.py │  │ attention/utils.py  │  │ context_parallel/*.py   │
│  slot_mapping   │  │ AscendPrefillCPMeta │  │ attention_cp (GQA)      │
│  cp_rank unified│  │ hybrid + std fields │  │ mla_cp, dsa_cp, sfa_cp  │
└─────────────────┘  └─────────────────────┘  └─────────────────────────┘
```

### 2.2 Token Partitioning Strategies (Today)

#### Strategy A: DualChunkSwap (default)

- **Used when:** `pcp_use_hybrid_attn == False`
- **Mechanism:** Pad each prefill request to `2 × pcp_world_size`, split into head/tail chunks assigned to ranks in interleaved fashion; decode tokens are duplicated across PCP ranks.
- **Key outputs:** `pcp_tokens`, `pcp_positions`, `pcp_allgather_restore_idx`, `pcp_unpad_mask_cpu`
- **Doc reference:** [PCP Partition diagram](../../assets/cp/head-tail-style.png)

#### Strategy B: Hybrid FLA+FA Alignment

- **Used when:** `model_type ∈ {qwen3_next, qwen3_5, qwen3_5_moe}`
- **Mechanism:** Prefill tokens split via `_get_cp_local_seq_lens()` to align with linear-attention (FLA) layer sharding; separate enter/exit indices bridge FLA ↔ full-attention (FA) sub-layers.
- **Key outputs:** All of Strategy A **plus** `pcp_enter_fa_restore_idx`, `pcp_exit_fa_scatter_idx`, `pcp_fa_query_idx`, `pcp_padded_tokens_fla`, `max_num_tokens_across_pcp`
- **Downstream:** `get_restore_hidden_states()` and `get_logits_indices()` use entirely different formulas.

```text
  Standard path                          Hybrid path
  ─────────────                          ───────────
  positions = head/tail map              positions_linear + FLA padding
  restore = allgather_restore_idx        restore = enter_fa_restore_idx
  logits  = cu * pcp_size - pads         logits  = cumsum(tokens + decode_pads)
```

### 2.3 DCP Integration Points

DCP is **not** a separate manager; it appears as:

| Function | Role |
|----------|------|
| `block_table.compute_slot_mapping()` | Virtual block + interleaved KV placement (`cp_kv_cache_interleave_size`) |
| `_get_cp_local_seq_lens()` | Per-rank local context length for chunked prefill |
| `generate_mtp_attention_mask_for_decode()` | MTP token assignment via `position % cp_size` |
| `common_cp._process_attn_out_lse()` | DCP all-to-all on LSE, then PCP all-gather |
| `attention_cp.py` chunked prefill | DCP: `kv_inverse_idx` set to `None` when `pcp_size == 1` |

**Observation:** DCP-only (`pcp_size=1, dcp_size>1`) and PCP-only (`pcp_size>1, dcp_size=1`) are special cases of the unified `cp_rank` formula, but metadata and token-split code still assume PCP semantics in many paths.

### 2.4 Attention Backend CP Matrix

| Backend | CP module | Chunked prefill comm | PCP prefill | DCP decode LSE |
|---------|-----------|---------------------|-------------|----------------|
| GQA | `attention_cp.py` | AllGatherQ | Head/tail indices | `cp_lse_ag_out_rs` / all-to-all |
| MLA | `mla_cp.py` | AllGatherKV | AllGather KV per layer | LSE return required |
| DSA | `dsa_cp.py` | Custom `DSACPMetadata` | Partial | Partial |
| SFA | `sfa_cp.py` | Custom | Partial | Partial |

`common_cp.py` unifies decode output merging (`_process_attn_out_lse`, `_npu_attention_update`) but **not** prefill metadata construction—each builder duplicates PCP head/tail index logic or re-implements chunked context structs.

### 2.5 Call-Site Fragmentation (`model_runner_v1.py`)

Representative branches tied to `pcp_use_hybrid_attn`:

| Location | Behavior difference |
|----------|---------------------|
| `_prepare_inputs` ~L852 | `query_lens` from `num_scheduled_tokens_padded` vs scheduler output |
| `_prepare_inputs` ~L1284 | `get_logits_indices` hybrid vs standard |
| `_build_attention_metadata` ~L2988 | `maybe_pcp_full_tokens` calculation |
| Graph capture ~L2150, L2175 | Skip token count sync for hybrid |
| `_calc_mrope_positions` ~L2988 | Hybrid slot mapping size |

Any new partitioning strategy must be replicated at each site unless centralized.

### 2.6 Test Coverage Gaps

- `tests/ut/worker/test_pcp_manager.py` — metadata generation; does not cover hybrid FLA path or strategy selection.
- `tests/ut/attention/a2/test_*_cp*.py` — per-backend precision tests; no cross-backend metadata contract tests.
- `tests/e2e/.../long_sequence/` — end-to-end CP scenarios; tied to specific models (Qwen).
- **Missing:** strategy plugin unit tests, DCP-only + hybrid model matrix, v2 runner CP parity tests.

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. **Single CP entry point** — Replace ad-hoc `PCPManager` branches with a `ContextParallelManager` orchestrating pluggable strategies.
2. **Declarative model profiles** — New models register a `CPModelProfile` (strategy, backends, MTP rules) without editing core runner logic.
3. **Align with upstream vLLM** — Implement `supports_pcp`, `can_return_lse_for_decode`, and call `check_attention_cp_compatibility()` at init.
4. **Unified metadata contract** — Split `AscendPrefillContextParallelMetadata` into a **core** struct + **strategy extension** struct.
5. **Phased migration** — Keep backward compatibility; no behavior change for existing Qwen CP models in Phase 1.
6. **v2 readiness** — Design APIs consumable by `worker/v2/model_runner.py` without copy-paste.

### 3.2 Non-Goals (this RFC)

- Changing DualChunkSwap or FLA alignment **algorithms** (only reorganize code).
- Ring Attention or new communication patterns.
- Multi-node CP topology changes.
- Full ACL Graph hybrid support in Phase 1 (tracked as Phase 3).

---

## 4. Proposed Design

### 4.1 Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                  ContextParallelManager (new)                    │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ TokenPartition │  │ KVShardCalculator │  │ MetadataFactory │ │
│  │ Strategy       │  │ (wraps block_table│  │ (per attn type) │ │
│  │ (pluggable)    │  │  formulas)        │  │                 │ │
│  └───────┬────────┘  └────────┬─────────┘  └────────┬────────┘ │
│          │                      │                      │          │
│          └──────────────────────┼──────────────────────┘          │
│                                 ▼                                 │
│                    CPBatchLayout (immutable snapshot)               │
└──────────────────────────────────────────────────────────────────┘
          ▲
          │ resolved from
┌─────────┴──────────┐
│ CPModelProfile     │  ← model config / registry / layer scan
│ - partition_strategy│
│ - attn_cp_backends │
│ - mtp_cp_policy    │
└────────────────────┘
```

### 4.2 Core Types

#### `CPBatchLayout`

Immutable per-forward snapshot consumed by ModelRunner and attention builders—**replaces** reading a dozen `PCPManager` fields directly.

```python
@dataclass(frozen=True)
class CPBatchLayout:
    cp_size: int
    cp_rank: int
    pcp_size: int
    dcp_size: int
    num_reqs: int
    num_decode_reqs: int
    num_prefill_reqs: int

    # Token layout (local to this rank)
    local_num_scheduled_tokens: np.ndarray      # [num_reqs]
    local_positions: np.ndarray                 # flattened
    total_local_tokens: int

    # Restore / mask (strategy-specific; optional fields)
    allgather_restore_idx: torch.Tensor | None
    unpad_mask: np.ndarray | None
    hybrid: "HybridCPExtension | None"        # None for standard models

    # DCP
    local_context_lens: np.ndarray | None       # [num_reqs, pcp, dcp] or flat
    dcp_mtp_attn_mask: torch.Tensor | None
```

#### `TokenPartitionStrategy` (Protocol)

```python
class TokenPartitionStrategy(Protocol):
    name: str  # e.g. "dual_chunk_swap", "hybrid_fla_fa"

    def partition(
        self,
        num_scheduled_tokens: np.ndarray,
        batch_info: CPBatchInfo,
        cp_env: CPEnvironment,
    ) -> CPBatchLayout: ...

    def logits_indices(
        self, layout: CPBatchLayout, cu_num_tokens: np.ndarray, tokens_original: list[int] | None
    ) -> torch.Tensor: ...

    def restore_hidden_states(
        self, layout: CPBatchLayout, hidden_states: torch.Tensor, pcp_group
    ) -> torch.Tensor: ...

    def padded_slot_mapping(
        self, layout: CPBatchLayout, slot_mapping: torch.Tensor, kv_cache_group_id: int
    ) -> torch.Tensor: ...
```

**Implementations (Phase 1):**

| Class | Replaces |
|-------|----------|
| `DualChunkSwapStrategy` | `update_tokens_for_pcp` else-branch |
| `HybridFlaFaStrategy` | `update_tokens_for_pcp` `pcp_use_hybrid_attn` branch |

#### `CPModelProfile`

```python
@dataclass
class CPModelProfile:
    partition_strategy: str  # registry key
    hybrid_attention: bool = False
    # Optional overrides; default from layer scan
    requires_pcp_long_seq_metadata: bool = False
    chunked_prefill_comm: Literal["all_gather_q", "all_gather_kv"] | None = None
```

**Resolution order:**

1. Explicit `parallel_config.cp_partition_strategy` (new config field, optional).
2. `CPModelRegistry` lookup by `model_type` / `architectures`.
3. **Auto-detect:** scan model layers for FLA+FA pattern (same info `hybrid_with_attn_and_mamba` uses in runner).
4. Fallback: `dual_chunk_swap`.

This removes the hardcoded tuple in `PCPManager.__init__`.

#### `CPMetadataFactory`

Builds `AscendPrefillContextParallelMetadata` from `CPBatchLayout` + attention backend type:

```python
class CPMetadataFactory:
    def build(
        self,
        layout: CPBatchLayout,
        backend_kind: Literal["gqa", "mla", "dsa", "sfa"],
        input_batch: NPUInputBatch,
        block_table: torch.Tensor,
    ) -> AscendPrefillContextParallelMetadata: ...
```

Long-sequence head/tail index generation (today ~140 LOC in `generate_pcp_metadata`) moves to `DualChunkSwapMetadataBuilder`. Hybrid extensions move to `HybridFlaFaMetadataBuilder`.

### 4.3 Metadata Schema Split

**Today:** `AscendPrefillContextParallelMetadata` mixes 30+ fields; hybrid-only fields are `None` for standard models.

**Proposed:**

```python
@dataclass
class CPMetadataCore:
    num_actual_tokens_pcp_padded: int
    num_computed_tokens_of_pcp_dcp: np.ndarray
    pcp_unpad_mask: torch.Tensor | None
    query_lens_pcp_full_cpu: torch.Tensor
    max_query_len_pcp_full: int
    dcp_mtp_attn_mask: torch.Tensor | None

@dataclass
class DualChunkSwapPCPExtension:
    pcp_allgather_restore_idx: torch.Tensor
    q_head_idx_tensor: torch.Tensor
    q_tail_idx_tensor: torch.Tensor
    # ... existing head/tail KV index fields

@dataclass
class HybridFlaFaPCPExtension:
    pcp_enter_fa_restore_idx: torch.Tensor
    pcp_exit_fa_scatter_idx: torch.Tensor
    pcp_fa_query_idx: torch.Tensor
    pcp_padded_tokens_fla: int
    max_num_tokens_across_pcp: int
    total_num_scheduled_tokens: int
    attn_chunk_seqlens: torch.Tensor

@dataclass
class AscendPrefillContextParallelMetadata:
    core: CPMetadataCore
    dual_chunk_swap: DualChunkSwapPCPExtension | None = None
    hybrid_fla_fa: HybridFlaFaPCPExtension | None = None
```

Attention CP builders read via typed accessors; deprecate flat field access over two release cycles.

### 4.4 Upstream Capability Integration

Each Ascend `*CPImpl` should set:

```python
class AscendAttentionCPImpl(AttentionImpl):
    supports_pcp = True
    can_return_lse_for_decode = True
    supports_mtp_with_cp_non_trivial_interleave_size = False  # per backend
```

Call at worker init (patch or `platform.py`):

```python
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
check_attention_cp_compatibility(vllm_config)
```

Fail fast when DCP is enabled but backend cannot return LSE.

### 4.5 ModelRunner Integration

**Before:**

```python
num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(...)
if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
    self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
```

**After:**

```python
layout = self.cp_manager.build_batch_layout(num_scheduled_tokens, num_reqs, ...)
positions_np = layout.local_positions
self.query_lens = layout.query_lens_for_attention  # property, strategy-defined
# No hybrid-specific if/else in runner
```

`model_runner_v1.py` should only check `self.use_cp` and consume `CPBatchLayout` / `CPMetadataCore`—not strategy names.

### 4.6 Registry Example

```python
# vllm_ascend/context_parallel/registry.py
CP_MODEL_REGISTRY: dict[str, CPModelProfile] = {
    "qwen3_next": CPModelProfile(
        partition_strategy="hybrid_fla_fa",
        hybrid_attention=True,
        requires_pcp_long_seq_metadata=True,
        chunked_prefill_comm="all_gather_q",
    ),
    "qwen3_5": CPModelProfile(...),
    "qwen3_5_moe": CPModelProfile(...),
    # New model: one entry, no pcp_utils edit
    "qwen4_next": CPModelProfile(partition_strategy="hybrid_fla_fa", ...),
}
```

### 4.7 File Layout (Target)

```text
vllm_ascend/context_parallel/
├── __init__.py
├── manager.py              # ContextParallelManager
├── layout.py               # CPBatchLayout, CPBatchInfo
├── registry.py             # CPModelProfile, CP_MODEL_REGISTRY
├── kv_sharding.py          # _get_cp_local_seq_lens (moved from pcp_utils)
├── strategies/
│   ├── base.py             # TokenPartitionStrategy protocol
│   ├── dual_chunk_swap.py
│   └── hybrid_fla_fa.py
├── metadata/
│   ├── factory.py
│   ├── dual_chunk_swap.py
│   └── hybrid_fla_fa.py
└── mtp.py                  # generate_mtp_attention_mask_for_decode

vllm_ascend/worker/pcp_utils.py  → thin shim re-exporting ContextParallelManager (deprecated alias)
```

---

## 5. Migration Plan

### Phase 1: Extract & Shim (4–6 weeks)

| Task | Risk |
|------|------|
| Introduce `CPBatchLayout` + strategies without behavior change | Low — copy existing logic |
| `PCPManager` delegates to `ContextParallelManager` | Low — keep public API |
| Add `CPModelRegistry`; map existing 3 model types | Low |
| Unit tests: parity `test_pcp_manager` vs new strategies | **Gate for merge** |
| e2e: re-run `long_sequence/test_*.py` | **Gate for merge** |

**Exit criteria:** Bit-exact outputs for existing CP e2e tests on Qwen3-Next / Qwen3.5 with PCP2 DCP2 TP4.

### Phase 2: Unify Metadata & Runner (4–6 weeks)

| Task | Risk |
|------|------|
| Split `AscendPrefillContextParallelMetadata` | Medium — touch all `*_cp.py` |
| Refactor `model_runner_v1` to use `CPBatchLayout` only | Medium |
| Implement `supports_pcp` on all CP impls + init check | Low |
| Consolidate chunked-prefill metadata in `common_cp.py` | Medium |
| Add registry auto-detect for hybrid layers | Medium |

**Exit criteria:** No `pcp_use_hybrid_attn` references outside `context_parallel/` package.

### Phase 3: v2 + ACL Graph + New Models (ongoing)

| Task | Risk |
|------|------|
| Wire `worker/v2/model_runner.py` through `ContextParallelManager` | High |
| ACL Graph support for `HybridFlaFaStrategy` | High |
| Document "Add a new CP model" guide (5-step checklist) | Low |
| Remove `PCPManager` shim | Low after v2 parity |

---

## 6. Adding a New Model (Target Developer Experience)

```text
1. Determine CP profile:
   - Standard dense/MLA → dual_chunk_swap
   - FLA+FA hybrid      → hybrid_fla_fa

2. Add registry entry in registry.py (or enable auto-detect).

3. Confirm attention backend CP impl exists (or extend metadata factory).

4. Set impl flags: supports_pcp, can_return_lse_for_decode.

5. Run: tests/ut/context_parallel/test_strategy_parity.py
         tests/e2e/.../long_sequence/test_accuracy.py
```

**No edits** to `model_runner_v1.py` branching logic.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Behavior regression on hybrid models | Mandatory bit-exact parity tests before/after per strategy |
| Metadata schema migration breaks plugins | Deprecation shim on flat metadata fields for 2 releases |
| DCP-only paths still broken for chunked prefill | Explicit `pcp_size==1` test matrix in Phase 2 |
| v1/v2 divergence | Shared `context_parallel/` package, runner-agnostic layouts |
| Performance regression from extra indirection | `CPBatchLayout` built once per forward; strategies use numpy/torch same as today |

---

## 8. Open Questions

1. **Auto-detect vs registry:** Should hybrid FLA+FA detection scan `model_config` layers at init, or rely solely on explicit registry for predictability?

2. **Config surface:** Expose `--cp-partition-strategy` CLI flag for debugging, or keep internal-only?

3. **DSA/SFA unification priority:** Phase 2 focuses on GQA+MLA (production CP paths). Should DSA/SFA share `CPMetadataFactory` in Phase 2 or Phase 3?

4. **Upstream contribution:** Can `CPBatchLayout` / strategy protocol be proposed to vLLM core to reduce v1/v2 fork?

5. **Mamba + CP:** `MambaSpec` block table expansion is separate from token partition—should `KVShardCalculator` own Mamba-specific rules?

---

## 9. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Files touched to add hybrid CP model | 5–8 | 1–2 (registry + optional backend) |
| `pcp_use_hybrid_attn` branches in runner | ~15 | 0 |
| LOC in monolithic `pcp_utils.py` | ~1450 | <400 shim + distributed package |
| CP unit tests with strategy parameterization | 1 file | Full strategy × {PCP,DCP,CP} matrix |
| v2 CP support | TODO comment | `dcp_local_seq_lens` populated |

---

## 10. References

### Internal

- `vllm_ascend/worker/pcp_utils.py` — `PCPManager`
- `vllm_ascend/worker/model_runner_v1.py` — CP call sites
- `vllm_ascend/worker/block_table.py` — unified `cp_rank` slot mapping
- `vllm_ascend/attention/context_parallel/` — per-backend CP
- `vllm_ascend/attention/utils.py` — `AscendPrefillContextParallelMetadata`
- `tests/ut/worker/test_pcp_manager.py`
- `tests/e2e/pull_request/four_card/long_sequence/`

### Upstream

- `vllm/v1/worker/cp_utils.py` — `check_attention_cp_compatibility`
- `vllm/v1/attention/backend.py` — `supports_pcp`, `can_return_lse_for_decode`
- [vLLM Issue #25749](https://github.com/vllm-project/vllm/issues/25749)

---

## Appendix A: Field Mapping (PCPManager → CPBatchLayout)

| PCPManager field | CPBatchLayout / extension |
|------------------|---------------------------|
| `pcp_tokens` | `local_num_scheduled_tokens` |
| `pcp_allgather_restore_idx` | `allgather_restore_idx` |
| `pcp_unpad_mask_cpu` | `unpad_mask` |
| `num_scheduled_tokens_padded` | `hybrid.query_lens_padded` |
| `pcp_enter_fa_restore_idx` | `hybrid.enter_fa_restore_idx` |
| `pcp_exit_fa_scatter_idx` | `hybrid.exit_fa_scatter_idx` |
| `num_computed_tokens_of_pcp_dcp` | `local_context_lens` |
| `dcp_mtp_attn_mask` | `dcp_mtp_attn_mask` |

## Appendix B: Glossary

| Term | Meaning |
|------|---------|
| **PCP** | Prefill Context Parallel — extra devices, sequence split at prefill |
| **DCP** | Decode Context Parallel — KV dedup within TP domain |
| **CP** | Context Parallel — collective term; `cp_size = pcp × dcp` |
| **DualChunkSwap** | Head-tail interleaved sequence partition for PCP |
| **Hybrid FLA+FA** | Models mixing linear attention (FLA) and full attention (FA) layers |
| **MTP** | Multi-Token Prediction / speculative decode tokens |
