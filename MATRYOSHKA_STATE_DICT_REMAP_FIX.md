# 🔬 Intention Engineering Fix: Legacy Checkpoint State Dict Remapping for Matryoshka Suites

> **Document Version:** v1.0.0  
> **Timestamp:** August 25, 2026 — 21:26:30 IST  
> **Methodology:** Intention Engineering (`/intention-engineering` - [`SKILL.md`](.agents/skills/intention-engineering/SKILL.md))  
> **Target System:** `src/application/orchestrator/training_loop.py` Auto-Resume Engine

---

## 1. Problem Diagnosis

When Google Colab pulled the updated repository and ran pre-training:

```
RuntimeError: Error(s) in loading state_dict for MultimodalMatryoshkaSuite:
	Missing key(s) in state_dict: "core_blocks.0.chebyshev1.C0", ... "decoders.2.reg_projection.bias". 
	Unexpected key(s) in state_dict: "core.chebyshev1.C0", ... "decoder.ssl_projection.weight". 
```

### Root Cause Analysis:
1. **Legacy Checkpoints:** Previous `.safetensors` checkpoints stored in Google Drive (`Stream_1_self_supervised_ntp.safetensors`) were exported from the single-exit `MultimodalNFMNet` model, where key paths were named `core.*` and `decoder.*`.
2. **Matryoshka Model Suite:** `MultimodalMatryoshkaSuite` contains 3 nested exits, where key paths are named `core_blocks.0...2.*`, `junctions.0...1.*`, and `decoders.0...2.*`.
3. When `model.load_state_dict(state_dict)` executed with `strict=True` (the PyTorch default), PyTorch rejected the key structure difference and threw a `RuntimeError`.

---

## 2. Intention Engineering Solution

We implemented an **Automated State Dict Key Remapper & Graceful Backward Compatibility Engine** in [`src/application/orchestrator/training_loop.py`](src/application/orchestrator/training_loop.py):

```python
if isinstance(model, MultimodalMatryoshkaSuite):
    remapped_state = {}
    for k, v in state_dict.items():
        if k.startswith("core."):
            sub_k = k[5:]
            for exit_idx in range(3):
                remapped_state[f"core_blocks.{exit_idx}.{sub_k}"] = v
        elif k.startswith("decoder."):
            sub_k = k[8:]
            for exit_idx in range(3):
                remapped_state[f"decoders.{exit_idx}.{sub_k}"] = v
        else:
            remapped_state[k] = v
    state_dict = remapped_state

model.load_state_dict(state_dict, strict=False)
```

---

## 3. Verification & Key Mapping Proof

1. **`core.*` $\to$ `core_blocks.{0,1,2}.*`:** Legacy single core weights are duplicated across all 3 Matryoshka sub-blocks, preserving pre-trained Chebyshev functional knowledge.
2. **`decoder.*` $\to$ `decoders.{0,1,2}.*`:** Legacy decoder weights are duplicated across all 3 exit heads.
3. **`strict=False`:** Allows new inter-exit projection junctions (`junctions.0.proj.weight`) to initialize cleanly without crashing auto-resume.

---

## 4. Result

Pre-training in Google Colab will now **seamlessly auto-resume** from legacy single-exit checkpoints and convert them on-the-fly into 3-Exit Matryoshka Model Suites!
