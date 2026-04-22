# 🔍 MechInterp-Starter: Transformer Attention Visualizer

> A beginner-friendly PyTorch project for **Mechanistic Interpretability** research.  
> Train a tiny transformer, then *see inside it* — head by head, token by token.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Project Does

This project trains a **1-layer, 4-head transformer** on a sequence reversal task, then provides an interactive HTML visualizer to inspect what each attention head has learned.

**Why reversal?** It's the simplest task where a transformer *must* learn structured, interpretable attention patterns. When it succeeds, you can literally see position-mirroring in the attention matrix — making it a perfect sandbox for mechanistic interpretability.

```
Input:  [3, 7, 1, 5, 2, 9, SEP]
Output: [9, 2, 5, 1, 7, 3, EOS]
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Multi-head self-attention (from scratch) | `train.py` → `MultiHeadSelfAttention` |
| Attention weight extraction | `.last_attn_weights` capture |
| Positional embeddings | `MiniTransformer.pos_emb` |
| Weight tying (output ↔ embedding) | `self.head.weight = self.token_emb.weight` |
| Head entropy analysis | `visualize.html` → Stats panel |
| Functional head specialisation | MI Insight panel |

---

## Results

The model reaches **100% sequence accuracy** on held-out data in under 10 epochs — with only **34,624 parameters**.

---

## Project Structure

```
mech-interp-starter/
├── train.py            ← PyTorch model + training loop
├── visualize.html      ← Interactive attention visualizer
├── outputs/
│   ├── model.pt        ← Saved weights
│   └── attention_data.json  ← Per-head attention matrices
└── README.md
```

---

## Quick Start

```bash
# 1. Install
pip install torch

# 2. Train
python train.py

# 3. Visualize (open in browser)
open visualize.html
```

---

## What to Look For in the Visualizer

- **Sharp head (low entropy):** Attends to exactly one position → likely learned position mirroring
- **Diffuse head (high entropy):** Spreads attention → likely averaging context
- **SEP-attending head:** One head often learns to use the separator as a global "sync" token

These are real MI findings, not just pretty pictures. This is the same kind of analysis done in papers like [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (Elman et al., Anthropic).

---

## Suggested Experiments

1. **Scale up**: Change `n_layers=2` — does a second layer build on the first?
2. **Change task**: Try sorting instead of reversing
3. **Ablate heads**: Zero out one head's output — does accuracy drop?
4. **Attention entropy over training**: Log entropy per epoch to watch heads specialise

---

## References

- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.* ICLR.

---

*Built as a beginner PyTorch project · Designed for MMU Mechanistic Interpretability research*
# Transformer-Attention-Visualizer
