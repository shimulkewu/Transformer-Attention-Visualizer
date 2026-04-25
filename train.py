"""
=============================================================
 MechInterp-Starter: Attention Head Visualizer
 A beginner-friendly Mechanistic Interpretability project
 Author: Md Yakub Hossan Khan
 Compatible with: PyTorch >= 2.0
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 1. DATASET  — Reverse-the-Sequence Task
#    The model must learn to reverse a token sequence.
#    Simple task → clean, interpretable attention patterns.
# ─────────────────────────────────────────────

class ReverseSequenceDataset(Dataset):
    """
    Each sample: input  = [a, b, c, d, <SEP>]
                 target = [d, c, b, a, <EOS>]
    Vocabulary: digits 0-9, plus special tokens
    """
    VOCAB_SIZE = 12           # 0-9 + SEP(10) + EOS(11)
    SEP_TOKEN  = 10
    EOS_TOKEN  = 11

    def __init__(self, n_samples=10_000, seq_len=8):
        self.seq_len  = seq_len
        self.n_samples = n_samples
        self.data = self._generate()

    def _generate(self):
        data = []
        for _ in range(self.n_samples):
            seq    = torch.randint(0, 10, (self.seq_len,))
            inp    = torch.cat([seq, torch.tensor([self.SEP_TOKEN])])
            target = torch.cat([seq.flip(0), torch.tensor([self.EOS_TOKEN])])
            data.append((inp, target))
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
# 2. MODEL — Single-Layer Transformer
#    Deliberately small so attention heads are
#    easy to inspect and understand.
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Vanilla multi-head attention with attention weight capture.
    We store the weights so we can visualize them later.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_model  = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ← This is what makes this project MI-friendly:
        #   we capture attention weights every forward pass
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)   # (B,H,T,Dh)
        K = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (Dh ** 0.5)    # (B,H,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach().cpu()         # ← SAVE

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, n_heads)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1   = nn.LayerNorm(d_model)
        self.ln2   = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class MiniTransformer(nn.Module):
    """
    Tiny 1-layer transformer for sequence reversal.
    Kept small intentionally: every component is interpretable.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=4,
                 d_ff=128, max_len=32, n_layers=1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: output projection shares weights with embedding
        self.head.weight = self.token_emb.weight

    def forward(self, x, mask=None):
        B, T    = x.shape
        pos     = torch.arange(T, device=x.device).unsqueeze(0)
        emb     = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            emb = block(emb, mask)
        logits  = self.head(self.ln_final(emb))
        return logits

    def get_attention_weights(self):
        """Return attention maps from all layers — key for MI."""
        return [block.attn.last_attn_weights for block in self.blocks]


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  MechInterp-Starter Training")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    # Data
    train_ds = ReverseSequenceDataset(n_samples=config["n_train"], seq_len=config["seq_len"])
    val_ds   = ReverseSequenceDataset(n_samples=1000,              seq_len=config["seq_len"])
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)

    # Model
    model = MiniTransformer(
        vocab_size = ReverseSequenceDataset.VOCAB_SIZE,
        d_model    = config["d_model"],
        n_heads    = config["n_heads"],
        d_ff       = config["d_ff"],
        max_len    = config["seq_len"] + 2,
        n_layers   = config["n_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config["epochs"] + 1):
        # — Train —
        model.train()
        total_loss = 0
        for inp, tgt in train_dl:
            inp, tgt = inp.to(device), tgt.to(device)
            logits   = model(inp)               # (B, T+1, vocab)
            loss     = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_dl)

        # — Validate —
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inp, tgt in val_dl:
                inp, tgt = inp.to(device), tgt.to(device)
                logits   = model(inp)
                val_loss += F.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1)
                ).item()
                preds    = logits.argmax(-1)
                correct += (preds == tgt).all(dim=1).sum().item()
                total   += tgt.size(0)

        avg_val = val_loss / len(val_dl)
        acc     = correct / total * 100

        history["train_loss"].append(round(avg_train, 4))
        history["val_loss"].append(round(avg_val, 4))
        history["val_acc"].append(round(acc, 2))

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} │ "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  acc={acc:.1f}%")

    # Save model + history
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pt")
    with open("outputs/history.json", "w") as f:
        json.dump(history, f)
    print(f"\n  ✓ Model saved to outputs/model.pt")

    # Extract and save attention weights for visualization
    probe_attention(model, device, config)
    return history


def probe_attention(model, device, config):
    """
    Run a few examples through the model and save the
    attention weight matrices for each head.
    This is the heart of mechanistic interpretability!
    """
    model.eval()
    examples = []
    ds = ReverseSequenceDataset(n_samples=20, seq_len=config["seq_len"])

    with torch.no_grad():
        for i in range(10):
            inp, tgt = ds[i]
            inp_d    = inp.unsqueeze(0).to(device)
            logits   = model(inp_d)
            pred     = logits.argmax(-1).squeeze().tolist()
            attn     = model.get_attention_weights()   # list of (1,H,T,T)

            examples.append({
                "input":   inp.tolist(),
                "target":  tgt.tolist(),
                "pred":    pred,
                "correct": pred == tgt.tolist(),
                # Store per-head attention as nested lists
                "attn_layer0": attn[0][0].tolist()  # (H, T, T)
            })

    with open("outputs/attention_data.json", "w") as f:
        json.dump({
            "examples":   examples,
            "vocab_size": ReverseSequenceDataset.VOCAB_SIZE,
            "seq_len":    config["seq_len"],
            "n_heads":    config["n_heads"],
            "sep_token":  ReverseSequenceDataset.SEP_TOKEN,
            "eos_token":  ReverseSequenceDataset.EOS_TOKEN,
        }, f)
    print("  ✓ Attention data saved to outputs/attention_data.json")


# ─────────────────────────────────────────────
# 4. RUN
# ─────────────────────────────────────────────

CONFIG = {
    "seq_len":    6,
    "n_train":    20_000,
    "batch_size": 128,
    "epochs":     40,
    "d_model":    64,
    "n_heads":    4,
    "d_ff":       128,
    "n_layers":   1,
    "lr":         3e-3,
}

if __name__ == "__main__":
    history = train(CONFIG)
    print("\n  Training complete! Run `python visualize.py` to see attention heads.")
