import json
import matplotlib.pyplot as plt
import numpy as np

# Load attention data
with open("outputs/attention_data.json", "r") as f:
    data = json.load(f)

examples = data["examples"]
n_heads = data["n_heads"]
seq_len = data["seq_len"]
sep_token = data["sep_token"]
eos_token = data["eos_token"]

# Token to string mapping for nicer labels
def token_to_str(tok):
    if tok == sep_token:
        return "SEP"
    elif tok == eos_token:
        return "EOS"
    else:
        return str(tok)

# Loop through each example
for idx, ex in enumerate(examples):
    if not ex["correct"]:
        continue   # skip wrong predictions (optional)
    
    inp_tokens = ex["input"]
    target_tokens = ex["target"]
    pred_tokens = ex["pred"]
    
    # Create token label strings
    input_labels = [token_to_str(t) for t in inp_tokens]
    target_labels = [token_to_str(t) for t in target_tokens]
    
    print(f"\nExample {idx+1}:")
    print(f"  Input : {input_labels}")
    print(f"  Target: {target_labels}")
    print(f"  Pred  : {[token_to_str(p) for p in pred_tokens]}")
    
    # Attention weights: shape (n_heads, seq_len+1, seq_len+1)
    attn = np.array(ex["attn_layer0"])  # (H, T, T)
    
    # Plot each head separately
    fig, axes = plt.subplots(1, n_heads, figsize=(5*n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Head {h+1}")
        ax.set_xlabel("Key position (input sequence)")
        ax.set_ylabel("Query position")
        
        # Set tick labels (optional, can be dense)
        ax.set_xticks(range(len(input_labels)))
        ax.set_xticklabels(input_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(input_labels)))
        ax.set_yticklabels(input_labels, fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Attention Patterns for Example {idx+1} (Correct: {ex['correct']})")
    plt.tight_layout()
    plt.show()
    
    # Stop after showing first few examples (optional)
    if idx >= 4:
        break
    # Aggregate attention across correct examples
correct_attns = [np.array(ex["attn_layer0"]) for ex in examples if ex["correct"]]
if correct_attns:
    avg_attn = np.mean(correct_attns, axis=0)   # (H, T, T)
    
    fig, axes = plt.subplots(1, n_heads, figsize=(5*n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(avg_attn[h], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Head {h+1} (average over {len(correct_attns)} correct examples)")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)
    
    plt.suptitle("Average Attention Patterns (Correct Only)")
    plt.tight_layout()
    plt.show()