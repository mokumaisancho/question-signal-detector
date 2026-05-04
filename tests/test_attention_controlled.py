"""Length-controlled attention focus test for knowledge boundary detection."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_attentions=True).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Matched-length questions
known = [
    "What is gravity and how does it work on Earth?",
    "What is DNA and what role does it play?",
    "What is Python and why is it popular today?",
    "What is CRISPR and how does gene editing work?",
    "What is climate change and what are the effects?",
]
unknown = [
    "Can topological persistence detect phase transitions?",
    "Can sheaf cohomology detect misinformation cascades?",
    "Does Wasserstein distance predict discovery novelty?",
    "Who won the 2032 presidential election result?",
    "What is Mars Colony population in the year 2035?",
]


def get_metrics(question):
    ids = tokenizer.encode(question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    attn = torch.stack(out.attentions)  # (24, 1, 16, seq, seq)
    attn = attn.squeeze(1)  # (24, 16, seq, seq)
    avg_attn = attn.mean(dim=(0, 1))  # (seq, seq)
    entropy = -(avg_attn * (avg_attn + 1e-10).log()).sum(dim=-1).mean().item()
    sparsity = avg_attn.topk(min(3, avg_attn.shape[-1]), dim=-1).values.sum(dim=-1).mean().item()
    concentration = avg_attn.max(dim=-1).values.mean().item()
    return entropy, sparsity, concentration, ids.shape[1]


print("LENGTH-CONTROLLED ATTENTION FOCUS TEST")
print("=" * 80)
print(f"{'Question':<55} {'Ent':>6} {'Spars':>6} {'Conc':>6} {'Toks':>5}")
print("-" * 80)

print("KNOWN (should have LOW entropy, HIGH sparsity/concentration):")
known_results = []
for q in known:
    e, s, c, t = get_metrics(q)
    known_results.append((e, s, c, t))
    print(f"  {q:<53} {e:6.3f} {s:6.3f} {c:6.3f} {t:5d}")

print("\nUNKNOWN (should have HIGH entropy, LOW sparsity/concentration):")
unknown_results = []
for q in unknown:
    e, s, c, t = get_metrics(q)
    unknown_results.append((e, s, c, t))
    print(f"  {q:<53} {e:6.3f} {s:6.3f} {c:6.3f} {t:5d}")

import numpy as np

k_ent = [r[0] for r in known_results]
u_ent = [r[0] for r in unknown_results]
k_spar = [r[1] for r in known_results]
u_spar = [r[1] for r in unknown_results]
k_conc = [r[2] for r in known_results]
u_conc = [r[2] for r in unknown_results]
k_tok = [r[3] for r in known_results]
u_tok = [r[3] for r in unknown_results]

print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)
print(f"  Token lengths: known={np.mean(k_tok):.1f} vs unknown={np.mean(u_tok):.1f} (diff={np.mean(u_tok)-np.mean(k_tok):+.1f})")
print(f"  Entropy:       known={np.mean(k_ent):.3f} vs unknown={np.mean(u_ent):.3f} (gap={np.mean(u_ent)-np.mean(k_ent):+.3f})")
print(f"  Sparsity:      known={np.mean(k_spar):.3f} vs unknown={np.mean(u_spar):.3f} (gap={np.mean(u_spar)-np.mean(k_spar):+.3f})")
print(f"  Concentration: known={np.mean(k_conc):.3f} vs unknown={np.mean(u_conc):.3f} (gap={np.mean(u_conc)-np.mean(k_conc):+.3f})")

pooled_std_ent = np.sqrt((np.var(k_ent) + np.var(u_ent)) / 2)
cohens_d_ent = (np.mean(u_ent) - np.mean(k_ent)) / pooled_std_ent if pooled_std_ent > 0 else 0
print(f"\n  Cohen's d (entropy): {cohens_d_ent:.2f}")
print(f"  Signal persists after length control: {'YES' if cohens_d_ent > 1.0 else 'MARGINAL' if cohens_d_ent > 0.5 else 'NO'}")
