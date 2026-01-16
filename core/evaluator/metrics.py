import torch
import torch.nn.functional as F

def calculate_refusal_probability(logits, refusal_token_ids):
    # Applica Softmax per ottenere le probabilità
    probs = F.softmax(logits, dim=-1)
    # Somma le probabilità dei token identificati come 'rifiuto'
    refusal_prob = probs[refusal_token_ids].sum().item()
    return refusal_prob