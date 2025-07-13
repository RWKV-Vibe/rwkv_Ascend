import torch
import torch.nn.functional as F


def sample_logits(logits: torch.Tensor, temperature:float=1.0, top_p:float=1.0, top_k:int=0) -> torch.tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, torch.tensor([[top_p]] * logits.shape[0], device=logits.device))
        cutoff = sorted_probs.gather(dim=1, index=cutoff_index)
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = probs == cutoff.broadcast_to(probs.shape)
            len_idx = idx.sum(axis=1, keepdims=True)
            len_idx[len_idx == 0] = 1 # 防止除零
            update_value = cutoff + (top_p - probs.sum(axis=1, keepdims=True)) / len_idx
            probs = torch.where(idx, update_value.broadcast_to(probs.shape), probs)
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)

# def sample_logits(out: torch.tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.tensor:
#     """
#     Sample from the logits tensor produced by the model.

#     Args:
#         out (torch.tensor): Logits tensor from the model, shape [* , vocab_size].
#         temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
#         top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

#     Returns:
#         torch.tensor: Sampled indices, shape [*].
#     """
#     # Apply temperature scaling
#     scaled_logits = out / temperature

#     # Convert logits to probabilities
#     probabilities = torch.softmax(scaled_logits, dim=-1)

#     # Sort the probabilities to identify the top-p candidates
#     sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

#     # Compute the cumulative distribution of probabilities
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

#     # Remove tokens with a cumulative probability above the threshold (top_p)
#     sorted_indices_to_remove = cumulative_probs > top_p
#     # Shift the indices to the right to keep the first token above the threshold
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
#     sorted_indices_to_remove[..., 0] = 0

#     # Create a mask for the indices to remove
#     indices_to_remove = sorted_indices_to_remove.scatter(axis=-1, index=sorted_indices, src=sorted_indices_to_remove)

#     # Use the mask to zero out probabilities that should be removed
#     probabilities[indices_to_remove] = 0.0

#     # Resample if probabilities are all zero (unlikely but just in case)
#     if torch.all(probabilities == 0):
#         probabilities = torch.ones_like(probabilities)
#         probabilities /= probabilities.sum()

#     # Sample from the modified distribution
#     sampled_indices = torch.multinomial(probabilities, 1)

#     return sampled_indices.squeeze(-1)
