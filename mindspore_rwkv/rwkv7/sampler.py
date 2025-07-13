import mindspore
from mindnlp.core import ops, nn


def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = nn.functional.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = ops.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = ops.cumsum(sorted_probs, dim=-1)
        cutoff_index = ops.searchsorted(cumulative_probs, mindspore.Tensor([[top_p]] * logits.shape[0]))
        cutoff = sorted_probs.gather(dim=1, index=cutoff_index)
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = probs == cutoff.broadcast_to(probs.shape)
            len_idx = idx.sum(axis=1, keepdims=True)
            len_idx[len_idx == 0] = 1 # 防止除零
            update_value = cutoff + (top_p - probs.sum(axis=1, keepdims=True)) / len_idx
            probs = ops.where(idx, update_value.broadcast_to(probs.shape), probs)
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return ops.multinomial(probs, num_samples=1).squeeze(-1)

# def sample_logits(out: mindspore.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> mindspore.Tensor:
#     """
#     Sample from the logits tensor produced by the model.

#     Args:
#         out (mindspore.Tensor): Logits tensor from the model, shape [* , vocab_size].
#         temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
#         top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

#     Returns:
#         mindspore.Tensor: Sampled indices, shape [*].
#     """
#     # Apply temperature scaling
#     scaled_logits = out / temperature

#     # Convert logits to probabilities
#     probabilities = ops.softmax(scaled_logits, dim=-1)

#     # Sort the probabilities to identify the top-p candidates
#     sorted_probs, sorted_indices = ops.sort(probabilities, descending=True)

#     # Compute the cumulative distribution of probabilities
#     cumulative_probs = ops.cumsum(sorted_probs, dim=-1)

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
#     if ops.all(probabilities == 0):
#         probabilities = ops.ones_like(probabilities)
#         probabilities /= probabilities.sum()

#     # Sample from the modified distribution
#     sampled_indices = ops.multinomial(probabilities, 1)

#     return sampled_indices.squeeze(-1)
