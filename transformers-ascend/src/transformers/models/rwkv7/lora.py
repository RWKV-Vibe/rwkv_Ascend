import math
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Optional, Tuple

class LoRA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True,
        activation: Optional[str] = 'tanh'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Not supported activation `{activation}`.")

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            self.activation,
            nn.Linear(low_rank_dim, output_dim, bias=bias)
        )
        self.apply(self._initialize_weights)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize weights to zero as in original code
        nn.init.zeros_(self.lora[0].weight)
        original_dtype = self.lora[2].weight.dtype
        shape = self.lora[2].weight.shape
        # Convert to float32 for numerical stability in orthogonal init
        weight_fp32 = self.lora[2].weight.float()

        # Calculate gain based on dimensions
        gain = math.sqrt(shape[1] / shape[0]) if shape[1] > shape[0] else 1

        # Apply orthogonal initialization with scaling factor 0.1
        nn.init.orthogonal_(weight_fp32, gain=gain * 0.1)

        # Convert back to original dtype
        self.lora[2].weight.data.copy_(weight_fp32.to(original_dtype))
        # Set Lora[2] bias to zero
        if self.lora[2].bias is not None:
            nn.init.zeros_(self.lora[2].bias)

        module._is_hf_initialized = True

    def set_bias_value(self, value):
        """Set bias to a specific value (for v0, w0 etc.)"""
        if self.bias and self.lora[2].bias is not None:
            if isinstance(value, torch.Tensor):
                # Handle tensor values
                self.lora[2].bias.data.copy_(value.to(self.lora[2].bias.dtype))
            else:
                # Handle scalar values
                nn.init.constant_(self.lora[2].bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)