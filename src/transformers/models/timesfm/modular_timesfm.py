# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch TimesFM model."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ..llama.modeling_llama import LlamaRMSNorm
from ..phi4_multimodal.modeling_phi4_multimodal import simple_eager_attention_forward
from .configuration_timesfm import TimesFmConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class TimesFmOutput(BaseModelOutput):
    r"""
    loc (`torch.Tensor` of shape `(batch_size, )`):
        The mean of the time series inputs.
    scale (`torch.Tensor` of shape `(batch_size,)`):
        The scale of the time series inputs.
    """

    loc: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None


@dataclass
@auto_docstring
class TimesFmOutputForPrediction(BaseModelOutput):
    r"""
    mean_predictions (`torch.Tensor` of shape `(batch_size, sequence_length)`):
        The mean predictions of the time series.
    full_predictions (`torch.Tensor` of shape `(batch_size, sequence_length)`):
        The full predictions of the time series including the mean and the quantiles.
    loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_values` is provided):
        The loss of the TimesFM model.
    """

    mean_predictions: Optional[torch.Tensor] = None
    full_predictions: Optional[torch.Tensor] = None
    loss: Optional[Union[torch.Tensor, float]] = None


class TimesFmMLP(nn.Module):
    """Pax MLP in pytorch."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFmResidualBlock(nn.Module):
    """TimesFM residual block."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.input_layer = nn.Linear(input_dims, hidden_dims)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = self.activation(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class TimesFmRMSNorm(LlamaRMSNorm):
    pass


class TimesFmPositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()
        min_timescale = config.min_timescale
        max_timescale = config.max_timescale
        self.embedding_dims = config.hidden_size

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        self.register_buffer(
            "inv_timescales",
            min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment),
        )

    def forward(self, seq_length=None, position=None):
        """Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int defining the output sequence length.
              if the `position` argument is specified.
            position: [B, seq_length], optional position for each token in the
              sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None and seq_length is None:
            raise ValueError("Either position or seq_length must be provided")

        if position is None:
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32, device=self.inv_timescales.device).unsqueeze(0)
        elif position.ndim != 2:
            raise ValueError(f"position must be 2-dimensional, got shape {position.shape}")

        scaled_time = position.view(*position.shape, 1) * self.inv_timescales.view(1, 1, -1)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)

        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class TimesFmAttention(nn.Module):
    """Implements the attention used in TimesFM. One key difference is that there is _per_dim_scaling of the query."""

    def __init__(self, config: TimesFmConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def _scale_query(self, query: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self._scale_query(query_states)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=1.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TimesFmDecoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config: TimesFmConfig, layer_idx: int):
        super().__init__()

        self.self_attn = TimesFmAttention(config, layer_idx=layer_idx)
        self.mlp = TimesFmMLP(config)
        self.input_layernorm = TimesFmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)

        return scores, hidden_states


@auto_docstring
class TimesFmPreTrainedModel(PreTrainedModel):
    config: TimesFmConfig
    base_model_prefix = "timesfm"
    _no_split_modules = ["TimesFmDecoderLayer"]
    main_input_name = "past_values"
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, TimesFmAttention):
            # Initialize scaling parameter
            nn.init.ones_(module.scaling)


@auto_docstring
class TimesFmModel(TimesFmPreTrainedModel):
    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = TimesFmResidualBlock(
            input_dims=2 * config.patch_length,
            output_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=config.freq_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList(
            [TimesFmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if self.config.use_positional_embedding:
            self.position_emb = TimesFmPositionalEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = self._timesfm_masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(self.config.pad_val, dtype=outputs.dtype, device=outputs.device),
            outputs,
        )
        return outputs, (mu, sigma)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> TimesFmOutput:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Past values of the time series that serves as input to the model.
        past_values_padding (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The padding indicator of the time series.
        freq (`torch.LongTensor` of shape `(batch_size,)`):
            Frequency indices for the time series data.
        """
        # Reshape into patches (using view for efficiency)
        bsize = past_values.shape[0]
        patched_inputs = past_values.view(bsize, -1, self.config.patch_length)
        patched_pads = past_values_padding.view(bsize, -1, self.config.patch_length)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[0]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1])
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = self._timesfm_shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb

        # Convert paddings to attention mask and combine with causal mask
        hidden_states = model_input
        attention_mask = self._prepare_4d_attention_mask(
            attention_mask=patched_padding,
            sequence_length=hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            is_causal=True,
        )

        all_attentions = []
        all_hidden_states = []

        for layer in self.layers[: self.config.num_hidden_layers]:
            scores, hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                paddings=patched_padding,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions.append(scores)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            all_hidden_states = [model_input] + all_hidden_states
        else:
            all_hidden_states = None

        return TimesFmOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions if output_attentions else None,
            loc=stats[0],
            scale=stats[1],
        )

    @staticmethod
    def _prepare_4d_attention_mask(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
        is_causal: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Creates 4D attention mask and combines causal and padding masks if needed.

        Args:
            attention_mask: Optional tensor of shape (batch_size, seq_length) containing padding mask
            sequence_length: Length of the sequence
            dtype: Data type of the mask
            device: Device of the mask
            is_causal: Whether to apply causal masking

        Returns:
            4D attention mask of shape (batch_size, 1, seq_length, seq_length)
        """
        # Get minimum value for the dtype
        min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min

        # Handle padding mask
        if attention_mask is not None:
            # Convert 2D padding mask to 4D attention mask
            attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
            attention_mask = attention_mask * min_value

        # Create causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones((sequence_length, sequence_length), dtype=dtype, device=device) * min_value,
                diagonal=1,
            )
            causal_mask = causal_mask.view(1, 1, sequence_length, sequence_length)

            # Combine with padding mask if it exists
            if attention_mask is not None:
                attention_mask = torch.minimum(attention_mask, causal_mask)
            else:
                attention_mask = causal_mask

        return attention_mask

    @staticmethod
    def _timesfm_masked_mean_std(inputs: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates mean and standard deviation of `inputs` across axis 1.

        It excludes values where `padding` is 1.

        Args:
            inputs: A PyTorch tensor of shape [b, n, p].
            padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

        Returns:
            A tuple containing the mean and standard deviation.
            We return the statistics of the first patch with more than three non-padded values.
        """

        # Selecting the first patch with more than 3 unpadded values.
        def _get_patch_index(arr: torch.Tensor):
            indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
            row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
            return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

        pad_sum = torch.sum(1 - padding, dim=2)
        patch_indices = _get_patch_index(pad_sum)
        bidxs = torch.arange(inputs.shape[0])

        arr = inputs[bidxs, patch_indices, :]
        pad = padding[bidxs, patch_indices, :]

        # Create a mask where padding is 0
        mask = 1 - pad

        # Calculate the number of valid elements
        num_valid_elements = torch.sum(mask, dim=1)
        num_valid_elements = torch.where(
            num_valid_elements == 0,
            torch.tensor(1, dtype=num_valid_elements.dtype, device=num_valid_elements.device),
            num_valid_elements,
        )

        # Calculate the masked sum and squared sum
        masked_sum = torch.sum(arr * mask, dim=1)
        masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

        # Calculate the masked mean and standard deviation
        masked_mean = masked_sum / num_valid_elements
        masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
        masked_var = torch.where(
            masked_var < 0.0,
            torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
            masked_var,
        )
        masked_std = torch.sqrt(masked_var)

        return masked_mean, masked_std

    @staticmethod
    def _timesfm_shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Shifts rows of seq based on the first 0 in each row of the mask.

        Args:
            mask: mask tensor of shape [B, N]
            seq: seq tensor of shape [B, N, P]

        Returns:
            The shifted sequence.
        """
        batch_size, num_seq, feature_dim = seq.shape

        new_mask: torch.BoolTensor = mask == 0

        # Use argmax to find the first True value in each row
        indices = new_mask.to(torch.int32).argmax(dim=1)

        # Handle rows with all zeros
        indices[~new_mask.any(dim=1)] = -1

        # Create index ranges for each sequence in the batch
        idx_range = torch.arange(num_seq, device=seq.device).view(1, -1, 1).expand(batch_size, -1, feature_dim)

        # Calculate shifted indices for each element in each sequence
        shifted_idx = (idx_range - indices[:, None, None]) % num_seq

        # Gather values from seq using shifted indices
        shifted_seq = seq.gather(1, shifted_idx)

        return shifted_seq


class TimesFmModelForPrediction(TimesFmPreTrainedModel):
    """TimesFM model for quantile and mean prediction."""

    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.context_len = config.context_length
        self.horizon_len = config.horizon_length

        self.decoder = TimesFmModel(config)

        # quantile and mean output
        self.horizon_ff_layer = TimesFmResidualBlock(
            input_dims=config.hidden_size,
            output_dims=config.horizon_length * (1 + len(config.quantiles)),
            hidden_dims=config.intermediate_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _preprocess(
        self, inputs: Sequence[torch.Tensor], freq: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d Tensors. Each Tensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """
        input_ts, input_padding, inp_freq = [], [], []

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = torch.zeros(input_len + self.horizon_len, dtype=ts.dtype, device=ts.device)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = torch.cat([torch.zeros(num_front_pad, dtype=ts.dtype, device=ts.device), ts], dim=0)
                padding = torch.cat([torch.ones(num_front_pad, dtype=ts.dtype, device=padding.device), padding], dim=0)
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        return (
            torch.stack(input_ts, dim=0),
            torch.stack(input_padding, dim=0),
            torch.tensor(inp_freq, dtype=torch.int32).reshape(-1, 1),
        )

    def _postprocess_output(
        self, model_output: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.config.horizon_length, len(self.config.quantiles) + 1)

        mu, sigma = stats
        return output_ts * sigma[:, None, None, None] + mu[:, None, None, None]

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.config.quantiles):
            errors = targets - predictions[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor],
        freq: Optional[Sequence[Union[torch.Tensor, int]]] = None,
        window_size: Optional[int] = None,
        future_values: Optional[torch.Tensor] = None,
        forecast_context_len: Optional[int] = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TimesFmOutputForPrediction:
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Past values of the time series that serves as input to the model.
        freq (`torch.LongTensor` of shape `(batch_size,)`):
            Frequency indices for the time series data.
        window_size (`int`, *optional*):
            Window size of trend + residual decomposition. If None then we do not do decomposition.
        future_values (`torch.Tensor`, *optional*):
            Optional future time series values to be used for loss computation.
        forecast_context_len (`int`, *optional*):
            Optional max context length.
        return_forecast_on_context (`bool`, *optional*):
            True to return the forecast on the context when available, i.e. after the first input patch.
        truncate_negative (`bool`, *optional*):
            Truncate to only non-negative values if any of the contexts have non-negative values,
            otherwise do nothing.
        output_attentions (`bool`, *optional*):
            Whether to output the attentions.
        output_hidden_states (`bool`, *optional*):
            Whether to output the hidden states.

        Example:

        ```python
        >>> from transformers import TimesFmModelForPrediction

        >>> model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch")

        >>> forecast_input = [torch.linspace(0, 20, 100).sin(), torch.linspace(0, 20, 200).sin(), torch.linspace(0, 20, 400).sin()]
        >>> frequency_input = torch.tensor([0, 1, 2], dtype=torch.long)

        >>> # Generate
        >>> with torch.no_grad():
        >>>     outputs = model(past_values=forecast_input, freq=frequency_input, return_dict=True)
        >>>     point_forecast_conv = outputs.mean_predictions
        >>>     quantile_forecast_conv = outputs.full_predictions
        ```
        """
        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len

        # Get device from first input tensor
        device = past_values[0].device

        # Truncate inputs to forecast_context_len
        inputs = [ts[-fcontext_len:] for ts in past_values]
        inp_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs = []
            new_freqs = []
            for i, ts in enumerate(inputs):
                new_inputs.extend(self._timesfm_moving_average(ts, window_size))
                if freq is not None:
                    new_freqs.extend([freq[i]] * 2)
            inputs = new_inputs
            if freq is not None:
                freq = new_freqs

        if freq is None:
            logger.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq = self._preprocess(inputs, freq)
        # Move tensors to the same device as input
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        inp_freq = inp_freq.to(device)

        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []

        if input_padding.shape[1] != final_out.shape[1] + self.horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {input_padding.shape[1]} != {final_out.shape[1]} + {self.horizon_len}"
            )
        output_patch_len = self.config.horizon_length

        num_decode_patches = (self.horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = input_padding[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -fcontext_len:]
            input_padding = current_padding[:, -fcontext_len:]
            decoder_output = self.decoder(
                past_values=input_ts,
                past_values_padding=input_padding,
                freq=inp_freq,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            fprop_outputs = self._postprocess_output(
                decoder_output.last_hidden_state,
                (decoder_output.loc, decoder_output.scale),
            )

            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_length, :]
                # We have to use reshape and not view for non-contiguous memory
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_length + self.horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0 : self.horizon_len, :]

        mean_outputs = full_outputs[:, :, 0]
        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = torch.maximum(mean_outputs, 0.0)
            full_outputs = torch.maximum(full_outputs, 0.0)

        loss = None
        if future_values is not None:
            mse_loss = F.mse_loss(mean_outputs, future_values)
            quantile_loss = self._quantile_loss(full_outputs[:, :, 1:], future_values)
            loss = mse_loss + quantile_loss

        return TimesFmOutputForPrediction(
            last_hidden_state=decoder_output.last_hidden_state,
            attentions=decoder_output.attentions if output_attentions else None,
            hidden_states=decoder_output.hidden_states if output_hidden_states else None,
            mean_predictions=mean_outputs,
            full_predictions=full_outputs,
            loss=loss,
        )

    @staticmethod
    def _timesfm_moving_average(arr: torch.Tensor, window_size: int) -> list[torch.Tensor]:
        """Calculates the moving average using PyTorch's convolution function."""
        # Pad with zeros to handle initial window positions
        arr_padded = F.pad(arr, (window_size - 1, 0), "constant", 0)
        # Create a convolution kernel
        kernel = torch.ones(window_size, dtype=arr.dtype, device=arr.device) / window_size
        # Apply convolution to calculate the moving average
        smoothed_arr = F.conv1d(arr_padded.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
        return [smoothed_arr, arr - smoothed_arr]


__all__ = ["TimesFmModelForPrediction", "TimesFmPreTrainedModel", "TimesFmModel"]
