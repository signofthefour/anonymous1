from typing import Any, Optional, Union, Tuple, List


import torch

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2Model,
    QWEN2_INPUTS_DOCSTRING,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.utils.doc import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from mamba_ssm.utils.generation import InferenceParams
from transformers.utils.logging import get_logger

import sys

sys.path.append("third_party/mamba_in_llama/")
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper

logger = get_logger(__name__)


# Custom Qwen2Model that allows injecting/modifying the forward inputs
class CustomQwen2Model(Qwen2Model):
    """A thin wrapper around Qwen2Model that provides a single hook in `forward`.

    Example usages:
    - accept an extra kwarg `mamba_hidden` (tensor) and add it to `inputs_embeds` before
      calling the original forward (useful when combining Mamba outputs with the transformer).
    - ensure `position_ids` and other optional args are accepted and forwarded.
    """

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
                # TODO: fix this default inference params
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if use_cache:
            if not isinstance(inference_params, InferenceParams):
                inference_params = InferenceParams(max_batch_size=1, max_seqlen=2048)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # print("Shape of hidden_states before loop:", hidden_states.shape)

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    inference_params=inference_params,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                if isinstance(decoder_layer, MambaTransformerHybridModelWrapper):
                    inference_params = layer_outputs[
                        2 if output_attentions else 1
                    ].inference_params

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        # MambaDecoderLayer update sequence length:

        inference_params.seqlen_offset += hidden_states.size(1)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=(next_cache, inference_params),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def _find_model_attr(obj: Any) -> Optional[str]:
    """Return the attribute name on `obj` that holds the base Qwen2Model instance.

    Common names include 'qwen2', 'model', 'transformer'.
    """
    for name in ("qwen2", "model", "transformer"):
        if hasattr(obj, name):
            return name
    return None


class HybridQwen2ForCausalLM(Qwen2ForCausalLM):
    """A Qwen2ForCausalLM subclass that replaces the internal Qwen2Model with our custom one.

    Usage:
      model = HybridQwen2ForCausalLM.from_pretrained(path_or_name)

    The class attempts to be flexible about the internal attribute name used by the
    HF model container (tries 'qwen2', 'model', 'transformer').
    """

    @classmethod
    def from_existing(cls, existing: Qwen2ForCausalLM, map_location=None):
        """Create a HybridQwen2ForCausalLM initialized from an existing
        Qwen2ForCausalLM instance but using the CustomQwen2Model internally.

        Args:
            existing: an instantiated Qwen2ForCausalLM (loaded in memory)
            map_location: optional device mapping passed to load_state_dict if needed

        Returns:
            a new `HybridQwen2ForCausalLM` instance with weights copied from `existing`
            and the internal model replaced by `CustomQwen2Model`.
        """
        # Instantiate a new hybrid model from the same config
        hybrid = cls(existing.config)

        # Copy the whole state dict from the existing model where possible
        try:
            sd = existing.state_dict()
            if map_location is not None:
                # attempt to remap tensors if requested
                for k, v in sd.items():
                    sd[k] = v.to(map_location)
            hybrid.load_state_dict(sd, strict=False)
        except Exception:
            # best effort: ignore load failures
            pass

        # Replace the internal base model with our custom one and copy its weights
        attr = _find_model_attr(hybrid)
        if attr is not None:
            base_model = getattr(hybrid, attr)
            try:
                custom = CustomQwen2Model(base_model.config)
                custom.load_state_dict(base_model.state_dict(), strict=False)
                setattr(hybrid, attr, custom)
            except Exception:
                # if replacement fails, keep the hybrid as-is
                pass

        return hybrid

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        inference_params: Optional[InferenceParams] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            inference_params=inference_params,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
