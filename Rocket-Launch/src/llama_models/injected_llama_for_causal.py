import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import LlamaPreTrainedModel
from typing import List, Optional, Tuple, Union

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from llama_models.injected_llama_model import InjectedLlamaModel
from pytorch_lightning import LightningModule


from transformers import ( 
    LlamaConfig as HFConfig
)

_CONFIG_FOR_DOC = "LlamaConfig"

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class LlamaForCausalLM(LlamaPreTrainedModel, LightningModule):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, tokenizer, irm_config):
        # The HF config
        hf_config = HFConfig(**irm_config.model_config)

        # Call init functions for both super classes
        LlamaPreTrainedModel.__init__(self, hf_config)
        LightningModule.__init__(self)

        # This is our config, not a HF Config
        self.irm_config = irm_config

        self.hf_config = hf_config

        # The Llama should have a reference to the tokenizer so it can save output during validation step.
        self.tokenizer = tokenizer
        
        self.model = InjectedLlamaModel(self.irm_config, self.hf_config)
        self.vocab_size = self.hf_config.vocab_size
        self.lm_head = nn.Linear(self.hf_config.hidden_size, self.hf_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.hf_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.hf_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

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
        )

        hidden_states = outputs[0]
        if self.hf_config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.hf_config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.hf_config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.hf_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if getattr(self.model.layers[0].self_attn, "past_key_value", None) is not None:
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def training_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, 
                            attention_mask=x_mask)

        loss = output.loss

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, 
                            attention_mask=x_mask)
        
        val_loss = output.loss
        y_hat = output.logits

        if self.irm_config.save_predictions_during_training:
            # Decode predictions and add to valuation predictions list
            probs = torch.softmax(y_hat, dim=2)
            preds = torch.argmax(probs, 2).detach().cpu().tolist()

            #y_true_decoded = self.tokenizer.decode(y_true[0].tolist())
            decoded = self.tokenizer.decode(preds[0])

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(val_loss)
        self.log('val_loss', 
                 val_loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
        
        self.log('val_perplexity', 
                 perplexity, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
            
        return val_loss
    
    def on_validation_epoch_end(self) -> None:
        if self.irm_config.save_predictions_during_training == True:
            dir_path = Path(self.irm_config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a', encoding="utf-8") as f:
                for item in self.validation_step_outputs:
                    f.write(str(self.current_epoch) + ': ')
                    f.write(str(item) + '\n')

            self.validation_step_outputs.clear()
    
    def on_test_start(self,):
        # Create data structures to store predictions
        self.y_trues = []
        self.y_hats = []

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Log/save any metrics you want to test here.
        """
        x, x_mask, y_true = batch

        output_ids = self.model.generate(input_ids=x, 
                                    attention_mask=x_mask,
                                    num_beams=5,
                                    min_length=0,
                                    max_new_tokens=self.irm_config.max_gen_len)
        
        self.y_trues += self.tokenizer.batch_decode(y_true.tolist())
        self.y_hats += self.tokenizer.batch_decode(output_ids.tolist())
    
    def on_test_epoch_end(self):
        """
        Configure any metrics/output you want to save at the end of testing here.
        """
        # Save predictions
        dir_path = Path(self.irm_config.default_root_dir)
        targets_path = dir_path / 'test_targets.txt'
        predictions_path = dir_path / 'test_predictions.txt'

        # Check if the directory exists. If not, create it
        dir_path.mkdir(parents=True, exist_ok=True)

        # Check if the file exists. If not, create it and append the outputs
        with targets_path.open('a', encoding="utf-8") as f:
            for item in self.y_trues:
                f.write(item + '\n')

        with predictions_path.open('a', encoding="utf-8") as f:
            for item in self.y_hats:
                f.write(item + '\n')
                    
        # Get chrf score
        chrf = corpus_chrf(self.y_trues, self.y_hats)

        # Get bleu score
        bleu = corpus_bleu([[tgt] for tgt in self.y_trues], self.y_hats)

        self.log('chrf', 
                 chrf, 
                 logger=True, 
                 sync_dist=True)
        self.log('bleu', 
                 bleu,
                 logger=True, 
                 sync_dist=True)

        scores = ['chrf: ' + str(chrf), 'bleu: ' + str(bleu)]

        print('Final scores: ', scores)
    
    def configure_optimizers(self):
        params = self.model.irm.parameters()
        optimizer = torch.optim.Adam(params, lr=self.irm_config.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.irm_config.gamma)
        return [optimizer], [lr_scheduler]

