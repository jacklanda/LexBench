# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
"""
 PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
 part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from opt_einsum import contract

from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

from transformers.models.distilbert.configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, \
    DistilBertConfig

from transformers.models.distilbert.modeling_distilbert import _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC, \
    _TOKENIZER_FOR_DOC, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST, create_sinusoidal_embeddings, \
    FFN, DISTILBERT_START_DOCSTRING, DISTILBERT_INPUTS_DOCSTRING, DistilBertPreTrainedModel, DistilBertModel

logger = logging.get_logger(__name__)


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #

def initialize_distilbertgraph(DISTILBERT_NAME_OR_PATH, config_graph=None):
    bertgconfig = GraphDistilBertConfig.from_pretrained(DISTILBERT_NAME_OR_PATH)
    init_bert = DistilBertModel.from_pretrained(DISTILBERT_NAME_OR_PATH)
    bertgconfig.add_graph_par(config_graph)
    model = GraphDistilBertModel(bertgconfig)
    model.load_state_dict(init_bert.state_dict(), strict=False)
    return model


class GraphDistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DistilBertModel` or a
    :class:`~transformers.TFDistilBertModel`. It is used to instantiate a DistilBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DistilBERT `distilbert-base-uncased
    <https://huggingface.co/distilbert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DistilBertModel` or
            :class:`~transformers.TFDistilBertModel`.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (:obj:`int`, `optional`, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilities used in the question answering model
            :class:`~transformers.DistilBertForQuestionAnswering`.
        seq_classif_dropout (:obj:`float`, `optional`, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            :class:`~transformers.DistilBertForSequenceClassification`.

    Examples::

        >>> from transformers import DistilBertModel, DistilBertConfig

        >>> # Initializing a DistilBERT configuration
        >>> configuration = DistilBertConfig()

        >>> # Initializing a model from the configuration
        >>> model = DistilBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "distilbert"

    def __init__(
            self,
            vocab_size=30522,
            max_position_embeddings=512,
            sinusoidal_pos_embds=False,
            n_layers=6,
            n_heads=12,
            dim=768,
            hidden_dim=4 * 768,
            dropout=0.1,
            attention_dropout=0.1,
            activation="gelu",
            initializer_range=0.02,
            qa_dropout=0.1,
            seq_classif_dropout=0.2,
            pad_token_id=0,
            **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout

    def add_graph_par(self, config_graph=None):
        # dep syn
        self.config_graph = config_graph

    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


class GraphEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.pos_embeddings = nn.Embedding(config.config_graph['pos_size'], config.dim)

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, pos_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        pos_embeddings = self.pos_embeddings(pos_ids)
        word_embeddings = word_embeddings + pos_embeddings

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class GraphMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.attention_head_size = self.dim // self.n_heads
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

        ############## g2g part
        self.config_graph = config.config_graph
        self.graph_exist = self.config_graph is not None
        if self.graph_exist:
            self.use_two_attn = self.config_graph['use_two_attn']
            self.just_attn = self.config_graph['just_attn']
            self.dp_relation_k_g = nn.Embedding(2 * self.config_graph["label_size"] + 1, self.attention_head_size)
            if self.config_graph['use_two_attn']:
                self.dp_relation_q_g = nn.Embedding(2 * self.config_graph["label_size"] + 1, self.attention_head_size)
            if not self.config_graph['just_attn']:
                self.dp_relation_v_g = nn.Embedding(2 * self.config_graph["label_size"] + 1, self.attention_head_size)

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def relative_matmul_dpv(self, x, z):
        """ Helper function for dependency parsing relations"""

        x = x.transpose(1, 2)
        out = torch.matmul(x, z)
        out = out.transpose(1, 2)

        return out

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False, graph_arc=None):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        ## convert matrix ids to matrix of embeddings
        if graph_arc is not None:
            dp_keys_g = self.dp_relation_k_g(graph_arc.to(k.device))
            if self.use_two_attn:
                dp_query_g = self.dp_relation_q_g(graph_arc.to(k.device))
            if not self.just_attn:
                dp_values_g = self.dp_relation_v_g(graph_arc.to(k.device))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if graph_arc is not None:
            # attention_scores = attention_scores + self.relative_matmul_dp(query_layer, dp_keys_g1)
            key_dp = contract('bhse,bsre->bhsr', q, dp_keys_g)
            if self.use_two_attn:
                query_dp = contract('bhse,bsre->bhsr', k, dp_query_g)
            else:
                query_dp = contract('bhse,bsre->bhsr', k, dp_keys_g)
            scores = scores + query_dp + key_dp

        scores = scores / math.sqrt(dim_per_head)

        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        if graph_arc is not None and not self.just_attn:
            context = context + self.relative_matmul_dpv(weights, dp_values_g)

        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class GraphTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = GraphMultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False, graph_arc=None):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            graph_arc=graph_arc,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = GraphTransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(
            self, x, attn_mask=None, head_mask=None, graph_arc=None, output_attentions=False,
            output_hidden_states=False, return_dict=None
    ):  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions,
                graph_arc=graph_arc
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class GraphDistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = GraphEmbeddings(config)  # Embeddings
        self.transformer = GraphTransformer(config)  # Encoder

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            pos_ids=None,
            graph_arc=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, pos_ids)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            graph_arc=graph_arc,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
