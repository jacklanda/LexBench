# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model. """

from transformers.file_utils import add_start_docstrings
from transformers.utils import logging

from transformers import XLMRobertaConfig
from model.modeling_g2groberta import RobertaGraphModel, RobertaGraphConfig
from opt_einsum import contract
from transformers import XLMRobertaModel, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST

logger = logging.get_logger(__name__)


def initialize_xlm_robertagraph(BERT_NAME_OR_PATH, config_graph=None):
    xlm_robertaconfig = XLMRobertaGraphConfig.from_pretrained(BERT_NAME_OR_PATH)
    init_bert = XLMRobertaModel.from_pretrained(BERT_NAME_OR_PATH)
    xlm_robertaconfig.add_graph_par(config_graph)
    model = XLMRobertaGraphModel(xlm_robertaconfig)
    model.load_state_dict(init_bert.state_dict(), strict=False)
    return model


class XLMRobertaGraphConfig(RobertaGraphConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = "xlm-roberta"


class XLMRobertaGraphModel(RobertaGraphModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaGraphConfig
