#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import math
import sys
from dataclasses import dataclass, field
from typing import Optional
import pickle
import json
import random

import spacy
import ipdb
import pdb
from tqdm import tqdm
import torch
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from transformers import pipeline
from nltk.corpus import wordnet as wn

from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
# from comet import download_model, load_from_checkpoint
from bleurt import score as bleurt_score

global local_files_only, metric_type
local_files_only = False

gpus = tf.config.list_physical_devices('GPU')
if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    train_from_scratch: Optional[bool] = field(
        default=False, metadata={"help": "Train model without pre-trained initialization."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    metric_type: Optional[str] = field(
        default='bleurt', metadata={"help": "The metric to use for evaluation."}
    )
    data_type: Optional[str] = field(
        default='base', metadata={"help": "The type of input to use."}
    )
    model_type: Optional[str] = field(
        default='uniGen', metadata={"help": "The type of model to use."}
    )
    debug_model: Optional[bool] = field(
        default=False, metadata={"help": "Run model in debug mode with less data."}
    )
    test_semeval: Optional[bool] = field(
        default=False, metadata={"help": "Run model in debug mode with less data."}
    )
    filter_non_comp: Optional[bool] = field(
        default=False, metadata={"help": "Run model in debug mode with less data."}
    )
    max_external_knowledge: Optional[int] = field(
        default=200, metadata={"help": "Maximum number of summary words to use"}
    )
    mode: Optional[str] = field(
        default='', metadata={"help": "news/empty"}
    )
    train_split: Optional[float] = field(
        default=1, metadata={"help": "Fraction of training data to use"}
    )

    two_stage: Optional[str] = field(
        default=None, metadata={"help": "Path to first stage none_classifier predictions."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='input',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default='output',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default='data/train.jsonl', metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default='data/dev.jsonl',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default='data/test.jsonl',
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    output_prediction_file: Optional[str] = field(
        default=None,
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
            # if self.train_file is not None:
            #     extension = self.train_file.split(".")[-1]
                # assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            # if self.validation_file is not None:
            #     extension = self.validation_file.split(".")[-1]
            #     assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

def remove_cnp(string):
    return ' '.join(string.split()[2:])

def get_cnp(string):
    words = string.split()
    return words[0], words[1]

def list_replace(lst, value1, value2):
    # replace value1 with value2 in the list
    return [value2 if x==value1 else x for x in lst]

def compute_metrics_helper(decoded_preds, decoded_labels, none_preds, test):
    global metric_type

    decoded_labels = [d.split(';') for d in decoded_labels]
    decoded_labels = [[' none '] if d[0].strip() == '' else d for d in decoded_labels]
    decoded_preds = [' none ' if d.strip() == '' else d for d in decoded_preds]
    # decoded_labels = [['NONE'] if 'None' in d[0] else d for d in decoded_labels]

    check_template = False
    if check_template:
        templates = []
        label_absent = 0
        for l in decoded_labels:
            l = l[0]
            if l.strip() == 'none':
                template = ' is None of '
                label_absent += 1
            else:
                compound = l.split()[:2]    
                template = f'{compound[0]} {compound[1]} is None of {compound[0]}'
            templates.append(template)
        decoded_preds = templates

    none_gold = np.array([1 if ' none ' in d[0].lower() else 0 for d in decoded_labels])
    if none_preds is not None:
        none_preds = np.array(none_preds)     
        decoded_preds_corr = []
        for n, l, p in zip(none_preds, decoded_labels, decoded_preds):
            if n == 1 and l[0].strip() != 'none':
                nnp, nn = p.split()[:2]
                decoded_preds_corr.append(f"{nnp} {nn} is None of {nn}")
            else:
                decoded_preds_corr.append(p)
        decoded_preds = decoded_preds_corr
    else:
        none_preds = np.array([1 if ' none ' in d.lower() else 0 for d in decoded_preds])

    result = {'nli_all_scores':[], 'bleu_all_scores': [], 'bleurt_all_scores': []}
    if test or metric_type == 'bleurt': 
        torch.cuda.empty_cache()

        # checkpoint = 'models/eval_metrics/BLEURT-20'
        checkpoint = 'models/eval_metrics/bleurt-base-128-fine1/export/bleurt_best/1643440561'
        # checkpoint = 'models/eval_metrics/bleurt-base-128/'
        # checkpoint = 'models/eval_metrics/bleurt-large-128/'
        # checkpoint = 'models/eval_metrics/bleurt-large-128-fine1/export/bleurt_best/1643438395'

        ## Using huggingface metrics
        # metric = load_metric('bleurt',  'bleurt-20') #, local_files_only=True) 

        scorer = bleurt_score.BleurtScorer(checkpoint)
        metric_batch_size = 128
        metric_num_batches = int(math.ceil(len(decoded_labels) / metric_batch_size))
        for k in tqdm(range(metric_num_batches)):
            scores = scorer.score(references=[l[0] for l in decoded_labels[k*metric_batch_size:(k+1)*metric_batch_size]], 
            candidates=decoded_preds[k*metric_batch_size:(k+1)*metric_batch_size])
            ## Using huggingface metrics
            # result_batch = metric.compute(predictions=decoded_preds[k*metric_batch_size:(k+1)*metric_batch_size], \
                # references=decoded_labels[k*metric_batch_size:(k+1)*metric_batch_size])
            result['bleurt_all_scores'].extend(scores)    

        # if none_preds is not None:
        org_bleurt_all_scores = np.array(result['bleurt_all_scores'])
        result['bleurt_all_scores'] = (1-none_gold)*(1-none_preds)*np.array(result["bleurt_all_scores"])+none_preds*none_gold            
        bleurt_cmp = ((1-none_gold)*(1-none_preds)*np.array(result["bleurt_all_scores"])).sum() / ((1-none_gold)*(1-none_preds)).sum()
        # bleurt_cmp = ((1-none_gold)*org_bleurt_all_scores).sum() / (1-none_gold).sum()
        bleurt_noncmp = (none_preds*none_gold).sum() / none_gold.sum() 
        result['scores'] = result['bleurt_scores'] = [np.mean(result['bleurt_all_scores']), bleurt_cmp, bleurt_noncmp]

    if test or metric_type == 'nli':
        ## None metric
        result['scores'] = result['nli_scores'] = [(none_preds == none_gold).mean(), precision_score(none_gold, none_preds), recall_score(none_gold, none_preds)]
        result['nli_all_scores'] = (none_preds == none_gold)

        ## COMET metric
        # model_path = download_model("wmt20-comet-da")
        # model = load_from_checkpoint(model_path)
        # data = [{'mt': p, 'ref': l} for l, p in zip(decoded_labels, decoded_preds)]
        # seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)

        ## BERTScore metric
        # _,_,F1 = bert_score.score(decoded_preds, [l[0] for l in decoded_labels], model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)        
        # result['nli_all_scores'] = F1.tolist()
        # if none_preds is not None:
        #     result['nli_all_scores'] = (1-none_gold)*(1-none_preds)*np.array(result["nli_all_scores"])+none_preds*none_gold
        # result['scores'] = result['nli_scores']  = np.mean(result['nli_all_scores'])

        ## NLI metric
        # classifier = pipeline("zero-shot-classification", model="models/bart-large-mnli", device=0, local_files_only=True)
        # for k in tqdm(range(len(decoded_preds))):
        #     pred = decoded_preds[k].split('\t')[0] # remove score at end, if present
        #     pred = remove_cnp(pred)
        #     if pred == '':
        #         pred = 'NA'
        #     score1 = []
        #     for gold in decoded_labels[k]:
        #         gold = remove_cnp(gold)
        #         if gold == '':
        #             continue
        #         pred_gold = classifier(pred, [gold])['scores'][0]
        #         gold_pred = classifier(gold, [pred])['scores'][0]
        #         score1.append((pred_gold+gold_pred)/2)
        #     result['nli_all_scores'].append(max(score1))
        # result['scores'] = result['nli_scores'] = np.mean(result['nli_all_scores'])

    if test or metric_type == 'sacrebleu':
        metric = load_metric('sacrebleu', local_files_only=True)
        # SacreBLEU requires same number of references per prediction        
        for k in range(len(decoded_preds)):
            decoded_labels[k] = [decoded_labels[k][0]]
            decoded_preds[k] = decoded_preds[k]
            result['bleu_all_scores'].append(metric.compute(predictions=[decoded_preds[k]], references=[decoded_labels[k]])['score']/100)

        org_bleu_all_scores = np.array(result['bleu_all_scores'])
        # if none_preds is not None:
        result["bleu_all_scores"] = (1-none_gold)*(1-none_preds)*np.array(result["bleu_all_scores"])+none_preds*none_gold
        bleu_cmp = ((1-none_gold)*(1-none_preds)*np.array(result["bleu_all_scores"])).sum() / ((1-none_gold)*(1-none_preds)).sum()
        # bleu_cmp = ((1-none_gold)*org_bleu_all_scores).sum() / (1-none_gold).sum()
        bleu_noncmp = (none_preds*none_gold).sum() / none_gold.sum() 
        result['scores'] = result['bleu_scores'] = [np.mean(result["bleu_all_scores"]), bleu_cmp, bleu_noncmp]

        ## Corpus level metric differs from average of all sentence comparisons
        # result['scores'] = result['bleu_scores'] = metric.compute(predictions=[p for p in decoded_preds], 
                                        # references=[[l[0]]for l in decoded_labels])['score']/100

    if test:
        result['nli_all_scores'] = ','.join([str(f) for f in result['nli_all_scores']])
        result['bleu_all_scores'] = ','.join([str(f) for f in result['bleu_all_scores']])
        result['bleurt_all_scores'] = ','.join([str(f) for f in result['bleurt_all_scores']])
    else:
        del result['nli_all_scores']
        del result['bleu_all_scores']
        del result['bleurt_all_scores']

    return result

def main():
    global metric_type, metric
    global local_files_only
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        if '--do_train' in sys.argv:
            sys.argv.append('--load_best_model_at_end')
            sys.argv.extend(['--evaluation_strategy', 'steps'])
            sys.argv.extend(['--eval_steps', '500'])
            sys.argv.extend(['--save_strategy', 'steps'])
            sys.argv.extend(['--save_total_limit', '1'])
        sys.argv.extend(['--per_device_eval_batch_size', '32'])
        sys.argv.extend(['--metric_for_best_model', 'eval_scores'])
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # default learning rate: 5e-5
    # training_args.output_dir = os.path.join(data_args.data_type, training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_args.data_type = data_args.data_type.split('_')
    if 'rand' in data_args.data_type:
        data_args.train_file = 'data/train_rand.jsonl'
        data_args.validation_file = 'data/dev_rand.jsonl'
        data_args.test_file = 'data/test_rand.jsonl'
    elif 'integrate' in data_args.data_type:
        data_args.train_file = 'data/train_integrate.jsonl'
        data_args.validation_file = 'data/dev_integrate.jsonl'

    if data_args.test_semeval:
        data_args.test_file = 'data/test_semeval_1int.jsonl'

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            # extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            # extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            # extension = data_args.test_file.split(".")[-1]        

        raw_datasets = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
        if 'integrate' in data_args.data_type:
            raw_datasets = raw_datasets
        else:
            raw_datasets = raw_datasets.map(lambda example: {'input': example['nnp']+' '+example['nn'], 'output': example['explicit_relation']})                

        if 'ner' in data_args.data_type:
            nlp = spacy.load('en_core_web_sm')
            def get_ner(sentence, word):
                doc = nlp(sentence)
                if len(doc.ents) > 0:
                    for ent in doc.ents:
                        if ent.text == word:
                            return word+' belongs to '+spacy.explain(ent.label_)
                return word+' belongs to NONE'

            raw_datasets = raw_datasets.map(lambda example: {'sentence': example['nnp']+' '+example['nn'] if not example['sentence'] else example['sentence']})            
            raw_datasets = raw_datasets.map(lambda example: {'input': example['input']+' [SEP] '+get_ner(example['sentence'], example['nnp'])},
                                            load_from_cache_file=False)        

        if 'sentence' in data_args.data_type:
            raw_datasets = raw_datasets.map(lambda example: {'input': example['sentence'][:200]+' [SEP] '+example['input']})

        def get_desc(word):
            if word in summariesD and summariesD[word][0].strip() != '':
                return word+' meaning: '+summariesD[word][0][:data_args.max_external_knowledge]
            else:
                synsets = wn.synsets(word)
                if len(synsets) > 0:
                    return word+' meaning: '+synsets[0].definition()[:data_args.max_external_knowledge]
                else:
                    return word+' meaning: '

        if 'knowledge' in data_args.data_type:
            summariesD = pickle.load(open('data/summaries.pkl','rb'))        
            if 'nnp' in data_args.data_type:
                raw_datasets = raw_datasets.map(lambda example: {'input': get_desc(example['nnp'])+' [SEP] '+example['input']}, load_from_cache_file=False)
            if 'nn' in data_args.data_type:
                raw_datasets = raw_datasets.map(lambda example: {'input': get_desc(example['nn'])+' [SEP] '+example['input']}, load_from_cache_file=False)    

        if 'shufflennp' in data_args.data_type:
            raw_datasets = raw_datasets.map(lambda example: {'shuff_nnp': ''.join(sorted(example['nnp'], key=lambda k: random.random()))})
            raw_datasets = raw_datasets.map(lambda example: {'input': example['input'].replace(example['nnp'], example['shuff_nnp']), 
                'output': example['output'].replace(example['nnp'], example['shuff_nnp']), 'nnp': example['shuff_nnp']})
        elif 'shufflenn' in data_args.data_type:
            raw_datasets = raw_datasets.map(lambda example: {'shuff_nn': ''.join(sorted(example['nn'], key=lambda k: random.random()))})
            raw_datasets = raw_datasets.map(lambda example: {'input': example['input'].replace(example['nn'], example['shuff_nn']), 
                'output': example['output'].replace(example['nn'], example['shuff_nn']), 'nn': example['shuff_nn']})
        
        ## Train Split
        if data_args.train_split != 1:
            num_train_examples = int(len(raw_datasets['train'])*data_args.train_split)
            raw_datasets['train'] = raw_datasets['train'].filter(lambda example, index: index < num_train_examples, with_indices=True)

        if data_args.debug_model:
            raw_datasets = raw_datasets.filter(lambda example, index: index < 100, with_indices=True)

        if 'integrate' not in data_args.data_type:
            raw_datasets['train'] = raw_datasets['train'].map(lambda example: {'output': ' '.join(example['output'].split()[2:])})

        if data_args.model_type == 'clsGen':
            raw_datasets = raw_datasets.map(lambda example: {'output': '[NONE] '+example['output'] if example['explicit_relation']!='' else '[NOTNONE] '+example['output']})
        elif data_args.model_type == 'uniGen':
            if 'integrate' in data_args.data_type:
                raw_datasets = raw_datasets
            else:
                raw_datasets = raw_datasets.map(lambda example: {'output': example['output'] if example['explicit_relation']!='' else example['nnp']+' '+example['nn']+' is None of '+example['nnp']})


    if not training_args.do_train:
        model_args.model_name_or_path = training_args.output_dir

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=local_files_only
    )

    tokenizer.add_special_tokens({'additional_special_tokens':["[NONE]","[NOTNONE]"]})

    if model_args.train_from_scratch:
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            local_files_only=local_files_only
        )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric_type = data_args.metric_type    

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics_val(eval_preds):
        preds, labels = eval_preds
        none_preds = None
        if isinstance(preds, tuple):
            preds, none_preds = preds[0], preds[1]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        decoded_preds_comp = []
        for p, l in zip(decoded_preds, decoded_labels):
            compound = ' '.join(l.split()[:2])
            if not p.startswith(compound):
                p = compound+' '+p
            decoded_preds_comp.append(p)

        decoded_preds = decoded_preds_comp
        return compute_metrics_helper(decoded_preds, decoded_labels, none_preds, test=False)

    def compute_metrics_test(eval_preds):
        preds, labels = eval_preds
        
        none_preds = None
        if isinstance(preds, tuple):
            preds, none_preds = preds[0], preds[1]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        decoded_preds_comp = []
        for p, l in zip(decoded_preds, decoded_labels):
            compound = ' '.join(l.split()[:2])
            if not p.startswith(compound):
                p = compound+' '+p
            decoded_preds_comp.append(p)

        decoded_preds = decoded_preds_comp
        return compute_metrics_helper(decoded_preds, decoded_labels, none_preds, test=True)


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_val,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if data_args.mode == 'news' or 'integrate' in data_args.data_type:
            trainer.compute_metrics = None
            predict_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )
        else:
            trainer.compute_metrics = compute_metrics_test
            predict_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )
            metrics = predict_results.metrics
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            # trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
            print('Performance = ', metrics['predict_scores'])

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                if not data_args.output_prediction_file:
                    predictions_fp = data_args.test_file.split('/')[-1]+'_predictions.txt.'+metric_type
                    data_args.output_prediction_file = os.path.join(training_args.output_dir, predictions_fp)

                with open(data_args.output_prediction_file, "w") as writer:
                    if data_args.test_semeval:
                        for example, pred in zip(raw_datasets['test'], predictions):
                            writer.write(f"{example['nnp']}\t{example['nn']}\t{' '.join(pred.split()[2:])}\t1\n")
                        print('BLEU, BLEURT, NLI: ', metrics['predict_bleu_scores'], metrics['predict_bleurt_scores'], metrics['predict_nli_scores'])
                        print('Predictions written to : ', writer.file)
                    elif 'integrate' in data_args.data_type:
                        for example, pred in zip(raw_datasets['test'], predictions):
                            writer.write(json.dumps({'input': example['input'], 'output': pred, 'initial_input': example['output']})+'\n')
                    elif data_args.mode == 'news':
                        for example, pred in zip(raw_datasets['test'], predictions):
                            # writer.write(f"{example['sentence']}\t{example['input']}\t{pred}\n")
                            if not pred.startswith(example['input']):
                                pred = example['input']+' '+pred
                            writer.write(json.dumps({'input': example['sentence']+' [SEP] '+pred, 'output': pred})+'\n')
                    elif 'merge' in data_args.data_type:
                        for example, p, bleu in zip(raw_datasets['test'], predictions, metrics['predict_bleurt_all_scores']):
                            writer.write(f"{example['input']}\t{p}\t{bleu}\n")
                    else:
                        writer.write("Input\tGold\tPrediction\tBLEURT\tNLI\tBLEU\n")
                        for example, p, bleurt, nli, bleu in zip(raw_datasets['test'], predictions, 
                            metrics['predict_bleurt_all_scores'].split(','), 
                            metrics['predict_nli_all_scores'].split(','), 
                            metrics['predict_bleu_all_scores'].split(',')):
                            if 'integrate' in data_args.data_type:
                                writer.write(f"{example['input']}\t{p}\t{bleurt}\t{nli}\t{bleu}\n")
                            else:
                                writer.write(f"{example['input']}\t{example['output']}\t{example['nnp']} {example['nn']} {p}\t{bleurt}\t{nli}\t{bleu}\n")
                        writer.write(f"Average:\tBLEURT:{metrics['predict_bleurt_scores']}\tNLI:{metrics['predict_nli_scores']}\tBLEU:{metrics['predict_bleu_scores']}\n")

                        print('BLEU, BLEURT, NLI: ', metrics['predict_bleu_scores'], metrics['predict_bleurt_scores'], metrics['predict_nli_scores'])

                        gold_labels_file = os.path.join(training_args.output_dir, "gold_labels.txt")
                        with open(gold_labels_file, "w") as writer:
                            for example in raw_datasets['test']:
                                writer.write(example['output']+"\n")


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
