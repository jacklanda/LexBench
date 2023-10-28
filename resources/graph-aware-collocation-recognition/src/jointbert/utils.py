import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, XLMConfig
from transformers import XLMRobertaConfig, RobertaConfig
from transformers import BertTokenizer, XLMRobertaTokenizer, RobertaTokenizer
from transformers import CamembertConfig, CamembertTokenizer

from model import JointXLMRoberta, JointRoberta, JointBERT

MODEL_CLASSES = {
    'bert-base-uncased': (BertConfig, JointBERT, BertTokenizer),
    'bert-base-cased': (BertConfig, JointBERT, BertTokenizer),
    'bert-large-uncased': (BertConfig, JointBERT, BertTokenizer),
    'bert-large-cased': (BertConfig, JointBERT, BertTokenizer),
    'roberta-base': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'roberta-large': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'roberta-large-mnli': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'xlm-mlm-xnli15-1024': (XLMConfig, JointXLMRoberta, XLMRobertaTokenizer),
    'xlm-roberta-base': (XLMRobertaConfig, JointXLMRoberta, XLMRobertaTokenizer),
    'xlm-roberta-large': (XLMRobertaConfig, JointXLMRoberta, XLMRobertaTokenizer),
    'roberta-base-bne': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'roberta-large-bne': (RobertaConfig, JointRoberta, RobertaTokenizer),
    'camembert-base': (RobertaConfig, JointRoberta, CamembertTokenizer)
}

MODEL_PATH_MAP = {
    'bert-base-uncased': 'bert-base-uncased',
    'bert-base-cased': 'bert-base-cased',
    'bert-large-uncased': 'bert-large-uncased',
    'bert-large-cased': 'bert-large-cased',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
    'roberta-large-mnli': 'roberta-large-mnli',
    'xlm-mlm-xnli15-1024': 'xlm-mlm-xnli15-1024',  # multilingual (15 XNLI languages)
    'xlm-roberta-base': 'xlm-roberta-base',  # multilingual
    'xlm-roberta-large': 'xlm-roberta-large',  # multilingual
    'roberta-base-bne': 'PlanTL-GOB-ES/roberta-base-bne',  # spanish
    'roberta-large-bne': 'PlanTL-GOB-ES/roberta-large-bne',  # spanish
    'camembert-base': 'camembert-base'  # french
}


def get_intent_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


# currently "btok" id is 1
def adjust_syndep_heads(base_vectors, tensor_heads, tensor_types, rearrange_ids, pos_labels):
    device = base_vectors.device

    tensor_heads = tensor_heads.to(device)
    tensor_types = tensor_types.to(device)
    pos_labels = pos_labels.to(device)

    new_heads = torch.gather(rearrange_ids, 1, tensor_heads)

    pred_arcs = []
    pred_rels = []
    pred_pos = []
    for i in range(base_vectors.size()[0]):
        assert int(torch.max(base_vectors[i])) < len(new_heads[i])

        pred_arcs.append(new_heads[i][base_vectors[i]])
        pred_rels.append(tensor_types[i][base_vectors[i]])
        pred_pos.append(pos_labels[i][base_vectors[i]])

    pred_arcs = torch.stack(pred_arcs).to(device)
    pred_rels = torch.stack(pred_rels).to(device)
    pred_pos = torch.stack(pred_pos).to(device)

    return pred_pos, pred_arcs, pred_rels


def build_graph_labeled_syn(pred_arcs, pred_rel, mask, num_labels):
    graph_arc = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[1]).long().to(mask.device)
    mask = mask.long()

    # mask = stop_sign.unsqueeze(1) * mask

    lengths = mask.sum(dim=1)

    for i, (arc, rel, lens, mask_instance) in enumerate(zip(pred_arcs, pred_rel, lengths, mask)):
        if lens != 0:
            graph_arc[i, torch.arange(mask.shape[1]), arc] = rel

            graph_arc[i, :, :] = graph_arc[i, :, :] * mask_instance.unsqueeze(1)
            assert not len(graph_arc[i, :, :][graph_arc[i, :, :] < 0]) and not len(
                graph_arc[i, :, :][graph_arc[i, :, :] > num_labels])

            mask_t = (graph_arc[i, :, :] > 0) * 1
            graph_arc_t = graph_arc[i, :, :].transpose(0, 1) + mask_t.transpose(0, 1) * num_labels
            graph_arc[i, :, :] = graph_arc[i, :, :] + graph_arc_t

            assert not len(graph_arc[i, :, :][graph_arc[i, :, :] > 2 * num_labels])

    return graph_arc


def build_syndep_graph(input_mask, rearrange_ids, base_vectors, heads, types, pos_labels, num_labels):
    vector_pos, vector_heads, vector_rels = adjust_syndep_heads(base_vectors, heads, types, rearrange_ids, pos_labels)

    graph_arc = build_graph_labeled_syn(vector_heads, vector_rels, input_mask, num_labels)

    return vector_pos, graph_arc
