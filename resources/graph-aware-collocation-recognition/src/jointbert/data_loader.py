import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset
import ujson
import numpy as np
from utils import get_intent_labels, get_slot_labels
import pickle

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, poss, tags, lemma, head_dep, rel_dep, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.poss = poss
        self.tags = tags
        self.lemma = lemma
        self.head_dep = head_dep
        self.rel_dep = rel_dep
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, rearange_ids, base_vectors, word_start_mask, word_end_mask,
                 token_type_ids, heads_dep, rels_dep, pos_labs, intent_label_id=None, slot_labels_ids=None):
        self.input_ids = input_ids
        self.rearrange_ids = rearange_ids
        self.word_start_mask = word_start_mask
        self.word_end_mask = word_end_mask
        self.base_vectors = base_vectors
        self.heads_dep = heads_dep
        self.rels_dep = rels_dep
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.pos_labs = pos_labs
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'
        self.udinfo_file = "parsed.in.json"

        self.dep_label = {"PAD": 0, "UNK": 1}
        self.pos_label = {"PAD": 0, "UNK": 1}

    # new function for reading parsing info
    def _read_ud(cls, input_file):
        """Reads JSON dictionary from Spacy UDpipe"""
        # output = []
        output = ujson.load(open(input_file, "r"))
        # for sent_id in dict_file.keys():
        #    output.append(dict_file[sent_id]['ud_tree_tokens'])
        return output

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    # new function for g2g
    def _create_examples(self, texts, udinfos, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, udinfo, intent, slot) in enumerate(zip(texts, udinfos, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            word_ud = []
            poss = []
            tags = []
            lemmas = []
            head_deps = []
            rel_deps = []
            for y in udinfo:
                word_ud.append(y['text'])
                poss.append(y['pos'])
                tags.append(y['tag'])
                lemmas.append(y['lemma'])
                head_deps.append(y['head_i'])
                rel_deps.append(y['dep'])

                if set_type == 'train':
                    if y['dep'] not in self.dep_label:
                        self.dep_label[y['dep']] = len(self.dep_label)

                if set_type == 'train':
                    if y['pos'] not in self.pos_label:
                        self.pos_label[y['pos']] = len(self.pos_label)

            #### UD file and initial file are not the same
            assert words == word_ud

            assert len(poss) == len(lemmas) == len(tags) == len(head_deps) == len(rel_deps)

            # 2. intent
            intent_label = self.intent_labels.index(
                intent) if intent in self.intent_labels else self.intent_labels.index("UNK")

            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=word_ud, poss=poss, tags=tags, lemma=lemmas,
                                         head_dep=head_deps, rel_dep=rel_deps, intent_label=intent_label,
                                         slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     udinfos=self._read_ud(os.path.join(data_path, self.udinfo_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    'dataset_for_jointbert_with_parsed_trees_extended': JointProcessor,
    'dataset_for_jointbert_with_parsed_trees_es': JointProcessor,
    'dataset_for_jointbert_with_parsed_trees_fr': JointProcessor,
    'dataset_for_jointbert_with_parsed_trees_en_es_shuffled': JointProcessor,
    'dataset_for_jointbert_with_parsed_trees_en_fr_shuffled': JointProcessor
}


def token_span(sent, tokenizer, slot_labels, pad_token_label_id):
    tokens = []
    word_start_mask = []
    word_end_mask = []
    idx = 0
    word_start_mask.append(1)
    word_end_mask.append(1)
    cleaned_words = sent

    slot_labels_ids = []
    assert len(cleaned_words) == len(slot_labels)

    for word, slot_label in zip(cleaned_words, slot_labels):
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]

        slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        for _ in range(len(word_tokens)):
            word_start_mask.append(0)
            word_end_mask.append(0)
            idx += 1
        word_start_mask[len(tokens) + 1] = 1  # since cls in the first token
        word_end_mask[-1] = 1
        tokens.extend(word_tokens)

    word_start_mask.append(1)
    word_end_mask.append(1)

    token_length = len(tokens)

    return tokens, word_start_mask, word_end_mask, token_length, slot_labels_ids


def truncate_word_bert(tokens, word_start_mask, word_end_mask, slot_labels_ids, heads_dep, rels_dep, pos_labels,
                       max_seq_len):
    word_start_mask.pop(0)
    word_end_mask.pop(0)
    offsets = np.array(word_start_mask).nonzero()[0]
    offsets = offsets[1:] - offsets[:-1]
    word_start_mask.pop(-1)
    word_end_mask.pop(-1)
    counter = len(offsets) - 1
    LEN = len(tokens)
    assert sum(offsets) == LEN
    assert len(tokens) == len(slot_labels_ids)
    while LEN > max_seq_len - 2:
        LEN -= offsets[counter]
        counter -= 1
    deleted_index = sum(offsets[counter + 1:])
    for i in range(deleted_index):
        word_start_mask.pop()
        word_end_mask.pop()
        tokens.pop()
        slot_labels_ids.pop()

    deleted_index_dep = len(offsets) - 1 - counter

    for i in range(deleted_index_dep):
        heads_dep.pop()
        rels_dep.pop()
        pos_labels.pop()

    assert len(tokens) == len(word_start_mask) == len(word_end_mask) == len(slot_labels_ids)
    assert len(heads_dep) == len(rels_dep) == len(pos_labels)


def convert_examples_to_features(examples, max_seq_len, tokenizer, processor,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        ### add token span function to find word_start and end mask
        tokens, word_start_mask, word_end_mask, token_length, slot_labels_ids = token_span(example.words, tokenizer,
                                                                                           example.slot_labels,
                                                                                           pad_token_label_id)

        heads_dep = example.head_dep
        rels_dep = [processor.dep_label.get(x, 1) for x in example.rel_dep]
        pos_labs = [processor.pos_label.get(x, 1) for x in example.poss]

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            truncate_word_bert(tokens, word_start_mask, word_end_mask, slot_labels_ids,
                               heads_dep, rels_dep, pos_labs, max_seq_len)
            word_start_mask += [1]
            word_end_mask += [1]
            word_start_mask = [1] + word_start_mask
            word_end_mask = [1] + word_end_mask

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        ### vectors for g2g integration
        rearange_ids = [i for i, e in enumerate(word_start_mask) if e != 0]
        len_arrange = len(rearange_ids)
        rearange_ids_temp = rearange_ids[1:] + [rearange_ids[-1] + 1]
        repeats = np.array(rearange_ids_temp) - np.array(rearange_ids)
        indicies = np.arange(0, len(repeats))
        base_vector = np.repeat(indicies, repeats).tolist()

        assert len(base_vector) == len(word_start_mask)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        if padding_length < 0:
            print(input_ids)
            assert False

        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        word_start_mask = word_start_mask + ([0] * padding_length)
        base_vector = base_vector + ([0] * padding_length)
        rearange_ids = rearange_ids + ([0] * (max_seq_len - len_arrange))
        word_end_mask = word_end_mask + ([0] * padding_length)

        heads_dep = [0] + heads_dep + [0]
        heads_dep = [min(len(heads_dep) - 1, x) for x in heads_dep]

        rels_dep = [0] + rels_dep + [0]
        heads_dep = heads_dep + ([0] * (max_seq_len - len(heads_dep)))
        rels_dep = rels_dep + ([0] * (max_seq_len - len(rels_dep)))
        pos_labs = [0] + pos_labs + [0]
        pos_labs = pos_labs + ([0] * (max_seq_len - len(pos_labs)))

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        assert len(word_start_mask) == max_seq_len, "Error with input length {} vs {}".format(len(word_start_mask),
                                                                                              max_seq_len)
        assert len(word_end_mask) == max_seq_len, "Error with input length {} vs {}".format(len(word_end_mask),
                                                                                            max_seq_len)
        assert len(rearange_ids) == max_seq_len, "Error with input length {} vs {}".format(len(rearange_ids),
                                                                                           max_seq_len)
        assert len(base_vector) == max_seq_len, "Error with input length {} vs {}".format(len(base_vector), max_seq_len)

        assert len(pos_labs) == max_seq_len

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          rearange_ids=rearange_ids,
                          base_vectors=base_vector,
                          word_start_mask=word_start_mask,
                          word_end_mask=word_end_mask,
                          token_type_ids=token_type_ids,
                          heads_dep=heads_dep,
                          rels_dep=rels_dep,
                          pos_labs=pos_labs,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if mode == "train":
            logger.info("Loading dependency labels from cached file...")
            processor.dep_label = pickle.load(open("./data/deplabel.pkl", 'rb'))
            processor.pos_label = pickle.load(open("./data/poslabel.pkl", 'rb'))
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For the mode, only train, dev, and test is available")

        if mode != "train":
            processor.dep_label = pickle.load(open("./data/deplabel.pkl", 'rb'))
            processor.pos_label = pickle.load(open("./data/poslabel.pkl", 'rb'))
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, processor,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        if mode == "train":
            logger.info("Saving dependency labels")
            pickle.dump(processor.dep_label, open('./data/deplabel.pkl', 'wb'))
            pickle.dump(processor.pos_label, open('./data/poslabel.pkl', 'wb'))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_word_start_mask = torch.tensor([f.word_start_mask for f in features], dtype=torch.long)
    all_word_end_mask = torch.tensor([f.word_end_mask for f in features], dtype=torch.long)
    all_rearange_ids = torch.tensor([f.rearrange_ids for f in features], dtype=torch.long)
    all_base_vectors = torch.tensor([f.base_vectors for f in features], dtype=torch.long)
    all_heads_dep = torch.tensor([f.heads_dep for f in features], dtype=torch.long)
    all_rels_dep = torch.tensor([f.rels_dep for f in features], dtype=torch.long)
    all_pos_labs = torch.tensor([f.pos_labs for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_rearange_ids, all_base_vectors, all_word_start_mask,
                            all_word_end_mask,
                            all_token_type_ids, all_heads_dep, all_rels_dep, all_pos_labs, all_intent_label_ids,
                            all_slot_labels_ids)

    return dataset, processor.dep_label, processor.pos_label
