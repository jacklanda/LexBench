# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/07


# This script is to handle all the tasks that are relevant with
# data processing, data preparation in our paper: ``Benchmarking LLMs for Collocation Understanding''.

import os
import random
from pprint import pprint
from typing import List, Optional

import rich
import torch
from torch.utils.data import Dataset

random.seed(42)


class IdiomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])

        return data, label


def prepare_data_idiom_extraction(
    data_path: str = "dataset/train_english.tsv",
    dump_data_path: Optional[str] = None,
    only_valid_example: bool = True,
    dedup_idiom: bool = True,
    max_data_limit: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    This function is to prepare the idiom data for evaluation.

    Args:
    - data_path: the path of the data file
    """

    # read the data
    with open(data_path, "r") as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    current_phrase = []
    phrase_set = set()
    num_w_phrase = 0
    num_wo_phrase = 0

    for line in lines:
        if line.strip() == "":
            if current_sentence:
                sentences.append((current_sentence, current_phrase))
                current_sentence = []
                current_phrase = []
        else:
            word, label = line.split("\t")
            word = word.strip()
            label = label.strip()
            current_sentence.append(word)
            if label.startswith("B-"):
                current_phrase.append(word)
            elif label.startswith("I-"):
                current_phrase.append(word)

    # Add the last sentence if not empty
    if current_sentence:
        sentences.append((current_sentence, current_phrase))

    if dedup_idiom:
        dedup_sentences = []
        for sentence, phrase in sentences:
            if phrase:
                phrase_span = " ".join(phrase)
                if phrase_span not in phrase_set:
                    dedup_sentences.append((sentence, phrase))
                    phrase_set.add(phrase_span)
            else:
                dedup_sentences.append((sentence, phrase))
        sentences = dedup_sentences

    # Create the output
    output_data = []
    for sentence, phrase in sentences:
        sentence = " ".join(sentence)
        if phrase:
            num_w_phrase += 1
            phrase_span = " ".join(phrase)
        else:
            num_wo_phrase += 1
            continue
        output_data.append(f"{sentence}\t{phrase_span}\n")

    if max_data_limit:
        output_data = random.sample(output_data, max_data_limit)

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"Context\tIdiom\n")
            f.writelines(output_data)

    if verbose:
        print(
            f"Total instances: {num_w_phrase + num_wo_phrase}\nInstances w/ phrase: {num_w_phrase}\nInstances w/o phrase: {num_wo_phrase}\n"
        )

    return output_data


def prepare_data_idiom_detection(
    data_path="",
    dump_data_path="dataset/idiom_detection/prepared/idiom_detection_prepared.tsv",
) -> List[str]:
    pass


if __name__ == "__main__":
    # idiom_extraction_examples = prepare_data_idiom_extraction(
    # data_path="dataset/idiom_extraction/dev_english.tsv",
    # dump_data_path="dataset/idiom_extraction/prepared/idiom_extraction_prepared.tsv",
    # )
    # print(len(idiom_extraction_examples))
    # pprint(data=idiom_list)
    # rich.print_json(data=idiom_list)
    pass
