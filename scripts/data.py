# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/07


# This script is to handle all the tasks that are relevant with
# data processing, data preparation in our paper: ``Benchmarking LLMs for Collocation Understanding''.

import os
import random
import pandas as pd
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
    refer_data_path: str,
    idiom_data_path: str,
    dump_data_path: str = "dataset/idiom_detection/prepared/idiom_detection_prepared.tsv",
    dedup_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    This function is to prepare the idiom data for evaluation.

    Args:
    - refer_data_path: the path of the refer data file
    - idiom_data_path: the path of the idiom data file
    - dump_data_path: the path of the prepared data file
    """

    data_examples = []

    # read the data
    refer_df = pd.read_csv(refer_data_path)
    idiom_df = pd.read_csv(idiom_data_path)

    # get a dict include the column "sentence1" and "sentence2"
    refer_dict = {
        item["sentence1"]: item["sentence2"]
        for item in refer_df.to_dict(orient="records")
    }

    idiom_df["idiom"] = idiom_df["sentence1"].apply(lambda x: refer_dict.get(x, ""))

    # filter all rows that have the 3 times repeats value of "sentence1" in `idiom_df`
    sent2freq = dict()
    for item in idiom_df["sentence1"].tolist():
        sent2freq[item] = sent2freq.get(item, 0) + 1
    idiom_df["frequency"] = idiom_df["sentence1"].apply(lambda x: sent2freq[x])

    # get the row of frequency == 4
    idiom_df_four = idiom_df[idiom_df["frequency"] == 4]

    # iterate each 4 rows as a group in `idiom_df_four`
    for i in range(0, len(idiom_df_four), 4):
        # assign A, B, C, D to each row in this group
        data_example = {
            "id": str(i // 4).zfill(3),
            "context": idiom_df_four.iloc[i, 1].replace("\t", " "),
            "idiom": idiom_df_four.iloc[i, 3].replace("\t", " "),
        }
        letter_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        for j in range(4):
            letter = letter_mapping[j]
            data_example[letter] = idiom_df_four.iloc[i + j, 2]

            if idiom_df_four.iloc[i + j, 0] == "1":
                data_example["target"] = letter

        data_examples.append(data_example)

    # deduplicate `data_example` by field "idiom" for each item
    if dedup_idiom:
        idx = 0
        dedup_data_examples = []
        dedup_idiom_set = set()
        for item in data_examples:
            dedup_idiom_set.add(item["idiom"] + item["target"])
        for item in data_examples:
            if item["idiom"] + item["target"] in dedup_idiom_set:
                item["id"] = str(idx).zfill(3)
                dedup_data_examples.append(item)
                dedup_idiom_set.remove(item["idiom"] + item["target"])
                idx += 1
        data_examples = dedup_data_examples

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"id\tcontext\tidiom\tA\tB\tC\tD\ttarget\n")
            for item in data_examples:
                f.write(
                    f"{item['id']}\t{item['context']}\t{item['idiom']}\t{item['A']}\t{item['B']}\t{item['C']}\t{item['D']}\t{item['target']}\n"
                )

    if verbose:
        # rich.print_json(data=data_examples)
        print("Total instances:", len(data_examples))

    return data_examples


if __name__ == "__main__":
    # idiom_extraction_examples = prepare_data_idiom_extraction(
    # data_path="dataset/idiom_extraction/dev_english.tsv",
    # dump_data_path="dataset/idiom_extraction/prepared/idiom_extraction_prepared.tsv",
    # )
    # print(len(idiom_extraction_examples))
    # pprint(data=idiom_list)
    # rich.print_json(data=idiom_list)
    idiom_detection_examples = prepare_data_idiom_detection(
        refer_data_path="/Users/jacklanda/Desktop/LexBench/scripts/dataset/idiom_detection/reference_data.csv",
        idiom_data_path="/Users/jacklanda/Desktop/LexBench/scripts/dataset/idiom_detection/idiom_data.csv",
    )
