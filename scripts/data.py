# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/07


# This script is to handle all the tasks that are relevant with
# data processing, data preparation in our paper: ``Benchmarking LLMs for Collocation Understanding''.

import os
import ast
import random
import pandas as pd
from pprint import pprint
from typing import Any, Dict, List, Optional

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
    dedup_by_idiom: bool = True,
    max_data_limit: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the idiom data for extraction evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - only_valid_example: whether to only keep the valid examples
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - max_data_limit: the max data limit
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
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

    if dedup_by_idiom:
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
    dedup_by_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    This function is to prepare the idiom data for evaluation.

    Args:
    - refer_data_path: the path of the refer data file
    - idiom_data_path: the path of the idiom data file
    - dump_data_path: the path of the prepared data file
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - verbose: whether to print the statistics

    Returns:
    - data_examples: the prepared data
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
    if dedup_by_idiom:
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


def prepare_data_idiom_paraphrase(
    idiom_data_path: str,
    dump_data_path: str = "dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv",
    dedup_by_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the idiom data for paraphrase evaluation.

    Args:
    - idiom_data_path: the path of the idiom data file
    - dump_data_path: the path of the prepared data file
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - verbose: whether to print the statistics
    """
    data_examples = []
    idiom_df = pd.read_csv(idiom_data_path)

    for i in range(len(idiom_df)):
        data_example = {
            "id": str(i).zfill(4),
            "idiom": idiom_df.iloc[i, 0],
            "paraphrase": idiom_df.iloc[i, 1],
            "context_idiomatic": idiom_df.iloc[i, 2],
            "context_literal": idiom_df.iloc[i, 3],
        }
        data_examples.append(data_example)

    if dedup_by_idiom:
        idx = 0
        dedup_data_examples = []
        dedup_idiom_set = set()
        for item in data_examples:
            dedup_idiom_set.add(item["idiom"])
        for item in data_examples:
            if item["idiom"] in dedup_idiom_set:
                item["id"] = str(idx).zfill(3)
                dedup_data_examples.append(item)
                dedup_idiom_set.remove(item["idiom"])
                idx += 1
        data_examples = dedup_data_examples

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"id\tidiom\tparaphrase\tcontext_idiomatic\tcontext_literal\n")
            for item in data_examples:
                f.write(
                    f"{item['id']}\t{item['idiom']}\t{item['paraphrase']}\t{item['context_idiomatic']}\t{item['context_literal']}\n"
                )
    if verbose:
        print("Total instances:", len(data_examples))

    return data_examples


def sample_data_collocate_retrieval(
    data_path: str = "dataset/collocate_retrieval/Collocations_en.csv",
    dump_data_path: Optional[str] = None,
    category_num: int = 8,
    max_instance_num_per_category: int = 30,
    min_context_size: int = 16,
    max_context_size: int = 64,
    dedup: bool = True,
) -> None:
    """
    Sample the collocate retrieval data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - category_num: the number of categories to sample
    - max_instance_num_per_category: the max instance number per category
    - min_context_size: the min context size
    - max_context_size: the max context size
    - dedup: whether to deduplicate the data by the whole collocation

    Returns:
    - None
    """
    if category_num == 8:
        lf_category = [
            "Magn",
            "AntiMagn",
            "Ver",
            "AntiVer",
            "Bon",
            "AntiBon",
            "Son",
            "Oper1",
        ]
    else:
        raise NotImplementedError(f"category_num={category_num} is not implemented.")

    # Step 0. load all instances from the original file
    df_all = pd.read_csv(data_path, sep="\t")

    # Step 1. deduplicate by the 2nd and 5th value of column (collocation = base ⊕ collocate), if needed
    if dedup:
        # df_all = df_all.drop_duplicates(subset=[df_all.columns[1]])
        df_all = df_all.drop_duplicates(subset=[df_all.columns[1], df_all.columns[4]])

    # Step 2. filter out all the rows that 11th column's value word size is not in the interval [16, 64]
    df_all = df_all[
        df_all[df_all.columns[10]].apply(
            lambda x: min_context_size <= len(x.split()) <= max_context_size
        )
    ]

    # Step 3. random select 30 examples for each group which is clustered by the 6th column, from the `lf_category`
    df_sample = pd.DataFrame()
    for category in lf_category:
        df_category = df_all[df_all[df_all.columns[5]] == category]
        df_category_sample = df_category.sample(
            n=max_instance_num_per_category,
            random_state=42,
            # replace=True,
        )
        df_sample = df_sample._append(df_category_sample)

    # Step 4. dump the sampled data to the `dump_data_path`
    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        df_sample.to_csv(dump_data_path, sep="\t", index=False)


def prepare_data_collocate_retrieval(
    data_path: str = "dataset/collocate_retrieval/Collocations_en.csv",
    dump_data_path: Optional[str] = None,
    base_word_num: Optional[int] = None,
    collocate_word_num: Optional[int] = None,
    max_data_limit: int = 320,
    max_instance_num_per_category: int = 40,
    mask_collocate: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the collocate retrieval data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - base_word_num: the number of words in the base (constraint)
    - collocate_word_num: the number of words in the collocate (constraint)
    - max_data_limit: the max data limit
    - max_instance_num_per_category: the max instance number per category
    - mask_collocate: whether to mask the collocate
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """
    df = pd.read_csv(data_path, sep="\t")

    category2freq = dict()
    instances_processed = []
    for i in range(len(df)):
        base = df.iloc[i, 1].replace("_", "")
        collocate_idx = int(df.iloc[i, 6]) - 1
        collocate = df.iloc[i, 4].replace("_", "")
        if base_word_num and len(base.split()) > base_word_num:
            continue
        if collocate_word_num and len(collocate.split()) > collocate_word_num:
            continue
        collocation = (
            f"{base} {collocate}"
            if int(df.iloc[i, 6]) > int(df.iloc[i, 7])
            else f"{collocate} {base}"
        )
        label = df.iloc[i, 5]
        if category2freq.get(label, 0) >= max_instance_num_per_category:
            continue
        if mask_collocate:
            words = df.iloc[i, 10].split()
            if words[collocate_idx] != collocate:
                print(
                    f"id: {str(i).zfill(3)}\twords[collocate_idx] ({words[collocate_idx]}) != collocate ({collocate})."
                    "\tplease compile this instance manually."
                )
                # continue
            else:
                words[collocate_idx] = "[MASK]"
            context = " ".join(words)
        else:
            context = df.iloc[i, 10]
        instance = {
            "id": str(i).zfill(3),
            "base": base,
            "collocate": collocate,
            "collocation": collocation,
            "label": df.iloc[i, 5],
            "context": context,
        }
        instances_processed.append(instance)
        category2freq[label] = category2freq.get(label, 0) + 1

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `instances_processed`
        with open(dump_data_path, "w") as f:
            for instance in instances_processed:
                f.write(
                    f"{instance['id']}\t{instance['base']}\t{instance['collocate']}\t{instance['collocation']}\t{instance['label']}\t{instance['context']}\n"
                )


def prepare_noun_compound_interpretation(
    data_path: str,
    dump_data_path: Optional[str] = None,
    max_data_limit: int = 110,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Prepare the noun compound interpretation data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - max_data_limit: the max data limit
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """
    df = pd.read_csv(data_path, sep=",")
    output_data = []
    for i in range(len(df)):
        noun_compound = df.iloc[i, 1]
        paraphrases = ast.literal_eval(df.iloc[i, 2])
        output_data.append(
            {
                "id": str(i).zfill(3),
                "noun_compound": noun_compound,
                "paraphrases": paraphrases,
            }
        )

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `output_data`
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['id']}\t{item['noun_compound']}\t{item['paraphrases']}\n"
                )

    if verbose:
        print("Total instances:", len(output_data))

    return output_data


if __name__ == "__main__":
    # Task 1: Idiomatic Expression Detection (IED)
    # idiom_detection_examples = prepare_data_idiom_detection(
    # refer_data_path="/Users/jacklanda/Desktop/LexBench/scripts/dataset/idiom_detection/reference_data.csv",
    # idiom_data_path="/Users/jacklanda/Desktop/LexBench/scripts/dataset/idiom_detection/idiom_data.csv",
    # )

    # Task 2: Idiomatic Expression Extraction (IEE)
    # idiom_extraction_examples = prepare_data_idiom_extraction(
    # data_path="dataset/idiom_extraction/dev_english.tsv",
    # dump_data_path="dataset/idiom_extraction/prepared/idiom_extraction_prepared.tsv",
    # )

    # Task 3: Idiomatic Expression Paraphrase (IEP)
    # idiom_paraphrase_examples = prepare_data_idiom_paraphrase(
    # idiom_data_path="/Users/jacklanda/Desktop/LexBench/scripts/dataset/idiom_paraphrase/data_cleaned.csv"
    # )

    # Task 4: Lexical Collocate Retrieval (LCR)
    # sample_data_collocate_retrieval(
        # data_path="dataset/collocate_retrieval/Collocations_en.tsv",
        # dump_data_path="dataset/collocate_retrieval/Collocations_en_test.tsv",
        # category_num=8,
        # max_instance_num_per_category=40,
    # )
    # collocate_retrieval_examples = prepare_data_collocate_retrieval(
        # data_path="dataset/collocate_retrieval/Collocations_en_test.tsv",
        # dump_data_path="dataset/collocate_retrieval/prepared/collocate_retrieval_prepared.tsv",
        # base_word_num=3,
        # collocate_word_num=10,
    # )

    # Task 9: Noun Compound Interpretation (NCI)
    # noun_compound_interpretation_examples = prepare_noun_compound_interpretation(
        # data_path="dataset/noun_compound_interpretation/test_df.csv",
        # dump_data_path="dataset/noun_compound_interpretation/prepared/noun_compound_interpretation_prepared.tsv",
    # )
    noun_compound_interpretation_examples = prepare_noun_compound_interpretation(
        data_path="dataset/noun_compound_interpretation/valid_df.csv",
        dump_data_path="dataset/noun_compound_interpretation/prepared/examples.tsv",
    )
