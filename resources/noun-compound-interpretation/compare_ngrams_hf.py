import os
import re
import json

import time
import tqdm
import urllib
import random
import logging
import argparse

import pandas as pd

from bs4 import BeautifulSoup
from collections import Counter
from datasets import load_dataset
from xml.sax.saxutils import unescape


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Prefix and File to Compare Ngrams')
    parser.add_argument('--in_file', metavar='path', required=True, help='nc paraphrase file path')
    parser.add_argument('--out_dir', metavar='path', required=True, help='where to store the results')
    parser.add_argument('--n', type=int, required=True, help='n for the ngrams')
    args = parser.parse_args()

    # Load ngrams
    ngrams = load_ngrams(args.in_file, args.n)

    # Set streaming to true so we can use the dataset immediately without downloading everything
    c4dataset = load_dataset("c4", "en", split='train', streaming=True)

    # Search for ngram overlap with C4
    out_file = f"{args.out_dir}/{args.n}gram.tsv"
    count_ngrams(out_file, ngrams, c4dataset, args.n)


def split_ngrams(n, list_of_words):
    return {" ".join(list_of_words[i:i + n]) for i in range(len(list_of_words) - n + 1)}


def load_ngrams(in_file, n):
    df = pd.read_csv(in_file)
    ngrams = set()

    for i, row in df.iterrows():
        paraphrases = [row[i].split() for i in range(1, 6) if type(row[i]) is not float]

        for paraphrase in paraphrases:
            ngrams.update(split_ngrams(n, paraphrase))

    return ngrams


def count_ngrams(out_file, ngrams, c4dataset, n):
    num_examples = 364868892
    bs = 512
    num_batches = int(num_examples / bs)
    pbar = tqdm.tqdm(total=num_batches)  # number of egs in c4 en
    iterable = iter(c4dataset)

    batches = 0
    curr_ngrams = set(ngrams)
    counter = Counter([])

    try:
        while True:
            batch = [w for _ in range(bs) for w in next(iterable)['text'].split()]
            intersection = list(curr_ngrams.intersection(split_ngrams(n, batch)))
            counter.update(Counter(intersection))
            pbar.update(1)
            batches += 1

            if batches % 1000 == 0:
                df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
                df.to_csv(out_file.replace(".tsv", f"_{batches}.tsv"), sep="\t")

    except Exception as exp:
        logger.error(exp)

    pbar.close()

    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df.to_csv(out_file.replace(".tsv", "_last.tsv"), sep="\t")


if __name__ == "__main__":
    main()
