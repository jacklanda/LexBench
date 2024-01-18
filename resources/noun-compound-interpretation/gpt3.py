import argparse
import time

import openai
import pandas as pd

from datasets import load_dataset

import util

openai.organization = "<ORG-KEY>"
openai.api_key = "<API-KEY>"


def get_random_prompt(train_df, test_df, num_egs=1):
    test_row = test_df.sample()
    test_nc = test_row['nc'].values[0]
    test_paras = test_row['paraphrases'].values[0]

    prompt = ""

    prompt += train_nc_prompt(train_df, num_egs)
    prompt += test_nc_prompt(test_nc)

    return prompt, test_nc, test_paras


def train_nc_prompt(train_df, num_egs, prompt_text="Q: what is the meaning of {}?\n"):
    prompt = ""
    for i in range(num_egs):
        train_row = train_df.sample()
        train_nc = train_row['nc'].values[0]
        train_para = train_row['paraphrase'].values[0]

        train_q = prompt_text.format(train_nc)
        train_a = "A: " + train_para + "\n"

        prompt += train_q + train_a
    return prompt


def test_nc_prompt(test_nc, prompt_text="Q: what is the meaning of {}?\n"):
    test_q = prompt_text.format(test_nc)
    test_a = "A:"
    return test_q + test_a


def get_all_test_prompts(train_df, test_df, num_egs=10, prompt_text="Q: what is the meaning of {}?\n"):
    prompts = []
    for i, test_row in test_df.iterrows():
        test_nc = test_row['nc']
        gold_paras = test_row['paraphrases']

        prompt = ""
        prompt += train_nc_prompt(train_df, num_egs, prompt_text)
        prompt += test_nc_prompt(test_nc, prompt_text)

        prompts.append((test_nc, gold_paras, prompt))

    return prompts


def main():
    parser = argparse.ArgumentParser(description='Noun Compound Interpretation Model')
    parser.add_argument('--out', required=True,
                        help='output filename')

    args = parser.parse_args()
    save_filename = args.out

    train_df = util.load_train_df()
    test_df = util.load_saved_test_df()

    test_prompts = get_all_test_prompts(train_df, test_df, 1, "Q: what is the meaning of {}?\n")
    test_responses = {}

    curr_index = 0  # index to resume from if GPT-3 loop is interrupted

    for test_nc, gold_paras, prompt in test_prompts[curr_index:]:
        # generate number of responses equal to the number of gold paraphrases
        time.sleep(15)
        num_to_gen = 1
        generated_paras = []
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            n=num_to_gen
        )

        for r in response['choices']:
            generated_paras.append(r.text.lstrip())

        test_responses[test_nc] = generated_paras

    to_save = {'nc': [], 'paraphrase': []}
    for nc, paras in test_responses.items():
        for p in paras:
            to_save['nc'].append(nc)
            to_save['paraphrase'].append(p)

    pd.DataFrame(to_save).to_csv('gpt3-responses_{}.csv'.format(save_filename))
