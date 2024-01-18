import ast

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_train_df(file_path='train_gold.csv'):
    train_df_raw = pd.read_csv(file_path)
    train_dict = {'nc': [], 'paraphrase': []}

    for index, row in train_df_raw.iterrows():
        nc = row['w1'] + " " + row['w2']
        train_dict['nc'].append(nc)
        train_dict['paraphrase'].append(row['paraphrase'])

    train_df = pd.DataFrame.from_dict(train_dict)
    return train_df


def load_saved_test_df(file_path='test_df.csv'):
    df = pd.read_csv(file_path, usecols=['nc', 'paraphrases'])

    for i, row in df.iterrows():
        string_literal = df.at[i, 'paraphrases']
        df.at[i, 'paraphrases'] = ast.literal_eval(string_literal)

    return df


def load_difficult_set(file_path='difficult_set.csv'):
    diff_df_raw = pd.read_csv(file_path)
    diff_dict = {'nc': [], 'paraphrases': []}

    for index, row in diff_df_raw.iterrows():
        nc = row['nc']
        diff_dict['nc'].append(nc)
        diff_dict['paraphrases'].append([row['para']])

    diff_df = pd.DataFrame.from_dict(diff_dict)
    return diff_df


def split_train_df_to_dataset(df, split=0.2):
    train_df, valid_df = train_test_split(df, test_size=0.2)
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    return train_dataset, valid_dataset


# Old Methods

def load_valid_dataset(file_path='valid_ds.csv'):
    valid_df_raw = pd.read_csv(file_path)
    dict_for_df = {'nc': [], 'paraphrases': []}

    for index, row in valid_df_raw.iterrows():
        dict_for_df['nc'].append(row['nc'])
        dict_for_df['paraphrases'].append(row['paraphrases'])

    df = pd.DataFrame.from_dict(dict_for_df)
    dataset = Dataset.from_pandas(df)

    return dataset


def load_test_valid_dataset(file_path='test_gold.csv'):
    test_df_raw = pd.read_csv(file_path)
    test_dict = {}

    for index, row in test_df_raw.iterrows():
        nc = row['w1'] + " " + row['w2']
        if nc in test_dict.keys():
            test_dict[nc].append(row['paraphrase'])
        else:
            test_dict[nc] = [row['paraphrase']]

    test_dict_for_df = {'nc': [], 'paraphrases': []}
    for noun_compound, paras in test_dict.items():
        test_dict_for_df['nc'].append(noun_compound)
        test_dict_for_df['paraphrases'].append(paras)

    test_df = pd.DataFrame.from_dict(test_dict_for_df)
    test_df, valid_df = train_test_split(test_df, test_size=0.2)
    test_dataset = Dataset.from_pandas(test_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    return test_dataset, valid_dataset


def load_test_valid_df(file_path):
    df = pd.read_csv(file_path, usecols=['nc','paraphrases'])

    for i, row in df.iterrows():
        string_literal = df.at[i, 'paraphrases']
        df.at[i, 'paraphrases'] = ast.literal_eval(string_literal)

    return df
