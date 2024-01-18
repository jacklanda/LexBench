import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets import Dataset

from transformers import Adafactor, AutoTokenizer, AutoModelForSeq2SeqLM

import semeval_scorer
import util
import eval_helper


def main():
    parser = argparse.ArgumentParser(description='Noun Compound Interpretation Model')
    parser.add_argument('--architechture', required=True,
                        help='name of model architechture to use (huggingface model)')
    parser.add_argument('--lr', default=3e-4,
                        help='learning rate')
    parser.add_argument('--bs', default=1,
                        help='batch size')
    parser.add_argument('--epochs', default=10,
                        help='number of epochs')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ", device)

    train_dataset, valid_dataset = util.load_train_valid_dataset('data/train_gold.csv')
    train_loader = DataLoader(train_dataset, batch_size=int(args.bs))
    valid_loader = DataLoader(valid_dataset, batch_size=int(args.bs))

    test_valid_df = util.load_test_valid_df('data/valid_df.csv')
    test_valid_dataset = Dataset.from_pandas(test_valid_df)
    test_valid_loader = DataLoader(test_valid_dataset, batch_size=1)  # TODO: figure out how to change this from bs=1

    if args.architechture.startswith('t5'):
        if args.load:
            model = AutoModelForSeq2SeqLM.from_pretrained("./model_" + args.architechture)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.architechture)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.architechture)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.architechture)
            train(model, tokenizer, train_loader, valid_loader, device, args)
            model.save_pretrained(os.getcwd() + '/model_' + args.architechture)

        test(model, tokenizer, test_valid_loader, device)
    else:
        print("unsupported model")


def train(model, tokenizer, train_loader, valid_loader, device, args):
    
    lr = float(args.lr)
    num_epochs = int(args.epochs)

    num_batches = len(train_loader)

    optim = Adafactor(model.parameters(), lr=lr, relative_step=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optim.zero_grad()

            ncs = batch['nc']
            paraphrases = batch['paraphrase']

            tokenized_ncs = tokenizer(ncs, padding=True, truncation=True, return_tensors='pt')
            tokenized_paras = tokenizer(paraphrases, padding=True, truncation=True, return_tensors='pt')

            input_ids = tokenized_ncs['input_ids'].to(device)
            attention_mask = tokenized_ncs['attention_mask'].to(device)

            labels = tokenized_paras['input_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            epoch_loss += loss.item()

            loss.backward()
            optim.step()

        model.eval()
        valid_loss = 0
        valid_num_batches = len(valid_loader)
        with torch.no_grad():
            for batch in valid_loader:
                ncs = batch['nc']
                paraphrases = batch['paraphrase']

                tokenized_ncs = tokenizer(ncs, padding=True, truncation=True, return_tensors='pt')
                tokenized_paras = tokenizer(paraphrases, padding=True, truncation=True, return_tensors='pt')

                input_ids = tokenized_ncs['input_ids'].to(device)
                attention_mask = tokenized_ncs['attention_mask'].to(device)
                labels = tokenized_paras['input_ids'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                valid_loss += loss.item()

        epoch_loss /= num_batches
        valid_loss /= valid_num_batches
        print("epoch: " + str(epoch + 1) + ", train loss: " + str(epoch_loss)+ ", valid loss: " + str(valid_loss))


def test(model, tokenizer, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_score = 0

        # TODO: right now batch size needs to be 1 for this... figure out how to change maybe...
        # TODO: might not actually be able to use a "loader" because it turns strings into tuples of strings
        for batch in test_loader:
            torch.cuda.empty_cache()

            ncs = batch['nc']
            gold_paraphrases = batch['paraphrases']
            # change from tuples to just strings
            gold_paraphrases = list(map(lambda x: x[0], gold_paraphrases))

            tokenized_ncs = tokenizer(ncs, padding=True, truncation=True, return_tensors='pt')

            input_ids = tokenized_ncs['input_ids'].to(device)

            num_paras = len(gold_paraphrases)
            gen_paras = []

            for i in range(num_paras):
                output = generate_top_p(input_ids, model, 0.92)
                gen_paras.append(tokenizer.decode(output, skip_special_tokens=True))

            # TODO: add other scoring metrics
            nc_average_score = eval_helper.batch_meteor(gen_paras, gold_paraphrases)
            total_score += nc_average_score

        average_score = total_score / len(test_loader)
        print("test meteor score: ", average_score)


def generate_top_p(input_ids, model, p):
    tokenized_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_p=p,
        top_k=0
    )
    return tokenized_output[0]


def generate_top_k(input_ids, model, k, temp=0):
    tokenized_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_k=k,
        temperature=temp
    )
    return tokenized_output[0]


def generate_top_p_k(input_ids, model, p, k, temp=0):
    tokenized_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_p=p,
        top_k=k,
        temperature=temp
    )
    return tokenized_output[0]


if __name__ == "__main__":
    main()
