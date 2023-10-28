import os

os.environ['TRANSFORMERS_CACHE'] = './hf_models/'

import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset, dep_label, pos_label = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, _, _ = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, _, _ = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, len(dep_label), len(pos_label), train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")

    if args.do_eval:
        trainer.load_model()
        results = trainer.evaluate("test")
        return results


if __name__ == '__main__':
    # python src/jointbert/main.py --task dataset_for_jointbert_with_parsed_trees_extended --model_type bert-large-uncased --num_train_epochs 5 --max_seq_len 150 --train_batch_size 8 --model_dir bert-large-uncased_5_g2g_pos_try1_999 --do_train --do_eval --learning_rate 1e-5 --warmup_steps 0.01 --logging_steps 7000 --save_steps 7000 --use_g2g --use_pos --seed 999
    # python src/jointbert/main.py --task dataset_for_jointbert_with_parsed_trees_fr --model_type camembert-base --num_train_epochs 5 --max_seq_len 150 --train_batch_size 32 --model_dir camembert-fr-fr_5_g2g_pos_try1_999 --do_train --do_eval --learning_rate 1e-5 --warmup_steps 0.01 --logging_steps 4000 --save_steps 4000 --use_g2g --use_pos --seed 999
    # python src/jointbert/main.py --task dataset_for_jointbert_with_parsed_trees_es --model_type roberta-base-bne --num_train_epochs 5 --max_seq_len 150 --train_batch_size 32 --model_dir roberta-base-bne-es-es_5_g2g_pos_try1_999 --do_train --do_eval --learning_rate 1e-5 --warmup_steps 0.01 --logging_steps 4000 --save_steps 4000 --use_g2g --use_pos --seed 999
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0.0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")

    ############# g2g config
    parser.add_argument("--use_g2g", action="store_true", help="Use G2G for encoder")
    parser.add_argument("--use_pos", action="store_true", help="Use POS for encoder")
    parser.add_argument("--use_two_attn", action="store_true", help="Difference attention for query and key")
    parser.add_argument("--just_attn", action="store_true", help="Ignore value interaction")

    parser.add_argument("--use_two_lr", action="store_true", help="Use two learning rate for BERT and non-BERT")
    parser.add_argument("--lr_nonbert", default=1e-3, type=float, help="The initial learning rate for non-BERT params")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    res = main(args)
