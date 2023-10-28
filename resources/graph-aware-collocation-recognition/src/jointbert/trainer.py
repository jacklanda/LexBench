import os
import logging
from tqdm import tqdm, trange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
import json

from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

from utils import build_syndep_graph
from itertools import chain

logger = logging.getLogger(__name__)


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


class Trainer(object):
    def __init__(self, args, size_dep_label=0, size_pos=0, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.test_result = []
        self.dev_result = []
        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.size_dep_label = size_dep_label

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        if self.args.use_g2g:
            args.size_dep_label = size_dep_label
            args.pos_size = size_pos
            self.model = self.model_class(args=args, config=self.config, intent_label_lst=self.intent_label_lst,
                                          slot_label_lst=self.slot_label_lst)
        else:
            self.model = self.model_class(config=self.config,
                                          args=args,
                                          intent_label_lst=self.intent_label_lst,
                                          slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = 'cuda'
            free_gpu_id = get_freer_gpu()
            print('Freer gpu id: ', free_gpu_id)
            torch.cuda.set_device(int(free_gpu_id))
        else:
            self.device = "cpu"

        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        warmup_steps = int(self.args.warmup_steps * t_total)
        if self.args.use_two_lr:
            model_nonbert = []
            model_bert = []
            layernorm_params = ['layernorm_key_layer', 'layernorm_value_layer', 'dp_relation_k', 'dp_relation_v']
            for name, param in self.model.named_parameters():
                if 'bert' or 'distilbert' in name and not any(nd in name for nd in layernorm_params):
                    model_bert.append((name, param))
                else:
                    model_nonbert.append((name, param))

            # Prepare optimizer and schedule (linear warmup and decay) for Non-bert parameters
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters_nonbert = [
                {'params': [p for n, p in model_nonbert if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model_nonbert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_nonbert = AdamW(optimizer_grouped_parameters_nonbert, lr=self.args.lr_nonbert)

            scheduler_nonbert = get_linear_schedule_with_warmup(optimizer_nonbert,
                                                                num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)

            # Prepare optimizer and schedule (linear warmup and decay) for Bert parameters
            optimizer_grouped_parameters_bert = [
                {'params': [p for n, p in model_bert if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer_bert = AdamW(optimizer_grouped_parameters_bert, lr=self.args.learning_rate)
            scheduler_bert = get_linear_schedule_with_warmup(
                optimizer_bert, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )

        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        # train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        do_early_stop_dev = False
        early_stop_flag = False
        best_score = 0.0
        counter = 0

        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                # 0 : input_ids
                # 1 : attention_mask
                # 2 : rearrange_ids
                # 3 : base_vectors
                # 4 : word_start_mask
                # 5 : word_end_mask
                # 6 : token_type_ids
                # 7 : heads_dep
                # 8 : rels_dep
                # 9 : pos_labels
                # 10 : intent label
                # 11 : slot_label

                if self.args.use_g2g:
                    pos_labels, graph_dep = build_syndep_graph(batch[1], batch[2], batch[3], batch[7], batch[8],
                                                               batch[9], self.size_dep_label)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[10],
                          'slot_labels_ids': batch[11]}

                if self.args.use_g2g:
                    inputs['graph_dep'] = graph_dep

                if self.args.use_pos:
                    inputs['pos_ids'] = pos_labels

                if not self.args.model_type in ['distilbert', 'distilbert-base-cased',
                                                'distilbert-base-multilingual-cased',
                                                'distilbert-base-nli-stsb-mean-tokens']:
                    inputs['token_type_ids'] = batch[6]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    if self.args.use_two_lr:
                        optimizer_bert.step()
                        optimizer_nonbert.step()
                        scheduler_bert.step()
                        scheduler_nonbert.step()
                    else:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        eval_results = self.evaluate("dev", global_step)
                        test_results = self.evaluate("test", global_step)

                        self.dev_result.append({"intent_acc": eval_results['intent_acc'],
                                                "loss": eval_results['loss'],
                                                "sementic_frame_acc": eval_results['sementic_frame_acc'],
                                                "slot_f1": eval_results['slot_f1'],
                                                "slot_precision": eval_results['slot_precision'],
                                                "slot_recall": eval_results['slot_recall']})
                        self.test_result.append({"intent_acc": test_results['intent_acc'],
                                                 "loss": test_results['loss'],
                                                 "sementic_frame_acc": test_results['sementic_frame_acc'],
                                                 "slot_f1": test_results['slot_f1'],
                                                 "slot_precision": test_results['slot_precision'],
                                                 "slot_recall": test_results['slot_recall']})

                        counter += 1
                        if eval_results["slot_f1"] > best_score:
                            best_score = eval_results["slot_f1"]
                            self.save_model()
                            counter = 0

                    if do_early_stop_dev and counter == 10:
                        early_stop_flag = True
                        break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stop_flag:
                # train_iterator.close()
                break

        pickle.dump(self.dev_result, open(os.path.join(self.args.model_dir, "eval.pkl"), 'wb'))
        pickle.dump(self.test_result, open(os.path.join(self.args.model_dir, "test.pkl"), 'wb'))

        print("final step: " + str(global_step))
        # train_iterator.close()
        return global_step, tr_loss / global_step

    def evaluate(self, mode, global_step=0):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {}
                # 0: input_ids
                #  1: attention_mask
                #  2: rearange_ids
                #  3: base_vectors
                #  4: word_start_mask
                #  5: word_end_mask
                #  6: token_type_ids
                #  7: heads_dep
                #  8: rels_dep
                #  9: pos_labels
                # 10: intent label
                # 11: slot label
                if self.args.use_g2g:
                    pos_labels, graph_dep = build_syndep_graph(batch[1], batch[2], batch[3], batch[7], batch[8],
                                                               batch[9],
                                                               self.size_dep_label)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[10],
                          'slot_labels_ids': batch[11]}

                if self.args.use_g2g:
                    inputs['graph_dep'] = graph_dep

                if self.args.use_pos:
                    inputs['pos_ids'] = pos_labels

                if not self.args.model_type in ['distilbert', 'distilbert-base-cased',
                                                'distilbert-base-multilingual-cased',
                                                'distilbert-base-nli-stsb-mean-tokens']:
                    inputs['token_type_ids'] = batch[6]
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        json.dump(slot_preds_list,
                  open(os.path.join(self.args.model_dir, "slot_preds_list_%s_%d.json" % (mode, global_step)), "w"))
        json.dump(out_slot_label_list,
                  open(os.path.join(self.args.model_dir, "out_slot_label_list_%s_%d.json" % (mode, global_step)), "w"))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        torch.save(model_to_save.state_dict(), os.path.join(self.args.model_dir, "model.bin"))
        torch.save(model_to_save.config, os.path.join(self.args.model_dir, "config.bin"))
        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            args = torch.load(os.path.join(self.args.model_dir, 'training_args.bin'))
            config = torch.load(os.path.join(self.args.model_dir, "config.bin"))

            self.model = self.model_class(args=args, config=config,
                                          intent_label_lst=self.intent_label_lst,
                                          slot_label_lst=self.slot_label_lst)
            model_weights = torch.load(os.path.join(self.args.model_dir, "model.bin"))
            self.model.load_state_dict(model_weights)

            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
