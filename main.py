import argparse

import argparse
import collections
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm, trange
from sklearn.decomposition import PCA as sklearnPCA
from collections import Counter

from bert_model.py.processors import *

#from bert.tokenizers import FullTokenizer
#from bert.processors import Dataset_single_Processor

parser = argparse.ArgumentParser()

parser.add_argument("--task_name")
parser.add_argument("--data_dir")
parser.add_argument("--vocab_file")
parser.add_argument("--bert_config_file")
parser.add_argument("--output_dir")
parser.add_argument("--init_checkpoint")
parser.add_argument("--max_seq_len")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--no_cuda", type=bool)
parser.add_argument("--accumulate_gradients", type=int)
parser.add_argument("--train_batch_size", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--max_seq_length", type=int)
parser.add_argument("--do_lower_case", type=bool)

arguments = ['--task_name=dataset_single',
            '--data_dir=data',
            '--vocab_file=bert/vocab.txt',
            '--bert_config_file=bert/config.json',
            '--output_dir=results/dataset/single',
            '--init_checkpoint=ru_conversational_cased_L-12_H-768_A-12/pytorch_model.bin',
            '--max_seq_len=512',
            '--accumulate_gradients=1',
            '--no_cuda=true',
            '--train_batch_size=256',
            '--seed=42',
            '--max_seq_length=512']

args = parser.parse_args(arguments)

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')

#logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

if args.accumulate_gradients < 1:
    raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(args.accumulate_gradients))

args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

bert_config = BertConfig.from_json_file(args.bert_config_file)

if args.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
os.makedirs(args.output_dir, exist_ok=True)

# prepare dataloader
processors = {
    "dataset_single": Dataset_single_Processor
}

processor = processors[args.task_name]()
label_list = processor.get_labels()

tokenizer = FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

# training set
train_examples = None
num_train_steps = None
train_examples = processor.get_train_examples(args.data_dir)

"""
num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)

train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

# test set
if args.eval_test:
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

# model and optimizer
model = BertForSequenceClassification(bert_config, len(label_list))

if args.init_checkpoint is not None:
    model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
model.to(device)

if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

no_decay = ['bias', 'gamma', 'beta']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
]

optimizer = BERTAdam(optimizer_parameters,
                      lr=args.learning_rate,
                      warmup=args.warmup_proportion,
                      t_total=num_train_steps)

# train
output_log_file = os.path.join(args.output_dir, "log.txt")
print("output_log_file=", output_log_file)
with open(output_log_file, "w") as writer:
    if args.eval_test:
        writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
    else:
        writer.write("epoch\tglobal_step\tloss\n")

global_step = 0
epoch = 0
for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    epoch += 1
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()  # We have accumulated enought gradients
            model.zero_grad()
            global_step += 1

    # eval_test
    if args.eval_test:
        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        with open(os.path.join(args.output_dir, "test_ep_" + str(epoch) + ".txt"), "w") as f_test:
            for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                logits = F.softmax(logits, dim=-1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                outputs = np.argmax(logits, axis=1)
                for output_i in range(len(outputs)):
                    f_test.write(str(outputs[output_i]))
                    for ou in logits[output_i]:
                        f_test.write(" " + str(ou))
                    f_test.write("\n")
                tmp_test_accuracy = np.sum(outputs == label_ids)

                test_loss += tmp_test_loss.mean().item()
                test_accuracy += tmp_test_accuracy

                nb_test_examples += input_ids.size(0)
                nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

    result = collections.OrderedDict()
    if args.eval_test:
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': tr_loss / nb_tr_steps,
                  'test_loss': test_loss,
                  'test_accuracy': test_accuracy}
    else:
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': tr_loss / nb_tr_steps}

    logger.info("***** Eval results *****")
    with open(output_log_file, "a+") as writer:
        for key in result.keys():
            logger.info("  %s = %s\n", key, str(result[key]))
            writer.write("%s\t" % (str(result[key])))
        writer.write("\n")
"""