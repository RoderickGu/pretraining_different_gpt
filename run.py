import os
import json
import numpy as np
import time
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
import logging

from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM
from torchfly.utils import get_pretrained, init_logging
from transformers import AdamW, WarmupLinearSchedule

from dialog_utils import DialogFragmentSampler
from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params
from model import ARDM

logger = logging.getLogger(__name__)


class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = tokenizer.encode("\n\n\n")
        self.sampler = DialogFragmentSampler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        sample = self.data[index]
        dialog = {
            "token_ids": [tokenizer.encode(item) for item in sample['text']]
        }
        dialog_fragments = self.sampler(dialog)
        return dialog_fragments["token_ids"]

    def collate(self, batch):
        batch = [torch.LongTensor([item]) for item in batch[0]]
        return batch

def dialog_to_tensor(tokenzier, dialog, device=None):
    res = [torch.LongTensor([tokenizer.encode(item)]) for item in dialog]
    if device:
        res = [item.to(device) for item in res]
    return res

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    manager = DistributedManager(args)

    # define the tokenizer
    tokenizer = UnifiedTokenizer()

    # construct dataset
    with open("train.json") as f:
        train_data = json.load(f)

    train_dataset = PersuadeDataset(train_data, tokenizer)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate
    )

    # define the model
    model = ARDM(args)

    num_train_optimization_steps = (
        1 * args.num_train_epochs // args.batch_size //
        args.gradient_accumulation_steps
    )

    # dialog = dialog_to_tensor(tokenizer, dialog, device)
    optimizer_parameters = get_transformer_optim_params(args, model)
    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=1e-06)

    if args.warmup_steps < 0:
        args.warmup_steps = int(args.warmup_ratio * len(train_dataset))

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps
    )

    manager.init_training(model, optimizer)

    update_count = 0
    if manager.is_main_rank():
        progress_bar = tqdm.tqdm
    else:
        progress_bar = iter

    if manager.is_main_rank():
        start = time.time()
        update_loss = 0.0
        update_kl = 0.0

    for ep in range(args.num_train_epochs):
        pbar = progress_bar(train_dataloader)

        for batch in pbar:
            batch = [item.to(args.device) for item in batch]

            loss, kl = model.train_one_step(batch)
            manager.backward_loss(loss, model, optimizer)
            update_count += 1

            if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                manager.clip_grad_norm(model, optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # timer
                if manager.is_main_rank():
                    end = time.time()
                    speed = args.batch_size * args.n_gpu * args.gradient_accumulation_steps  / (end - start)
                    start = end
                    # show progress
                    pbar.set_postfix(loss=update_loss, kl=update_kl, speed=speed)
            
            # post-processing
            if manager.is_main_rank():
                update_loss = update_loss * 0.9 + 0.1 * loss.item()
                update_kl = update_kl * 0.9 + 0.1 * kl