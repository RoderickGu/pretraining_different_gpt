import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM
from torchfly.utils import get_pretrained, init_logging
from transformers import AdamW, WarmupLinearSchedule

from distributed_utils import DistributedManager
from utils import parse_args, freeze_model
from model import ARDM

logger = logging.getLogger(__name__)


class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = tokenizer.encode("\n\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dial_tokens = [
            tokenizer.encode(item) + self.turn_ending
            for item in self.data[index]
        ]
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens

    def collate(self, batch):
        return batch


if __name__ == '__main__':
    init_logging()
    args = parse_args()
    manager = DistributedManager(args)

    # define the tokenizer
    tokenizer = UnifiedTokenizer()
    # define the model
    model = ARDM(args)

    num_train_optimization_steps = (
        1 * args.num_train_epochs // args.batch_size //
        args.gradient_accumulation_steps
    )

    param_optimizer = model.named_parameters()

    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params":
                [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
            "weight_decay": 0.01,
        },
        {
            "params":
                [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
            "weight_decay": 0.0,
        },
    ]

    # dialog = dialog_to_tensor(tokenizer, dialog, device)
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=1e-06)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=500, t_total=num_train_optimization_steps
    )

    manager.init_training(model, optimizer)

    for i in range(1000):
        dialog = [
            torch.LongTensor([np.arange(200)]).to(args.device) for i in range(5)
        ]

        loss, kl = model.train_one_dialog(dialog)

        optimizer.zero_grad()
        manager.backward_loss(loss, optimizer)

        manager.clip_grad_norm(model, optimizer)

        optimizer.step()
        scheduler.step()

        logger.info(f"KL is {kl}")