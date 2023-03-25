from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn

from config import *
from model import BertClassifier
from dataset import CoLADataset


class Trainer:

    def __init__(self):
        self.model = BertClassifier(1)
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.model.cuda()
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.train_dataset = CoLADataset(TRAIN_PATH, tokenizer)
        self.val_dataset = CoLADataset(VAL_PATH, tokenizer)
        self.loss = nn.BCEWithLogitsLoss()

    def configure_optimizer(self):
        """
        We only want to train the adapter and layer norm layers.
        """
        trainable_layers = ["adapter", "LayerNorm"]
        # Keep attention, ff, etc. layers frozen.
        params = [
            param for name, param in self.model.named_parameters()
            if any([(layer in name) for layer in trainable_layers])
        ]
        self.optimizer = AdamW(params, lr=LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * len(self.train_dataset)) * EPOCHS,
        )

    def compute_loss(self, output, labels):
        return self.loss(output, labels)

    def train(self):
        train = self.get_train_dataloader()
        val = self.get_eval_dataloader()
        self.configure_optimizers()
        for epoch in range(EPOCHS):
            train_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            for batch in train:
                x, y = batch["sentence"], batch["label"]
                if self.gpu_available:
                    x, y = x.cuda(), y.cuda()
                outputs = self.model(x)
                loss = self.compute_loss(outputs, y)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            val_loss = 0.0
            for batch in val:
                x, y = batch["sentence"], batch["label"]
                if self.gpu_available:
                    x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    outputs = self.model(x)
                    loss = self.compute_loss(outputs, y)
                    val_loss += loss.item()

        print(
            f"Epoch {epoch} training loss: {train_loss/len(train)}, val loss: {val_loss/len(val)}"
        )
