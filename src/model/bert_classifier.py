from torch import nn
from .bert import BertModel
from config import *


class BertClassifier(nn.Module):

    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.head = nn.Linear(BERT_HIDDEN, num_labels)

    def forward(self, X):
        cls_token = self.bert(X).last_hidden_state[0, 0, :]
        return self.head(cls_token)
