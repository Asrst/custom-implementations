from enum import IntEnum
from losses import *
from transformers import *


NLP_MODELS = {
    "bert": (BertConfig, BertModel, BertTokenizer, 'bert-base-uncased'),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer, 'albert-base-v2'),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer, 'roberta-base'),
    "xlnet" : (XLNetConfig, XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    "electra" : (ElectraConfig, ElectraModel, ElectraTokenizer, 'google/electra-small-generator')
}

LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "nerloss" : NERLoss
}

class ModelType(IntEnum):
    BERT = 1
    DISTILBERT = 2
    ALBERT = 3
    ROBERTA = 4
    XLNET = 5
    ELECTRA = 6

class TaskType(IntEnum):
    SingleSenClassification = 1
    SentencePairClassification = 2
    NER = 3

class LossType(IntEnum):
    CrossEntropyLoss = 0
    NERLoss = 1
