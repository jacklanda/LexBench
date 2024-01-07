import os
from tqdm import tqdm
import pandas as pd
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset
from pprint import pprint
from collections import Counter
import random
import numpy as np
from typing import List, Dict
import json
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
'''from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score'''
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from TorchCRF import CRF
import unicodedata
import re