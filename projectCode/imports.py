import os 
import time
import sys 
import json
from argparse import Namespace

from torch.utils.data import DataLoader

sys.path.append("./models/ExpansionNet_v2")
from projectCode.ModelLoader import loadModel,getExpNetTokenId
from projectCode.CocoDataset import CocoDataset
from projectCode.CartoonDataset import *
from projectCode.Perturbator import *
from projectCode.Evaluator import *
import gc
from projectCode.attack import Attacker

import matplotlib.pyplot as plt

def set_device(device = None):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    return device
   