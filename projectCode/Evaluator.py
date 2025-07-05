import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import string
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu

from tqdm import tqdm
def remove_punctuation(anns):
    #no_punct = map(lambda ch: '' if ch in string.puntuation else ch ,ann)
    array = np.array(anns, dtype=object)
    translator = str.maketrans('', '', string.punctuation) 
    tokenizer = np.vectorize(lambda x: x.translate(translator) )
    tokenized_array = tokenizer(array)
    return tokenized_array.tolist()

def meteor_eval_single(anns,pred):
    references = [anot.split() for anot in anns]
    pred = pred.split()
    return meteor_score(references, pred)
    
def METEOR_eval(lst_annotations,lst_pred):
    return [meteor_eval_single(ann,pred) for ann,pred in zip(lst_annotations,lst_pred)]
    
def bleu_eval_single(anns,pred):
    references = [ann.split() for ann in anns]
    pred = pred.split()
    pred[0]=pred[0].capitalize()
    return sentence_bleu(references, pred)
    
def BLEU_eval(lst_annotations,lst_pred):
    return [bleu_eval_single(ann,pred) for ann,pred in zip(lst_annotations,lst_pred)]
    
def CIDEr_eval(annotations,pred):
    pass

def findEvalFunc(eval_func):
    if eval_func == "METEOR":
        return METEOR_eval
    elif eval_func == "BLEU":
        return BLEU_eval
    else:
        return None
        
class Evaluator:
    def __init__(self,model,dataset,corruptions=None,eval_func=None, batch_size = 1, seed=42, device='cpu'):
        torch.manual_seed(seed)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle=True)
        self.corruptions = corruptions
        self.model = model
        self.eval_func = findEvalFunc(eval_func)
        self.device = device

    def evaluate_dataset(self):
        scores = [[] for i in range(len(self.corruptions))]
        for batch in tqdm(self.dl):
            corrupted_batches = self.corrupt_batch(batch[0])
            btch = list(zip(*list(map(list,batch[1]))))
            for i,c_batch in enumerate(corrupted_batches):
                pred = self.model(c_batch.to(self.device))

                if isinstance(pred, tuple) and len(pred) == 4:
                    pred, _, _, _ = pred
                '''
                print(pred)
                print(btch)
                for i,p in enumerate(pred):
                    #print("Current prediction: ", p)
                    #print("Compared to: ", btch[i])
                    scores.append(self.eval_func(list(btch[i]),p))
                '''
                #print(pred)
                labels = remove_punctuation(btch)
                scores[i].extend(self.eval_func(labels,pred))

                c_batch = c_batch.detach().to('cpu')
                
        return scores
    def corrupt_batch(self,batch):
        lst =[corr.apply(batch) for corr in self.corruptions]
        '''
        transform = T.ToPILImage()
        for i in range(len(lst)):
            transform(lst[i][1]).show()
        '''
        return lst
