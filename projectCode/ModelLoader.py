import torch
from torch import nn
from torchvision import transforms

import numpy as np

from PIL import Image

import argparse
import pickle
from argparse import Namespace
import os
import sys

################################# EXPANSION NET 
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
################################# mPLUG
from transformers import AutoConfig, AutoModel
################################### BLIP 
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, BlipModel
################################### VIT
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
################################### POS
from flair.data import Sentence
from flair.models import SequenceTagger
################################### CLIP 
from transformers import CLIPProcessor, CLIPModel

def loadModel(name,model_path,**model_kwargs):
    if name == "GRIT":
        return getGRIT(model_path,**model_kwargs)
    elif name == "ExpansionNet":
        return getExpNet(model_path,**model_kwargs)
    elif name == "mPLUG":
        return getMPLUG(**model_kwargs)
    elif name == "BLIP":
        return getBLIP(**model_kwargs)
    elif name == "VIT":
        return getVIT(**model_kwargs)
    elif name == "POS":
        return getPOSClassifier()

def getBLIP(device, is_adversarial=False,blured = False,blur_kernel_size = 3):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    if blured:
        print("blur size:",blur_kernel_size)
        #blur = transforms.GaussianBlur(blur_kernel_size)
        sigma = (blur_kernel_size-1)/6
        blur = transforms.GaussianBlur(blur_kernel_size,sigma)
    def pseudomodel(images):
        if blured:
            images = blur(images)
            images = torch.clamp(images,0,1)

        if is_adversarial:
            inputs = processor(images,"",return_tensors="pt", padding = True).to(device)
            inputs['pixel_values']=images.to(device)
            
            model.train()

            out = model.forward(**inputs,return_dict=True) #returns logits for the last token only
            logits = out['logits']
            #probs = logits.softmax(dim=2)

            return logits
        else:
            inputs = processor(images,return_tensors="pt", padding = True).to(device)
            inputs['pixel_values']=images.to(device)
            model.eval()
            
            out = model.generate(
                inputs["pixel_values"],
                return_dict_in_generate=True,
                output_scores=True,
                max_length=20
            )
            logits = out.scores  # Tuple of logits, one per token
            return processor.batch_decode(out.sequences[0], skip_special_tokens=True), logits
            #out = model.generate(**inputs)
            #return processor.batch_decode(out, skip_special_tokens=True) 
    return pseudomodel

def getBLIPCLassifier(device, labels, blured=False,blur_kernel_size=3):
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()

    if blured:
        #print("blur size:",blur_kernel_size)
        #blur = transforms.GaussianBlur(blur_kernel_size)
        sigma = (blur_kernel_size-1)/6
        blur = transforms.GaussianBlur(blur_kernel_size,sigma)
    def pseudomodel(images):
        images = images.to(device)
        if blured:
            images = blur(images)
            images = torch.clamp(images,0,1)
        inputs = processor(text=labels, images=images,
                   return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_labels = probs.argmax(dim=1)#np.array(labels)[]

        return predicted_labels, logits_per_image
    return pseudomodel
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)
    
def getCLIPCLassifier_FT(device, labels, blured=False,blur_kernel_size=3):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    if blured:
        #print("blur size:",blur_kernel_size)
        #blur = transforms.GaussianBlur(blur_kernel_size)
        sigma = (blur_kernel_size-1)/6
        blur = transforms.GaussianBlur(blur_kernel_size,sigma)
    def pseudomodel(images):
        images = images.to(device)
        if blured:
            images = blur(images)
            images = torch.clamp(images,0,1)
        inputs = processor(text=labels, images=images,
                   return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_labels = probs.argmax(dim=1)#np.array(labels)[]

        return outputs, predicted_labels, logits_per_image
    return pseudomodel

def getCLIPCLassifier(device, labels, blured=False,blur_kernel_size=3):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    if blured:
        #print("blur size:",blur_kernel_size)
        #blur = transforms.GaussianBlur(blur_kernel_size)
        sigma = (blur_kernel_size-1)/6
        blur = transforms.GaussianBlur(blur_kernel_size,sigma)
    def pseudomodel(images):
        images = images.to(device)
        if blured:
            images = blur(images)
            images = torch.clamp(images,0,1)
        inputs = processor(text=labels, images=images,
                   return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_labels = probs.argmax(dim=1)#np.array(labels)[]

        return outputs, predicted_labels, logits_per_image
    return pseudomodel

def getVIT(device, is_adversarial=False):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    model.config.return_dict=True
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")    

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    def pseudomodel(images):
        if is_adversarial:
            # Forward pass
            model.train()

            outputs = model(
                pixel_values=images,
                decoder_input_ids=tokenizer("<|startoftext|>", return_tensors="pt").input_ids.to(device),
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract logits
            logits = outputs.logits 
            
            """
            out = model.generate(pixel_values=images, return_dict_in_generate=True, output_scores=True, output_logits = True)
            print(len(out.logits))
            print(out.logits[0].requires_grad)
            print(out.logits)
            # Extract logits
            logits = torch.stack(out.scores, dim=1)  # Shape: [batch_size, sequence_length, vocab_size]
            """
            probs = logits.softmax(dim=2)
            return probs
        else:
            pixel_values = images.to(device)
    
            output_ids = model.generate(pixel_values, **gen_kwargs)
        
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            return preds
            
    return pseudomodel

def getPOSClassifier(): 
    tagger = SequenceTagger.load("flair/pos-english")
    to_keep = [
    "NN", "NNS", "NNP", "NNPS",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    ]
    def model(sentence):
        
        sentence = Sentence(sentence)
        # predict NER tags
        tagger.predict(sentence)
        
        pos_list = []
        for word in sentence:
            if word.get_label('pos').value in to_keep:
                pos_list.append(1)
            else:
                pos_list.append(0)

        
        return pos_list
    return model
def getExpNetTokenId(word,tokens_path):
    with open(tokens_path, 'rb') as f:
        coco_tokens = pickle.load(f)
        
        word_idx = coco_tokens['word2idx_dict'][word]
    return word_idx
def getExpNet(model_path,tokens_path,device="cpu",blured = False, blur_kernel_size=3):
    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=512,
                           N_enc=3,
                           N_dec=3,
                           drop_args=drop_args)
    #'/home/yurii.kohan/7_semester/ImageCaptioningProject/models/ExpansionNet_v2/demo_material/demo_coco_tokens.pickle'
    with open(tokens_path, 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    img_size = 384
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,
    
                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=74, drop_args=model_args.drop_args,
                                rank=device).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    beam_search_kwargs = {'beam_size': 5,
                          'beam_max_seq_len': 74,
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': sos_idx,
                          'eos_idx': eos_idx}
    if blured:
        print("blur size:",blur_kernel_size)
        #blur = transforms.GaussianBlur(blur_kernel_size)
        sigma = (blur_kernel_size-1)/6
        blur = transforms.GaussianBlur(blur_kernel_size,sigma)
    def pseudomodel(img, gen_cap=True):
        if blured:
            img = blur(img)
        pred, pred_probs, logits, cross_enc_output = model(enc_x=img,
                    enc_x_num_pads=[0]*img.size(0),
                    mode='beam_search', **beam_search_kwargs)
        if not gen_cap:
            output = pred
        else:
            output_words = [pred[i][0] for i in range(len(pred))]
            output = [tokens2description(p, coco_tokens['idx2word_list'], sos_idx, eos_idx) for p in output_words]
            
        img = img.detach()
        del img
        return output, pred, logits[0], cross_enc_output
        
        '''
        if is_adversarial:
            pred, pred_probs, logits, cross_enc_output = model(enc_x=img,
                        enc_x_num_pads=[0]*img.size(0),
                        mode='beam_search', **beam_search_kwargs)

            img = img.detach()
            del img
            return logits[0], cross_enc_output
        else:
            with torch.no_grad():
                 pred, _, _, _ = model(enc_x=img,
                            enc_x_num_pads=[0]*img.size(0),
                            mode='beam_search', **beam_search_kwargs)
            output_words = [pred[i][0] for i in range(len(pred))]
            #TODO: delete for loop
            output = [tokens2description(p, coco_tokens['idx2word_list'], sos_idx, eos_idx) for p in output_words]
            img = img.detach()
            del img
            return output
        '''
    return pseudomodel

'''    
def getMPLUG(device, is_adversarial=False):
    model_path = 'mPLUG/mPLUG-Owl3-1B-241014'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # model = mPLUGOwl3Model(config).cuda().half()
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)

    messages = [
    {"role": "user", "content": """<|image|>
Describe this image."""},
    {"role": "assistant", "content": ""}
    ]
        
    def pseudomodel(images):
        if is_adversarial:
            inputs = processor(images,"",return_tensors="pt", padding = True).to(device)
            inputs['pixel_values']=images.to(device)
            model.train()
            out = model.forward(**inputs,return_dict=True)
            logits = out['logits']
            #out = model.generate(**inputs,return_dict_in_generate=True,output_scores=True)
            #logits = torch.stack(list(out['scores']),dim=1)  # this is the image-text similarity score   
            probs = logits.softmax(dim=2)
            return probs 
        else:
            inputs = processor(messages, images=images, videos=None)
            inputs.to('cuda')
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens':100,
                'decode_text':True,
            })
            g = model.generate(**inputs)
            return g
    return pseudomodel

def getGRIT(model_path,config,device = "cpu"):
    # initialize hydra config
    
    GlobalHydra.instance().clear()
    
    initialize(config_path=config)
    
    config = compose(config_name='coco_config.yaml', overrides=[f"exp.checkpoint={model_path}"])
    
    detector = build_detector(config).to(device)
    model = Transformer(detector, config=config)
    model = model.to(device)
    
    # load checkpoint
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"det missing:{len(missing)} det unexpected:{len(unexpected)}")
        
    model.cached_features = False
    
    # prepare utils
    transform = get_transform(config.dataset.transform_cfg)['valid']
    text_field = TextField(vocab_path=vocab_path)
    
    # inference and decode
    def pseudomodel(images):
        
        image = transform(images)
        images = nested_tensor_from_tensor_list([image]).to(device)

        with torch.no_grad():
            out, _ = model(
                images,
                seq=None,
                use_beam_search=True,
                max_len=config.model.beam_len,
                eos_idx=config.model.eos_idx,
                beam_size=config.model.beam_size,
                out_size=1,
                return_probs=False,
            )
            return text_field.decode(out, join_words=True)[0]
        return pseudomodel  
'''
    