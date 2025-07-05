import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

from PIL import Image as PIL_Image

import argparse

from pycocotools.coco import COCO

import os
import shutil
import urllib.request
import random
import json

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, all_annotations_file, num_images=100, img_dir=None, subset_annotations_file=None, model_type = "ExpansionNet", seed=42, img_size = 384):
        self.annotations_file = all_annotations_file
        self.subset_annotations_file = subset_annotations_file
        self.num_images = num_images
        self.img_dir = img_dir
        
        self.seed = seed
        
        self.model_type = model_type
        
        self.img_size = img_size
        #self.perturbations = [1,2,3,4,5,5,5]
        """
        # Remove the existing files, if any
        if os.path.exists(subset_captions_path):
            shutil.rmtree(subset_captions_path)
        if os.path.exists(subset_images_path):
            shutil.rmtree(subset_images_path)
        """
        if subset_annotations_file is not None:
            return
        else:
            self.subset_annotations_file = os.path.join(img_dir,'new_annotations.json')
            
        self.coco = COCO(all_annotations_file)
        
        # Get all image IDs
        all_img_ids = self.coco.getImgIds()
        random.seed(seed)
        self.subset_img_ids = random.sample(all_img_ids, num_images)
        print(f"Selected {len(self.subset_img_ids)} images.")

        # Download the images
        os.makedirs(img_dir, exist_ok=True)
        print("Downloading images")
        for img_id in self.subset_img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_url = img_info['coco_url']
            img_filename = os.path.join(self.img_dir, img_info['file_name'])
            #print(".")
            #print(f"Downloading {img_info['file_name']}...")
            
            if not os.path.exists(img_filename):
                urllib.request.urlretrieve(img_url, img_filename)
        
        # Get annotation IDs and load annotations
        '''
        subset_ann_ids = self.coco.getAnnIds(imgIds=self.subset_img_ids)
        subset_annotations = self.coco.loadAnns(subset_ann_ids)  
            
        with open(self.subset_annotations_file, 'w') as f:
            json.dump(subset_annotations, f, indent=4)
        print(f"Subset annotations saved to {self.subset_annotations_file}")
        '''
    def __len__(self):
        return self.num_images#*len(self.perturbations)

    def __getitem__(self, idx):
        #perturbation = self.perturbations[idx%len(self.perturbations)]
        
        img_id = self.subset_img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        pil_image = PIL_Image.open(img_path)
        if pil_image.mode != 'RGB':
            pil_image = PIL_Image.new("RGB", pil_image.size)

        if self.model_type == "ExpansionNet":
            img_size = self.img_size

            transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
            #transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

            preprocess_pil_image = transf_1(pil_image)
            image = torchvision.transforms.ToTensor()(preprocess_pil_image)
            #image = transf_2(image)
            #image = image.unsqueeze(0)
        else:
            image = torchvision.transforms.ToTensor()(pil_image)

        ann_id = self.coco.getAnnIds(imgIds=img_id)
        label = self.coco.loadAnns(ann_id)
        label = list(map(lambda x: x.get("caption"), label))
        return image, label


