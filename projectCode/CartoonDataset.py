import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from PIL import Image

class CartoonDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transform=None, img_size = 384):
        self.dataset = load_dataset("Norod78/cartoon-blip-captions", split=split)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        image = self.transform(image)
        return image, self.dataset[idx]["text"]

def get_cartoon_dataloader(batch_size=32, shuffle=True, split="train"):
    dataset = CartoonDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
