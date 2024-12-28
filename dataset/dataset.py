import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class BeardDataset(Dataset):
    def __init__(self, data_dir, transform=None, size=256, split="train", train_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Create correct pairs without duplicates
        beard_files = [f for f in os.listdir(data_dir) if '_beard.png' in f and 'no_beard' not in f]
        pairs = []
        for beard_file in beard_files:
            idx = beard_file.split('_')[0]
            no_beard_file = f'{idx}_no_beard.png'
            if os.path.exists(os.path.join(data_dir, no_beard_file)):
                pairs.append((int(idx), beard_file, no_beard_file))
        
        pairs.sort(key=lambda x: x[0])
        
        split_idx = int(len(pairs) * train_ratio)
        if split == "train":
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, no_beard_file, beard_file = self.pairs[idx]
        
        beard_path = os.path.join(self.data_dir, beard_file)
        no_beard_path = os.path.join(self.data_dir, no_beard_file)
        
        beard_img = Image.open(beard_path).convert('RGB')
        no_beard_img = Image.open(no_beard_path).convert('RGB')
        
        return {
            'input': self.transform(no_beard_img),
            'target': self.transform(beard_img),
            'index': self.pairs[idx][0]
        }