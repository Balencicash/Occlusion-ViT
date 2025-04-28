import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class LFWDatasetPair(Dataset):
    def __init__(self, root_dir, pairs_file='pairs.txt', transform=None, occlusion=None):
        self.root_dir = root_dir
        self.transform = transform
        self.occlusion = occlusion
        self.pairs = []

        with open(os.path.join(root_dir, pairs_file), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    name, idx1, idx2 = parts
                    img1_path = os.path.join(root_dir, name, f"{name}_{int(idx1):04d}.jpg")
                    img2_path = os.path.join(root_dir, name, f"{name}_{int(idx2):04d}.jpg")
                    label = 1
                else:
                    name1, idx1, name2, idx2 = parts
                    img1_path = os.path.join(root_dir, name1, f"{name1}_{int(idx1):04d}.jpg")
                    img2_path = os.path.join(root_dir, name2, f"{name2}_{int(idx2):04d}.jpg")
                    label = 0
                self.pairs.append((img1_path, img2_path, label))

    def add_occlusion(self, img):
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        if self.occlusion == 'eye':
            # black rectangle for covering eyes
            left = w * 0.3
            right = w * 0.7
            top = h * 0.40
            bottom = h * 0.55
        elif self.occlusion == 'mouth':
            left = w * 0.35
            right = w * 0.65
            top = h * 0.55
            bottom = h * 0.7
        else:
            return img
        
        draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.occlusion:
            img2 = self.add_occlusion(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
