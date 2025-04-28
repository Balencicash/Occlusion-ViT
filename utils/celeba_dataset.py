import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None, selected_ids=None, include_occlusions=False):
        self.img_dir = img_dir
        self.transform = transform
        self.include_occlusions = include_occlusions
        self.image_paths = []
        self.labels = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        id_map = {}
        current_id = 0

        for line in lines:
            img_name, identity = line.strip().split()
            identity = int(identity)
            if selected_ids and identity > selected_ids:
                continue
            if identity not in id_map:
                id_map[identity] = current_id
                current_id += 1

            label = id_map[identity]
            self.image_paths.append(os.path.join(img_dir, img_name))
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths) * (3 if self.include_occlusions else 1)

    def __getitem__(self, idx):
        if self.include_occlusions:
            img_idx = idx // 3
            mode = idx % 3
        else:
            img_idx = idx
            mode = 0  # no occlusion

        img_path = self.image_paths[img_idx]
        label = self.labels[img_idx]
        image = Image.open(img_path).convert("RGB")

        if self.include_occlusions:
            if mode == 1:
                image = self.add_occlusion(image, "eye")
            elif mode == 2:
                image = self.add_occlusion(image, "mouth")

        if self.transform:
            image = self.transform(image)

        return image, label

    def add_occlusion(self, img, region="eye"):
        draw = ImageDraw.Draw(img)
        w, h = img.size

        if region == "eye":
            left = w * 0.25
            right = w * 0.75
            top = h * 0.40
            bottom = h * 0.55
        elif region == "mouth":
            left = w * 0.3
            right = w * 0.6
            top = h * 0.6
            bottom = h * 0.75
        else:
            return img

        draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
        return img
