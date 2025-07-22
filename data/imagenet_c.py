import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from nltk.corpus import wordnet as wn
import json

class ImageNetC(Dataset):
    def __init__(self, data_dir, corruption_type, severity, transform=None, dictionary=None):
        self.data_dir = os.path.join(data_dir, corruption_type, str(severity))
        self.transform = transform
        self.images = []
        self.labels = []
        if dictionary is None:
            self.dictionary = json.load(open('data/imagenet_class_index.json'))

        for class_folder in sorted(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_folder)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    synset = wn.synset_from_pos_and_offset('n', int(class_folder[1:]))
                    class_label = synset.lemmas()[0].name()
                    if class_label in self.dictionary:
                        class_label = self.dictionary[class_label]
                    else:
                        class_label = -1
                    self.labels.append(class_label)
                    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_imagenet_c_loader(data_dir, corruption_type, severity, batch_size, num_workers=1):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageNetC(data_dir, corruption_type, severity, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader 