import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from nltk.corpus import wordnet as wn
import json
from utils.augmentations import get_transform

class ImageNetC(Dataset):
    def __init__(self, root, corruption_type, severity, transform_anchor=None, transform_aux=None):
        self.root = os.path.join(root, corruption_type, str(severity))
        self.transform_anchor = transform_anchor
        self.transform_aux = transform_aux
        self.image_paths = []
        self.labels = []
        self.dictionary = json.load(open('data/imagenet_class_index.json'))

        for class_folder in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    synset = wn.synset_from_pos_and_offset('n', int(class_folder[1:]))
                    class_label = synset.lemmas()[0].name()
                    if class_label in self.dictionary:
                        class_label = self.dictionary[class_label]
                    else:
                        class_label = -1
                    self.labels.append(class_label)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        img_anchor = self.transform_anchor(image) if self.transform_anchor else image
        img_aux = self.transform_aux(image) if self.transform_aux else image
            
        return img_anchor, img_aux, label

def get_imagenet_c_loader(data_dir, corruption_type, severity, batch_size, num_workers=2, anchor_model_name='vit_base_patch16_224', aux_model_name='resnet50'):
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)

    dataset = ImageNetC(data_dir, corruption_type, severity, transform_anchor=transform_anchor, transform_aux=transform_aux)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader