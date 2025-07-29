import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from utils.augmentations import get_transform

class ImageNetC(Dataset):
    def __init__(self, root, corruption_type, severity, transform_anchor=None, transform_aux=None, single_model=False):
        self.root = os.path.join(root, corruption_type, str(severity))
        self.transform_anchor = transform_anchor
        self.transform_aux = transform_aux
        self.single_model = single_model
        self.image_paths = []
        self.labels = []
        
        # Create a reverse mapping from synset ID to class index
        dictionary = json.load(open('data/imagenet_class_index.json'))
        self.synset_to_class = {v[0]: int(k) for k, v in dictionary.items()}

        for class_folder in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    
                    # Use the folder name (synset ID) to get the class label
                    class_label = self.synset_to_class.get(class_folder, -1)
                    self.labels.append(class_label)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        img_anchor = self.transform_anchor(image) if self.transform_anchor else image
        
        if self.single_model:
            return img_anchor, label

        img_aux = self.transform_aux(image) if self.transform_aux else image
            
        return img_anchor, img_aux, label

def get_imagenet_c_loader(data_dir, corruption_type, severity, batch_size, num_workers=8, anchor_model_name='vit_base_patch16_224', aux_model_name=None):
    transform_anchor = get_transform(anchor_model_name)
    
    single_model = aux_model_name is None
    transform_aux = None
    if not single_model:
        transform_aux = get_transform(aux_model_name)

    dataset = ImageNetC(data_dir, corruption_type, severity, transform_anchor=transform_anchor, transform_aux=transform_aux, single_model=single_model)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader