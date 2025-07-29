import torch
from torchvision import transforms
from data.imagenet_c import ImageNetC
from utils.augmentations import get_transform

def test_accuracy(model, data_root, batch_size, workers, corruption, severity, anchor_model_name='vit_base_patch16_224', aux_model_name='resnet50'):
    """
    Calculates the accuracy of the model on a given ImageNet-C corruption.
    """
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)
    
    dataset = ImageNetC(root=data_root, corruption_type=corruption, severity=severity, transform_anchor=transform_anchor, transform_aux=transform_aux)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    
    model.anchor_model.eval()
    model.aux_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images_anchor, _, labels in data_loader:
            if torch.cuda.is_available():
                images_anchor, labels = images_anchor.cuda(), labels.cuda()
            
            outputs = model.anchor_model(images_anchor)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total
