import torch
from torchvision import transforms
from data.imagenet_c import ImageNetC

def test_accuracy(model, data_root, batch_size, workers, corruption, severity):
    """
    Calculates the accuracy of the model on a given ImageNet-C corruption.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageNetC(root=data_root, corruption_type=corruption, severity=severity, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    
    model.anchor_model.eval()
    model.aux_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total
