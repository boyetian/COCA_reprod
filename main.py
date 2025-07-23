import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from models.coca import COCA
from data.imagenet_c import ImageNetC
from scripts.test_accuracy import test_accuracy
from models.resnet import resnet50
from models.vit import vit_base_patch16_224

def main():
    parser = argparse.ArgumentParser(description='COCA Test-Time Adaptation')
    parser.add_argument('--config', type=str, default='configs/resnet50_vit_base.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default='./data/Tiny-ImageNet-C', help='Path to ImageNet-C dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--corruption', type=str, default='gaussian_noise', help='Type of corruption to test')
    parser.add_argument('--severity', type=int, default=5, help='Severity of corruption')
    parser.add_argument('--lr_anchor', type=float, default=0.00025, help='Learning rate for anchor model')
    parser.add_argument('--lr_aux', type=float, default=0.001, help='Learning rate for auxiliary model')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    args = parser.parse_args()

    # Load models
    anchor_model = vit_base_patch16_224(pretrained=True)
    aux_model = resnet50(pretrained=True)

    # Setup COCA
    coca = COCA(anchor_model, aux_model, lr_anchor=args.lr_anchor, lr_aux=args.lr_aux, momentum=args.momentum)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageNetC(root=args.data_root, corruption_type=args.corruption, severity=args.severity, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    # Training loop (Test-Time Adaptation)
    for i, (images, _) in enumerate(data_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        
        coca.update(images)
        if (i+1) % 10 == 0:
            print(f'Adapted on batch {i+1}/{len(data_loader)}')

    # Evaluation
    accuracy = test_accuracy(coca, args.data_root, args.batch_size, args.workers, args.corruption, args.severity)
    print(f'Accuracy on {args.corruption} (severity {args.severity}): {accuracy:.2f}%')

if __name__ == '__main__':
    main()
