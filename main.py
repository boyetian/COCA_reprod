import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from models.coca import COCA
from data.imagenet_c import ImageNetC
from scripts.test_accuracy import test_accuracy
from models.resnet import resnet50
from models.vit import vit_base_patch16_224
from utils.augmentations import get_transform

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
    anchor_model_name = 'vit_base_patch16_224'
    aux_model_name = 'resnet50'
    anchor_model = vit_base_patch16_224(pretrained=True)
    aux_model = resnet50(pretrained=True)

    # Setup COCA
    coca = COCA(anchor_model, aux_model, lr_anchor=args.lr_anchor, lr_aux=args.lr_aux, momentum=args.momentum)

    # Data loading
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)
    
    dataset = ImageNetC(root=args.data_root, corruption_type=args.corruption, severity=args.severity, transform_anchor=transform_anchor, transform_aux=transform_aux)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    # Training loop (Test-Time Adaptation)
    for i, (images_anchor, images_aux, _) in enumerate(data_loader):
        if torch.cuda.is_available():
            images_anchor = images_anchor.cuda()
            images_aux = images_aux.cuda()
        
        coca.update(images_anchor, images_aux)
        if (i+1) % 10 == 0:
            print(f'Adapted on batch {i+1}/{len(data_loader)}')

    # Evaluation
    accuracy = test_accuracy(coca, args.data_root, args.batch_size, args.workers, args.corruption, args.severity)
    print(f'Accuracy on {args.corruption} (severity {args.severity}): {accuracy:.2f}%')

if __name__ == '__main__':
    main()
