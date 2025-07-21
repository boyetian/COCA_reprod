import argparse
import yaml
import torch
import torch.optim as optim
from data.imagenet_c import get_imagenet_c_loader
from models.coca import COCA
from models.vit import get_vit, get_mobilevit
from models.resnet import get_resnet
from utils.metrics import accuracy
import os

def main():
    parser = argparse.ArgumentParser(description='COCA Test-Time Adaptation')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create models
    if 'vit' in config['model']['large_model']['name']:
        large_model = get_vit(config['model']['large_model']['name'], config['model']['large_model']['pretrained'])
    else:
        large_model = get_resnet(config['model']['large_model']['name'], config['model']['large_model']['pretrained'])

    if 'mobilevit' in config['model']['small_model']['name']:
        small_model = get_mobilevit(config['model']['small_model']['name'], config['model']['small_model']['pretrained'])
    else:
        small_model = get_resnet(config['model']['small_model']['name'], config['model']['small_model']['pretrained'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    large_model.to(device)
    small_model.to(device)

    # Create COCA model
    coca_model = COCA(large_model, small_model,
                      config['coca']['lambda_co_adaptation'],
                      config['coca']['lambda_self_adaptation'])
    coca_model.to(device)

    # Create optimizer
    optimizer = optim.Adam(coca_model.parameters(), lr=config['optimizer']['lr'])

    # Get ImageNet-C loaders for all corruptions and severities
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
        'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
        'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur'
    ]
    severities = [1, 2, 3, 4, 5]

    for corruption in corruptions:
        for severity in severities:
            print(f"Testing on {corruption}, severity {severity}")
            loader = get_imagenet_c_loader(config['dataset']['path'], corruption, severity,
                                            config['dataset']['batch_size'])

            coca_model.eval()
            total_acc = 0
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)

                # Adapt the model
                coca_model.train()
                optimizer.zero_grad()
                _, loss = coca_model(images)
                loss.backward()
                optimizer.step()

                # Evaluate the adapted model
                coca_model.eval()
                with torch.no_grad():
                    logits, _ = coca_model(images)
                    acc = accuracy(logits, labels)
                    total_acc += acc

            print(f"Accuracy: {total_acc / len(loader)}")


if __name__ == '__main__':
    main() 