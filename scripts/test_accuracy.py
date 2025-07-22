import argparse
import yaml
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.imagenet_c import get_imagenet_c_loader
from models.coca import COCA
from models.vit import get_vit, get_mobilevit
from models.resnet import get_resnet
from utils.metrics import accuracy

def test_accuracy():
    parser = argparse.ArgumentParser(description='COCA Accuracy Test')
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
    coca_model.eval()

    # Define corruption categories
    corruption_categories = {
        'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'weather': ['frost', 'snow', 'fog', 'brightness'],
        'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    }
    all_corruptions = [corr for cat in corruption_categories.values() for corr in cat]
    severities = [1, 2, 3, 4, 5]

    category_accuracies = {cat: [] for cat in corruption_categories.keys()}

    for corruption in all_corruptions:
        for severity in severities:
            print(f"Testing on {corruption}, severity {severity}")
            loader = get_imagenet_c_loader(config['dataset']['path'], corruption, severity,
                                            config['dataset']['batch_size'])

            total_acc = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(loader):
                    images, labels = images.to(device), labels.to(device)
                    logits, _ = coca_model(images)
                    acc = accuracy(logits, labels)
                    total_acc += acc
            
            avg_acc = total_acc / len(loader)
            print(f"Accuracy: {avg_acc}")

            for category, corruptions_in_category in corruption_categories.items():
                if corruption in corruptions_in_category:
                    category_accuracies[category].append(avg_acc)

    print("\n--- Corruption Category Accuracies ---")
    for category, accuracies in category_accuracies.items():
        if accuracies:
            avg_cat_acc = sum(accuracies) / len(accuracies)
            print(f"{category.capitalize()} Accuracy: {avg_cat_acc}")

if __name__ == '__main__':
    test_accuracy()
