import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import json
import os
from datetime import datetime
import yaml
from models.coca import COCA, get_model
from data.imagenet_c import ImageNetC
from scripts.test_accuracy import test_accuracy
from utils.augmentations import get_transform

def main():
    parser = argparse.ArgumentParser(description='COCA Test-Time Adaptation')
    parser.add_argument('--config', type=str, default='configs/resnet50_vit_base.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None, help='Path to ImageNet-C dataset')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training and testing')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--corruption', type=str, default='gaussian_noise', help='Type of corruption to test, or "all" to test all')
    parser.add_argument('--severity', type=int, default=5, help='Severity of corruption')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override args with config values if not provided in command line
    if args.data_root is None:
        args.data_root = config['dataset']['path']
    if args.batch_size is None:
        args.batch_size = config['dataset']['batch_size']
    
    if args.corruption == 'all':
        corruption_types = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    else:
        corruption_types = [args.corruption]

    results = {}
    for corruption_type in corruption_types:
        print(f"--- Testing corruption: {corruption_type} severity: {args.severity} ---")
        accuracy = run_test(args, config, corruption_type)
        results[corruption_type] = accuracy

    save_results(args, config, results)

def run_test(args, config, corruption_type):
    anchor_model_config = config['model']['large_model']
    aux_model_config = config['model']['small_model']
    
    lr_anchor = anchor_model_config.get('lr', 0.001) # Default lr
    lr_aux = aux_model_config.get('lr', 0.00025) # Default lr

    # Load models
    anchor_model_name = anchor_model_config['name']
    aux_model_name = aux_model_config['name']
    anchor_model = get_model(anchor_model_name, pretrained=anchor_model_config['pretrained'])
    aux_model = get_model(aux_model_name, pretrained=aux_model_config['pretrained'])

    # Setup COCA
    coca = COCA(anchor_model, aux_model, lr_anchor=lr_anchor, lr_aux=lr_aux, momentum=args.momentum)

    # Data loading
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)
    
    dataset = ImageNetC(root=args.data_root, corruption_type=corruption_type, severity=args.severity, transform_anchor=transform_anchor, transform_aux=transform_aux)
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
    accuracy = test_accuracy(coca, args.data_root, args.batch_size, args.workers, corruption_type, args.severity, anchor_model_name=anchor_model_name, aux_model_name=aux_model_name)
    print(f'Accuracy on {corruption_type} (severity {args.severity}): {accuracy:.2f}%')
    return accuracy

def save_results(args, config, results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    anchor_model_name = config['model']['large_model']['name']
    aux_model_name = config['model']['small_model']['name']
    lr_anchor = config['model']['large_model'].get('lr', 0.001)
    lr_aux = config['model']['small_model'].get('lr', 0.00025)

    result_data = {
        'timestamp': timestamp,
        'models': {
            'anchor': anchor_model_name,
            'auxiliary': aux_model_name,
        },
        'dataset': config['dataset']['name'],
        'severity': args.severity,
        'hyperparameters': {
            'lr_anchor': lr_anchor,
            'lr_aux': lr_aux,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
        },
        'results': {corr: f"{acc:.2f}%" for corr, acc in results.items()}
    }

    if len(results) > 1:
        avg_accuracy = sum(results.values()) / len(results)
        result_data['average_accuracy'] = f"{avg_accuracy:.2f}%"
        corruption_name = "all_corruptions"
    else:
        corruption_name = args.corruption

    filename = f"result_{timestamp}_{anchor_model_name}_{aux_model_name}_{corruption_name}_sev{args.severity}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)
    
    print(f"Results saved to {filepath}")


if __name__ == '__main__':
    main()
