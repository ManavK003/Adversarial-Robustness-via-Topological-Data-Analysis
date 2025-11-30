import torch
import argparse
import os
from src.data_loader import get_data_loaders
from src.model import get_model
from src.train import train_standard, train_adversarial, save_checkpoint
from src.evaluate import evaluate_robustness, evaluate_multiple_epsilons, compare_models
from src.visualize import plot_training_history, plot_adversarial_examples, plot_robustness_comparison

def main():
    parser = argparse.ArgumentParser(description='Adversarial Robustness via TDA')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'evaluate', 'visualize', 'all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=0.3)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size)
    print(f'Loaded {args.dataset} dataset')
    
    if args.mode in ['train', 'all']:
        print('\n=== Training Standard Model ===')
        standard_model = get_model(args.dataset, device)
        standard_history = train_standard(standard_model, train_loader, test_loader, device, args.epochs, args.lr)
        save_checkpoint(standard_model, None, args.epochs, f'results/checkpoints/{args.dataset}_standard.pth')
        plot_training_history(standard_history, f'results/figures/{args.dataset}_standard_history.png')
        
        print('\n=== Training Adversarially Robust Model ===')
        robust_model = get_model(args.dataset, device)
        robust_history = train_adversarial(robust_model, train_loader, test_loader, device, args.epochs, args.lr, args.epsilon)
        save_checkpoint(robust_model, None, args.epochs, f'results/checkpoints/{args.dataset}_robust.pth')
        plot_training_history(robust_history, f'results/figures/{args.dataset}_robust_history.png')
    
    if args.mode in ['evaluate', 'all']:
        print('\n=== Evaluating Models ===')
        
        standard_model = get_model(args.dataset, device)
        robust_model = get_model(args.dataset, device)
        
        standard_path = f'results/checkpoints/{args.dataset}_standard.pth'
        robust_path = f'results/checkpoints/{args.dataset}_robust.pth'
        
        if not os.path.exists(standard_path) or not os.path.exists(robust_path):
            print(f'\n⚠️  WARNING: Trained models not found!')
            print(f'Standard model: {standard_path} - {"✓ exists" if os.path.exists(standard_path) else "✗ missing"}')
            print(f'Robust model: {robust_path} - {"✓ exists" if os.path.exists(robust_path) else "✗ missing"}')
            print(f'\nPlease run training first: python main.py --mode train --dataset {args.dataset}')
            return
        
        from src.train import load_checkpoint
        load_checkpoint(standard_model, path=standard_path)
        load_checkpoint(robust_model, path=robust_path)
        
        comparison = compare_models(standard_model, robust_model, test_loader, device, args.epsilon)
        
        print('\n=== Evaluating Across Multiple Epsilons ===')
        epsilons = [0.0, 0.1, 0.2, 0.3]
        
        print('\nStandard Model:')
        standard_results = evaluate_multiple_epsilons(standard_model, test_loader, device, epsilons)
        
        print('\nRobust Model:')
        robust_results = evaluate_multiple_epsilons(robust_model, test_loader, device, epsilons)
        
        plot_robustness_comparison(standard_results, f'results/figures/{args.dataset}_standard_robustness.png')
        plot_robustness_comparison(robust_results, f'results/figures/{args.dataset}_robust_robustness.png')
    
    if args.mode in ['visualize', 'all']:
        print('\n=== Generating Visualizations ===')
        
        model = get_model(args.dataset, device)
        model_path = f'results/checkpoints/{args.dataset}_robust.pth'
        
        if os.path.exists(model_path):
            from src.train import load_checkpoint
            load_checkpoint(model, path=model_path)
        else:
            print(f'⚠️  Warning: No trained model found at {model_path}')
            print(f'Using untrained model for visualization (results will be random)')
        
        plot_adversarial_examples(model, test_loader, device, args.epsilon, f'results/figures/{args.dataset}_adversarial_examples.png')
    
    print('\n=== Complete! ===')
    print(f'Results saved to results/')
    print(f'Checkpoints saved to results/checkpoints/')
    print(f'Figures saved to results/figures/')

if __name__ == '__main__':
    main()