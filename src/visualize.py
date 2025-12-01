import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from .topology import compute_persistence
from .adversarial_attacks import fgsm_attack, denormalize

def plot_training_history(history, save_path='results/figures/training_history.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history saved to {save_path}')
    plt.close()

def plot_adversarial_examples(model, data_loader, device='cuda', epsilon=0.3, save_path='results/figures/adversarial_examples.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mean = (0.1307,)
    std = (0.3081,)
    
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images[:8].to(device), labels[:8].to(device)
    
    adv_images = fgsm_attack(model, images, labels, epsilon, mean, std)
    
    with torch.no_grad():
        clean_outputs = model(images)
        adv_outputs = model(adv_images)
    
    _, clean_pred = clean_outputs.max(1)
    _, adv_pred = adv_outputs.max(1)
    
    images_denorm = denormalize(images, mean, std)
    adv_images_denorm = denormalize(adv_images, mean, std)
    
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    for i in range(8):
        if images.shape[1] == 1:
            axes[0, i].imshow(images_denorm[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].imshow(adv_images_denorm[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[2, i].imshow((adv_images_denorm[i] - images_denorm[i]).cpu().squeeze(), cmap='seismic', vmin=-epsilon, vmax=epsilon)
        else:
            axes[0, i].imshow(torch.clamp(images_denorm[i].cpu().permute(1, 2, 0), 0, 1).numpy())
            axes[1, i].imshow(torch.clamp(adv_images_denorm[i].cpu().permute(1, 2, 0), 0, 1).numpy())
            diff = (adv_images_denorm[i] - images_denorm[i]).cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(diff, vmin=-epsilon, vmax=epsilon)
        
        axes[0, i].set_title(f'Clean: {clean_pred[i].item()}')
        axes[1, i].set_title(f'Adv: {adv_pred[i].item()}')
        axes[2, i].set_title('Perturbation')
        
        for ax in axes[:, i]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Adversarial examples saved to {save_path}')
    plt.close()

def plot_persistence_diagram(dgms, save_path='results/figures/persistence_diagram.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for dim, dgm in enumerate(dgms):
        if len(dgm) > 0:
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            ax.scatter(births, deaths, alpha=0.5, label=f'H{dim}')
    
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title('Persistence Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Persistence diagram saved to {save_path}')
    plt.close()

def plot_robustness_comparison(results, save_path='results/figures/robustness_comparison.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epsilons = list(results.keys())
    clean_acc = [results[eps]['clean_accuracy'] for eps in epsilons]
    fgsm_acc = [results[eps]['fgsm_accuracy'] for eps in epsilons]
    pgd_acc = [results[eps]['pgd_accuracy'] for eps in epsilons]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epsilons, clean_acc, 'o-', label='Clean', linewidth=2)
    ax.plot(epsilons, fgsm_acc, 's-', label='FGSM', linewidth=2)
    ax.plot(epsilons, pgd_acc, '^-', label='PGD', linewidth=2)
    
    ax.set_xlabel('Epsilon', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness vs Perturbation Strength', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Robustness comparison saved to {save_path}')
    plt.close()
    