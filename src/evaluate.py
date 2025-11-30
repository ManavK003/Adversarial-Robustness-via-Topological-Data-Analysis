import torch
from tqdm import tqdm
from .adversarial_attacks import fgsm_attack, pgd_attack

def evaluate_robustness(model, data_loader, device='cuda', epsilon=0.3):
    model.eval()
    
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc='Evaluating Robustness'):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_pred = clean_outputs.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()
        
        fgsm_images = fgsm_attack(model, images, labels, epsilon)
        with torch.no_grad():
            fgsm_outputs = model(fgsm_images)
            _, fgsm_pred = fgsm_outputs.max(1)
            fgsm_correct += fgsm_pred.eq(labels).sum().item()
        
        pgd_images = pgd_attack(model, images, labels, epsilon, num_iter=20)
        with torch.no_grad():
            pgd_outputs = model(pgd_images)
            _, pgd_pred = pgd_outputs.max(1)
            pgd_correct += pgd_pred.eq(labels).sum().item()
        
        total += labels.size(0)
    
    results = {
        'clean_accuracy': 100. * clean_correct / total,
        'fgsm_accuracy': 100. * fgsm_correct / total,
        'pgd_accuracy': 100. * pgd_correct / total
    }
    
    return results

def evaluate_multiple_epsilons(model, data_loader, device='cuda', epsilons=[0.0, 0.1, 0.2, 0.3]):
    results = {}
    
    for eps in epsilons:
        print(f'\nEvaluating at epsilon={eps}')
        res = evaluate_robustness(model, data_loader, device, eps)
        results[eps] = res
        print(f"Clean: {res['clean_accuracy']:.2f}%, FGSM: {res['fgsm_accuracy']:.2f}%, PGD: {res['pgd_accuracy']:.2f}%")
    
    return results

def compare_models(standard_model, robust_model, data_loader, device='cuda', epsilon=0.3):
    print("\n=== Standard Model ===")
    standard_results = evaluate_robustness(standard_model, data_loader, device, epsilon)
    
    print("\n=== Robust Model ===")
    robust_results = evaluate_robustness(robust_model, data_loader, device, epsilon)
    
    comparison = {
        'standard': standard_results,
        'robust': robust_results
    }
    
    print("\n=== Comparison ===")
    print(f"Clean Accuracy - Standard: {standard_results['clean_accuracy']:.2f}%, Robust: {robust_results['clean_accuracy']:.2f}%")
    print(f"FGSM Accuracy - Standard: {standard_results['fgsm_accuracy']:.2f}%, Robust: {robust_results['fgsm_accuracy']:.2f}%")
    print(f"PGD Accuracy - Standard: {standard_results['pgd_accuracy']:.2f}%, Robust: {robust_results['pgd_accuracy']:.2f}%")
    
    return comparison