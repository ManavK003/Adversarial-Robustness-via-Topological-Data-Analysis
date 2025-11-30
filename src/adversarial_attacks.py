import torch
import torch.nn as nn

def denormalize(images, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
    return images * std + mean

def normalize(images, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
    return (images - mean) / std

def fgsm_attack(model, images, labels, epsilon=0.3, mean=(0.1307,), std=(0.3081,)):
    images_denorm = denormalize(images, mean, std)
    images_denorm.requires_grad = True
    
    images_norm = normalize(images_denorm, mean, std)
    outputs = model(images_norm)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    perturbation = epsilon * images_denorm.grad.sign()
    adversarial_images_denorm = images_denorm + perturbation
    adversarial_images_denorm = torch.clamp(adversarial_images_denorm, 0, 1)
    
    adversarial_images_norm = normalize(adversarial_images_denorm, mean, std)
    return adversarial_images_norm.detach()

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40, mean=(0.1307,), std=(0.3081,)):
    images_denorm = denormalize(images, mean, std)
    
    adversarial_images_denorm = images_denorm + torch.empty_like(images_denorm).uniform_(-epsilon, epsilon)
    adversarial_images_denorm = torch.clamp(adversarial_images_denorm, 0, 1)
    
    for i in range(num_iter):
        adversarial_images_denorm.requires_grad = True
        adversarial_images_norm = normalize(adversarial_images_denorm, mean, std)
        
        outputs = model(adversarial_images_norm)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbation = alpha * adversarial_images_denorm.grad.sign()
            adversarial_images_denorm = adversarial_images_denorm + perturbation
            
            eta = torch.clamp(adversarial_images_denorm - images_denorm, -epsilon, epsilon)
            adversarial_images_denorm = torch.clamp(images_denorm + eta, 0, 1)
    
    adversarial_images_norm = normalize(adversarial_images_denorm, mean, std)
    return adversarial_images_norm.detach()

def generate_adversarial_examples(model, data_loader, attack_type='fgsm', epsilon=0.3, device='cuda'):
    model.eval()
    adversarial_data = []
    clean_data = []
    labels_list = []
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        if attack_type == 'fgsm':
            adv_images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == 'pgd':
            adv_images = pgd_attack(model, images, labels, epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        adversarial_data.append(adv_images.cpu())
        clean_data.append(images.cpu())
        labels_list.append(labels.cpu())
    
    return torch.cat(clean_data), torch.cat(adversarial_data), torch.cat(labels_list)