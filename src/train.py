import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_standard(model, train_loader, test_loader, device='cuda', epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader), 'acc': 100.*correct/total})
        
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return history

def train_adversarial(model, train_loader, test_loader, device='cuda', epochs=10, lr=0.001, epsilon=0.3):
    from .adversarial_attacks import fgsm_attack
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Adversarial Training]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            adv_images = fgsm_attack(model, images, labels, epsilon)
            
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader), 'acc': 100.*correct/total})
        
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return history

def evaluate(model, data_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def save_checkpoint(model, optimizer, epoch, path='results/checkpoints/checkpoint.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')

def load_checkpoint(model, optimizer=None, path='results/checkpoints/checkpoint.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from {path} (epoch {epoch})')
        return epoch
    else:
        print(f'No checkpoint found at {path}')
        return 0