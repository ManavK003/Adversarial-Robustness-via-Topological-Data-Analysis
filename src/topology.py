import torch
import numpy as np
from ripser import ripser
from persim import wasserstein

def compute_persistence(features, max_dim=1):
    features_np = features.detach().cpu().numpy()
    if len(features_np.shape) > 2:
        features_np = features_np.reshape(features_np.shape[0], -1)
    
    result = ripser(features_np, maxdim=max_dim)
    return result['dgms']

def wasserstein_distance(dgm1, dgm2, dim=0):
    if len(dgm1[dim]) == 0 or len(dgm2[dim]) == 0:
        return 0.0
    return wasserstein(dgm1[dim], dgm2[dim])

def topology_preserving_loss(model, clean_features, adv_features, weight=0.1):
    clean_dgm = compute_persistence(clean_features, max_dim=0)
    adv_dgm = compute_persistence(adv_features, max_dim=0)
    
    w_dist = wasserstein_distance(clean_dgm, adv_dgm, dim=0)
    
    return weight * w_dist

def extract_features(model, images, layer_name='conv2'):
    features = {}
    
    def hook_fn(module, input, output):
        features['output'] = output
    
    if hasattr(model, layer_name):
        handle = getattr(model, layer_name).register_forward_hook(hook_fn)
    else:
        handle = list(model.children())[-2].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(images)
    
    handle.remove()
    
    return features['output']

def compute_topology_metrics(model, clean_loader, adv_loader, device='cuda', num_batches=10):
    model.eval()
    distances = []
    
    clean_iter = iter(clean_loader)
    adv_iter = iter(adv_loader)
    
    for _ in range(min(num_batches, len(clean_loader))):
        try:
            clean_images, _ = next(clean_iter)
            adv_images, _ = next(adv_iter)
            
            clean_images = clean_images.to(device)
            adv_images = adv_images.to(device)
            
            clean_features = extract_features(model, clean_images)
            adv_features = extract_features(model, adv_images)
            
            clean_dgm = compute_persistence(clean_features[:32])
            adv_dgm = compute_persistence(adv_features[:32])
            
            w_dist = wasserstein_distance(clean_dgm, adv_dgm, dim=0)
            distances.append(w_dist)
        except StopIteration:
            break
    
    return np.mean(distances) if distances else 0.0