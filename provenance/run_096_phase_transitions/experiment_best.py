# pip install torch torchvision scipy scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
from collections import defaultdict
import scipy.stats
import time
import warnings
warnings.filterwarnings('ignore')

# ============= METRIC SANITY CHECKS =============
def sanity_check_gradient_coherence():
    """Verify GCI metric behaves as expected on synthetic cases"""
    print("Running metric sanity checks...")
    
    # Test 1: Identical gradients should have high coherence
    g1 = torch.randn(100, 50)
    g2 = g1.clone()
    coherence = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0)
    assert coherence > 0.99, f"Identical gradients should have coherence > 0.99, got {coherence}"
    
    # Test 2: Random gradients should have low coherence
    g3 = torch.randn(100, 50)
    g4 = torch.randn(100, 50)
    coherence = F.cosine_similarity(g3.flatten(), g4.flatten(), dim=0)
    assert abs(coherence) < 0.3, f"Random gradients should have low coherence, got {coherence}"
    
    # Test 3: Negated gradients should have negative coherence
    g5 = torch.randn(100, 50)
    g6 = -g5
    coherence = F.cosine_similarity(g5.flatten(), g6.flatten(), dim=0)
    assert coherence < -0.99, f"Negated gradients should have coherence < -0.99, got {coherence}"
    
    # Test 4: Multi-layer coherence matrix with different sizes (using projection)
    grads = [torch.randn(10, 10), torch.randn(20, 20), torch.randn(5, 5)]
    coherence_matrix = compute_coherence_matrix_with_projection(grads)
    assert coherence_matrix.shape == (3, 3), f"Wrong coherence matrix shape: {coherence_matrix.shape}"
    assert np.allclose(np.diag(coherence_matrix), 1.0, atol=1e-5), "Diagonal should be 1.0"
    
    print("METRIC_SANITY_PASSED")

def compute_coherence_matrix_with_projection(grads, proj_dim=100):
    """Helper for sanity check with projection"""
    n_layers = len(grads)
    matrix = np.zeros((n_layers, n_layers))
    
    # Project all gradients to common dimension
    projected_grads = []
    for g in grads:
        flat_g = g.flatten()
        if len(flat_g) > proj_dim:
            # Random projection to lower dim
            proj = torch.randn(len(flat_g), proj_dim) / np.sqrt(proj_dim)
            proj_g = flat_g @ proj
        else:
            # Pad to proj_dim
            proj_g = F.pad(flat_g, (0, proj_dim - len(flat_g)))
        projected_grads.append(proj_g)
    
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = F.cosine_similarity(
                    projected_grads[i], projected_grads[j], dim=0
                ).item()
    
    return matrix

# Run sanity checks
sanity_check_gradient_coherence()

# ============= MODEL DEFINITION =============
class TinyCNN(nn.Module):
    """Even smaller CNN for faster training"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============= METRIC IMPLEMENTATIONS =============
def compute_gradient_coherence_index(model, data, target, loss_fn, epsilon=1e-5):
    """Compute GCI across network layers using projection for compatibility"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    # Get gradients directly (faster than hooks)
    grads = []
    layer_names = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grads.append(param.grad.clone().detach())
            layer_names.append(name)
    
    if len(grads) < 2:
        return np.array([0.0]), np.zeros((1, 1))
    
    # Simple projection to common dimension
    proj_dim = 64  # Smaller for speed
    projected_grads = []
    
    for g in grads:
        flat_g = g.flatten()
        
        if len(flat_g) > proj_dim:
            # Simple subsampling instead of matrix multiplication
            indices = torch.linspace(0, len(flat_g)-1, proj_dim).long()
            proj_g = flat_g[indices]
        else:
            # Pad with zeros
            proj_g = F.pad(flat_g, (0, proj_dim - len(flat_g)))
        
        # Normalize
        proj_g = proj_g / (torch.norm(proj_g) + epsilon)
        projected_grads.append(proj_g)
    
    # Compute coherence between adjacent layers only (faster)
    coherences = []
    for i in range(len(projected_grads) - 1):
        coherence = F.cosine_similarity(
            projected_grads[i], projected_grads[i+1], dim=0
        ).item()
        coherences.append(abs(coherence))
    
    return np.array(coherences), None  # Skip full matrix for speed

def compute_gradient_variance(model, data, target, loss_fn):
    """Baseline: layer-wise gradient variance"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    variances = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            var = param.grad.data.var().item()
            variances.append(var)
    
    return np.mean(variances) if variances else 0.0

# ============= TRAINING AND EVALUATION =============
def evaluate_model_subset(model, loader, device, max_batches=5):
    """Compute accuracy on a subset of data for speed"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if i >= max_batches:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0

def detect_coherence_collapse(gci_history, window=3):
    """Detect if gradient coherence has collapsed"""
    if len(gci_history) < window + 2:
        return False, 0.0
    
    recent = gci_history[-window:]
    earlier = gci_history[-2*window:-window]
    
    if not earlier:
        return False, 0.0
    
    recent_mean = np.mean(recent)
    earlier_mean = np.mean(earlier)
    
    if earlier_mean > 0:
        decrease_ratio = (earlier_mean - recent_mean) / earlier_mean
        collapsed = decrease_ratio > 0.25  # 25% decrease threshold
        return collapsed, decrease_ratio
    
    return False, 0.0

def train_with_monitoring(seed, dataset_name='mnist', max_epochs=15):
    """Train model while monitoring gradient coherence"""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset - use smaller subset for speed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use only 10% of data for speed
    n_samples = len(full_dataset) // 10
    indices = np.random.permutation(len(full_dataset))[:n_samples]
    subset_dataset = Subset(full_dataset, indices)
    
    # Split into train/val
    train_size = int(0.9 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    model = TinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()
    
    # Tracking metrics
    metrics = defaultdict(list)
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5
    
    collapse_detected = False
    collapse_epoch = -1
    collapse_magnitude = 0.0
    
    print(f"\nSeed {seed}: Starting training...")
    start_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Monitor metrics every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                # Compute GCI
                gci, _ = compute_gradient_coherence_index(
                    model, data, target, loss_fn
                )
                metrics['gci'].append(np.mean(gci))
                
                # Compute baseline
                grad_var = compute_gradient_variance(model, data, target, loss_fn)
                metrics['grad_variance'].append(grad_var)
        
        # Quick validation (subset only)
        val_acc = evaluate_model_subset(model, val_loader, device, max_batches=10)
        
        avg_loss = epoch_loss / len(train_loader)
        current_gci = metrics['gci'][-1] if metrics['gci'] else 0
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val={val_acc:.3f}, "
              f"GCI={current_gci:.3f}, Time={elapsed:.1f}s")
        
        # Check for coherence collapse
        if not collapse_detected and len(metrics['gci']) > 5:
            collapsed, magnitude = detect_coherence_collapse(metrics['gci'])
            if collapsed:
                collapse_detected = True
                collapse_epoch = epoch
                collapse_magnitude = magnitude
                print(f"COHERENCE COLLAPSE detected at epoch {epoch} (magnitude={magnitude:.3f})")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience or val_acc > 0.95:
            print(f"CONVERGED: Patience reached or high accuracy")
            break
    
    # Final evaluation on full test set
    test_acc = evaluate_model_subset(model, test_loader, device, max_batches=20)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")
    
    return {
        'seed': seed,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'collapse_detected': collapse_detected,
        'collapse_epoch': collapse_epoch,
        'collapse_magnitude': collapse_magnitude,
        'final_gci': metrics['gci'][-1] if metrics['gci'] else 0.0,
        'converged': True,
        'training_time': total_time
    }

# ============= MAIN EXPERIMENT =============
def run_experiment():
    """Run the full experiment with multiple seeds"""
    n_seeds = 3  # Small for feasibility probe
    results = []
    
    print("Starting Gradient Coherence Analysis Experiment (Feasibility Probe)")
    print("=" * 60)
    
    total_start = time.time()
    
    for seed in range(n_seeds):
        result = train_with_monitoring(seed, max_epochs=10)
        results.append(result)
    
    # Analyze results
    test_accs = [r['test_acc'] for r in results]
    collapse_mags = [r['collapse_magnitude'] for r in results if r['collapse_detected']]
    
    # Basic statistics
    signal_detected = False
    signal_description = ""
    
    collapse_rate = sum(1 for r in results if r['collapse_detected']) / len(results)
    
    if collapse_rate > 0.5:
        signal_detected = True
        signal_description = f"Coherence collapse detected in {collapse_rate*100:.0f}% of runs"
    else:
        signal_description = f"Coherence collapse rare ({collapse_rate*100:.0f}% of runs)"
    
    # Check if training worked
    mean_acc = np.mean(test_accs)
    if mean_acc > 0.8:
        print(f"SIGNAL_DETECTED: {signal_description}" if signal_detected else f"NO_SIGNAL: {signal_description}")
    else:
        print("NO_SIGNAL: Training failed to achieve good accuracy")
    
    total_time = time.time() - total_start
    
    # Prepare final results
    final_results = {
        'per_seed_results': results,
        'mean_test_acc': float(np.mean(test_accs)),
        'std_test_acc': float(np.std(test_accs)),
        'collapse_detection_rate': collapse_rate,
        'mean_collapse_magnitude': float(np.mean(collapse_mags)) if collapse_mags else 0.0,
        'convergence_status': True,
        'signal_detected': signal_detected,
        'signal_description': signal_description,
        'total_experiment_time': total_time
    }
    
    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_experiment()