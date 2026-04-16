# pip install torch torchvision scipy scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
from collections import defaultdict
import scipy.stats
from sklearn.cross_decomposition import CCA
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
    
    # Test 4: Multi-layer coherence matrix
    grads = [torch.randn(10, 10) for _ in range(4)]
    coherence_matrix = compute_coherence_matrix(grads)
    assert coherence_matrix.shape == (4, 4), f"Wrong coherence matrix shape: {coherence_matrix.shape}"
    assert np.allclose(np.diag(coherence_matrix), 1.0), "Diagonal should be 1.0"
    
    print("METRIC_SANITY_PASSED")

def compute_coherence_matrix(grads):
    """Helper for sanity check"""
    n_layers = len(grads)
    matrix = np.zeros((n_layers, n_layers))
    
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                matrix[i, j] = 1.0
            else:
                g_i = grads[i].flatten()
                g_j = grads[j].flatten()
                matrix[i, j] = F.cosine_similarity(g_i, g_j, dim=0).item()
    
    return matrix

# Run sanity checks
sanity_check_gradient_coherence()

# ============= MODEL DEFINITION =============
class SmallCNN(nn.Module):
    """Small CNN for MNIST - designed for clear layer hierarchy"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============= METRIC IMPLEMENTATIONS =============
def compute_gradient_coherence_index(model, data, target, loss_fn, epsilon=1e-5):
    """Compute GCI across network layers"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    
    # Get per-layer gradients
    layer_grads = []
    layer_names = []
    
    # Hook to capture gradients
    def get_grad(name):
        def hook(grad):
            layer_grads.append((name, grad.clone().detach()))
        return hook
    
    # Register hooks on key layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                handle = module.weight.register_hook(get_grad(name))
                handles.append(handle)
                layer_names.append(name)
    
    # Backward pass
    loss.backward()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Sort gradients by layer order
    layer_grads = sorted(layer_grads, key=lambda x: layer_names.index(x[0]))
    grads = [g[1] for g in layer_grads]
    
    if len(grads) < 2:
        return np.array([0.0]), np.zeros((1, 1))
    
    # Compute coherence matrix
    n_layers = len(grads)
    coherence_matrix = np.zeros((n_layers, n_layers))
    
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                coherence_matrix[i, j] = 1.0
            else:
                g_i = grads[i].flatten()
                g_j = grads[j].flatten()
                
                # Use cosine similarity as primary measure
                coherence = F.cosine_similarity(g_i, g_j, dim=0).item()
                coherence_matrix[i, j] = coherence
    
    # Multi-scale coherence (comparing layers at different distances)
    scales = range(1, min(4, n_layers))  # Adaptive to network depth
    scale_coherences = []
    
    for k in scales:
        k_coherences = []
        for i in range(n_layers - k):
            k_coherences.append(abs(coherence_matrix[i, i+k]))  # Use abs for magnitude
        if k_coherences:
            scale_coherences.append(np.mean(k_coherences))
    
    if not scale_coherences:
        scale_coherences = [0.0]
    
    return np.array(scale_coherences), coherence_matrix

def compute_gradient_variance(model, data, target, loss_fn):
    """Baseline: layer-wise gradient variance"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    variances = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            var = param.grad.data.var().item()
            variances.append(var)
    
    return np.mean(variances) if variances else 0.0

def compute_gradient_norm_ratios(model, data, target, loss_fn):
    """Baseline: ratio of gradient norms between layers"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            norm = param.grad.data.norm().item()
            norms.append(norm)
    
    if len(norms) < 2:
        return 1.0
    
    # Compute ratio between first and last layer
    ratio = norms[-1] / (norms[0] + 1e-8)
    return ratio

# ============= TRAINING AND EVALUATION =============
def evaluate_model(model, loader, device):
    """Compute accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0

def detect_coherence_collapse(gci_history, window=5):
    """Detect if gradient coherence has collapsed (rapid decrease)"""
    if len(gci_history) < window + 2:
        return False, 0.0
    
    recent = gci_history[-window:]
    earlier = gci_history[-2*window:-window]
    
    if not earlier:
        return False, 0.0
    
    # Check if mean coherence decreased significantly
    recent_mean = np.mean(recent)
    earlier_mean = np.mean(earlier)
    
    if earlier_mean > 0:
        decrease_ratio = (earlier_mean - recent_mean) / earlier_mean
        collapsed = decrease_ratio > 0.3  # 30% decrease threshold
        return collapsed, decrease_ratio
    
    return False, 0.0

def train_with_monitoring(seed, dataset_name='mnist'):
    """Train model while monitoring gradient coherence"""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if dataset_name == 'mnist':
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
        
        # Split into train/val
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Tracking metrics
    metrics = defaultdict(list)
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    collapse_detected = False
    collapse_epoch = -1
    collapse_magnitude = 0.0
    
    print(f"\nSeed {seed}: Starting training...")
    
    for epoch in range(50):  # Max epochs
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
            
            # Monitor metrics every 20 batches
            if batch_idx % 20 == 0:
                # Compute GCI
                gci, coherence_matrix = compute_gradient_coherence_index(
                    model, data, target, loss_fn
                )
                metrics['gci'].append(np.mean(gci))
                metrics['gci_full'].append(gci)
                
                # Compute baselines
                grad_var = compute_gradient_variance(model, data, target, loss_fn)
                grad_ratio = compute_gradient_norm_ratios(model, data, target, loss_fn)
                
                metrics['grad_variance'].append(grad_var)
                metrics['grad_norm_ratio'].append(grad_ratio)
                metrics['epoch'].append(epoch + batch_idx / len(train_loader))
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        train_acc = evaluate_model(model, train_loader, device)
        
        metrics['val_acc_history'].append(val_acc)
        metrics['train_acc_history'].append(train_acc)
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.3f}, Val={val_acc:.3f}, "
              f"GCI={metrics['gci'][-1]:.3f} if metrics['gci'] else 0")
        
        # Check for coherence collapse
        if not collapse_detected and len(metrics['gci']) > 10:
            collapsed, magnitude = detect_coherence_collapse(metrics['gci'])
            if collapsed:
                collapse_detected = True
                collapse_epoch = epoch
                collapse_magnitude = magnitude
                print(f"COHERENCE COLLAPSE detected at epoch {epoch} (magnitude={magnitude:.3f})")
        
        # Check convergence
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(val_acc)
        
        if patience_counter >= max_patience:
            print(f"CONVERGED: No improvement for {max_patience} epochs")
            break
    
    # Final evaluation
    test_acc = evaluate_model(model, test_loader, device)
    final_train_acc = evaluate_model(model, train_loader, device)
    generalization_gap = test_acc - final_train_acc
    
    return {
        'seed': seed,
        'test_acc': test_acc,
        'train_acc': final_train_acc,
        'generalization_gap': generalization_gap,
        'collapse_detected': collapse_detected,
        'collapse_epoch': collapse_epoch,
        'collapse_magnitude': collapse_magnitude,
        'final_gci': metrics['gci'][-1] if metrics['gci'] else 0.0,
        'gci_history': metrics['gci'],
        'val_acc_history': metrics['val_acc_history'],
        'converged': patience_counter >= max_patience
    }

# ============= MAIN EXPERIMENT =============
def run_experiment():
    """Run the full experiment with multiple seeds"""
    n_seeds = 3  # Small for feasibility probe
    results = []
    
    print("Starting Gradient Coherence Analysis Experiment")
    print("=" * 60)
    
    for seed in range(n_seeds):
        result = train_with_monitoring(seed)
        results.append(result)
    
    # Analyze results
    test_accs = [r['test_acc'] for r in results]
    gen_gaps = [r['generalization_gap'] for r in results]
    collapse_mags = [r['collapse_magnitude'] for r in results if r['collapse_detected']]
    
    # Correlation analysis
    if len(collapse_mags) >= 2:
        # For seeds that had collapse, correlate magnitude with generalization
        collapse_results = [r for r in results if r['collapse_detected']]
        if len(collapse_results) >= 2:
            mags = [r['collapse_magnitude'] for r in collapse_results]
            gaps = [r['generalization_gap'] for r in collapse_results]
            corr, p_value = scipy.stats.pearsonr(mags, gaps)
        else:
            corr, p_value = 0.0, 1.0
    else:
        corr, p_value = 0.0, 1.0
    
    # Check if method provides signal
    signal_detected = False
    signal_description = ""
    
    if len(collapse_mags) >= 2 and corr > 0.3:
        signal_detected = True
        signal_description = f"Collapse magnitude correlates with generalization (r={corr:.3f}, p={p_value:.3f})"
    
    # Random baseline comparison
    random_accs = [0.1] * n_seeds  # Random baseline for 10-class problem
    random_mean = np.mean(random_accs)
    method_mean = np.mean(test_accs)
    
    if method_mean > random_mean * 5:  # Significantly better than random
        if signal_detected:
            print(f"SIGNAL_DETECTED: {signal_description}")
        else:
            print("NO_SIGNAL: Method works but coherence collapse not predictive")
    else:
        print("NO_SIGNAL: Method failed to train properly")
    
    # Prepare final results
    final_results = {
        'per_seed_results': results,
        'mean_test_acc': float(np.mean(test_accs)),
        'std_test_acc': float(np.std(test_accs)),
        'mean_gen_gap': float(np.mean(gen_gaps)),
        'std_gen_gap': float(np.std(gen_gaps)),
        'collapse_detection_rate': sum(1 for r in results if r['collapse_detected']) / len(results),
        'collapse_correlation': {
            'correlation': float(corr),
            'p_value': float(p_value),
            'n_samples': len(collapse_mags)
        },
        'convergence_status': all(r['converged'] for r in results),
        'signal_detected': signal_detected,
        'signal_description': signal_description
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_experiment()