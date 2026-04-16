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
    
    print("METRIC_SANITY_PASSED")

# Run sanity checks
sanity_check_gradient_coherence()

# ============= MODEL DEFINITIONS =============
class TinyCNN(nn.Module):
    """Tiny CNN for fast training"""
    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth
        if depth == 2:
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc1 = nn.Linear(16 * 7 * 7, 32)
        else:  # depth == 1, ablation
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.fc1 = nn.Linear(16 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        if self.depth == 2:
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleBaseline(nn.Module):
    """Simple 2-layer MLP baseline"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============= FAST METRIC IMPLEMENTATIONS =============
def compute_gradient_coherence_index_fast(model, data, target, loss_fn, epsilon=1e-8):
    """Fast GCI computation - only adjacent layers"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    # Get weight gradients only
    grads = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grads.append(param.grad.flatten())
    
    if len(grads) < 2:
        return 0.0
    
    # Only compute adjacent layer coherence
    coherences = []
    for i in range(len(grads) - 1):
        g1, g2 = grads[i], grads[i+1]
        
        # Subsample for speed
        max_dim = 128
        if len(g1) > max_dim:
            idx1 = torch.randperm(len(g1))[:max_dim]
            g1 = g1[idx1]
        if len(g2) > max_dim:
            idx2 = torch.randperm(len(g2))[:max_dim]
            g2 = g2[idx2]
        
        # Pad to same size
        max_len = max(len(g1), len(g2))
        if len(g1) < max_len:
            g1 = F.pad(g1, (0, max_len - len(g1)))
        if len(g2) < max_len:
            g2 = F.pad(g2, (0, max_len - len(g2)))
        
        # Normalize and compute coherence
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        if g1_norm > epsilon and g2_norm > epsilon:
            g1 = g1 / g1_norm
            g2 = g2 / g2_norm
            coherence = F.cosine_similarity(g1, g2, dim=0).item()
            coherences.append(abs(coherence))
    
    return np.mean(coherences) if coherences else 0.0

def compute_gradient_variance_fast(model):
    """Fast gradient variance - first and last layer only"""
    grads = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grads.append(param.grad)
    
    if len(grads) >= 2:
        # Only check first and last
        var1 = grads[0].var().item()
        var2 = grads[-1].var().item()
        return (var1 + var2) / 2
    return 0.0

# ============= FAST TRAINING =============
def evaluate_fast(model, loader, device, max_batches=5):
    """Fast evaluation on subset"""
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

def train_fast(seed, model_type='cnn', n_train=5000, max_epochs=15, time_limit=20):
    """Fast training with aggressive time limits"""
    start_time = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load minimal data
    transform = transforms.ToTensor()
    full_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Tiny subset
    indices = torch.randperm(len(full_train))[:n_train]
    train_subset = Subset(full_train, indices)
    
    # Quick split
    n_val = int(0.2 * len(train_subset))
    n_train_final = len(train_subset) - n_val
    train_data, val_data = random_split(train_subset, [n_train_final, n_val])
    
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Model
    if model_type == 'cnn':
        model = TinyCNN(depth=2).to(device)
    elif model_type == 'cnn_shallow':
        model = TinyCNN(depth=1).to(device)
    elif model_type == 'mlp':
        model = SimpleBaseline().to(device)
    elif model_type == 'random':
        # Random baseline - no training
        return {
            'seed': seed, 'test_acc': 0.1, 'val_acc': 0.1,
            'collapse_detected': False, 'final_gci': 0.0,
            'converged': True, 'epochs': 0
        }
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()
    
    # Metrics
    gci_history = []
    val_acc_history = []
    best_val = 0.0
    patience = 0
    
    collapse_detected = False
    collapse_magnitude = 0.0
    
    for epoch in range(max_epochs):
        # Time check
        if time.time() - start_time > time_limit:
            break
            
        # Train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            # Monitor GCI occasionally
            if batch_idx % 10 == 0:
                gci = compute_gradient_coherence_index_fast(model, data, target, loss_fn)
                gci_history.append(gci)
        
        # Quick validation
        val_acc = evaluate_fast(model, val_loader, device, max_batches=3)
        val_acc_history.append(val_acc)
        
        # Collapse detection
        if len(gci_history) > 8 and not collapse_detected:
            recent = np.mean(gci_history[-4:])
            earlier = np.mean(gci_history[-8:-4])
            if earlier > 0.1 and (earlier - recent) / earlier > 0.25:
                collapse_detected = True
                collapse_magnitude = (earlier - recent) / earlier
        
        # Early stopping
        if val_acc > best_val:
            best_val = val_acc
            patience = 0
        else:
            patience += 1
            
        if patience >= 3 or val_acc > 0.95:
            break
    
    # Final test
    test_acc = evaluate_fast(model, test_loader, device, max_batches=10)
    
    return {
        'seed': seed,
        'test_acc': test_acc,
        'val_acc': best_val,
        'collapse_detected': collapse_detected,
        'collapse_magnitude': collapse_magnitude,
        'final_gci': gci_history[-1] if gci_history else 0.0,
        'converged': patience >= 3 or val_acc > 0.95,
        'epochs': epoch + 1
    }

# ============= MAIN EXPERIMENT =============
def run_publication_experiment():
    """Fast publication experiment that fits in 300s"""
    print("Gradient Coherence Analysis: Fast Publication Experiment")
    print("=" * 60)
    
    experiment_start = time.time()
    
    # Sanity check
    print("\n1. Sanity check...")
    first_result = train_fast(0, model_type='cnn', time_limit=30)
    
    if first_result['final_gci'] == 0.0:
        print("SANITY_ABORT: GCI is zero")
        exit(1)
        
    print(f"✓ Sanity: acc={first_result['test_acc']:.3f}, gci={first_result['final_gci']:.3f}")
    
    results = {'main': [first_result], 'mlp': [], 'random': [], 'shallow': []}
    
    # Main experiment: CNN (10 seeds total, but fast)
    print("\n2. Main experiment (CNN)...")
    for seed in range(1, 10):
        if time.time() - experiment_start > 200:  # Leave time for analysis
            print("Time limit approaching, reducing seeds...")
            break
        result = train_fast(seed, model_type='cnn', time_limit=15)
        results['main'].append(result)
        print(f"  Seed {seed}: acc={result['test_acc']:.3f}")
    
    # Baselines
    print("\n3. Baselines...")
    
    # Random (3 seeds, instant)
    for seed in range(3):
        result = train_fast(seed, model_type='random')
        results['random'].append(result)
    
    # MLP (5 seeds)
    for seed in range(5):
        if time.time() - experiment_start > 250:
            break
        result = train_fast(seed, model_type='mlp', time_limit=10)
        results['mlp'].append(result)
        print(f"  MLP {seed}: acc={result['test_acc']:.3f}")
    
    # Ablation: shallow CNN (3 seeds)
    print("\n4. Ablation (shallow)...")
    for seed in range(3):
        if time.time() - experiment_start > 270:
            break
        result = train_fast(seed, model_type='cnn_shallow', time_limit=10)
        results['shallow'].append(result)
    
    # Analysis
    def compute_stats(res_list):
        if not res_list:
            return {}
        accs = [r['test_acc'] for r in res_list]
        collapse_rate = sum(1 for r in res_list if r['collapse_detected']) / len(res_list)
        return {
            'n': len(res_list),
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'collapse_rate': float(collapse_rate),
            'converged': all(r['converged'] for r in res_list)
        }
    
    # Statistics
    stats = {k: compute_stats(v) for k, v in results.items()}
    
    # Statistical tests
    if len(results['main']) >= 2 and len(results['mlp']) >= 2:
        main_accs = [r['test_acc'] for r in results['main']]
        mlp_accs = [r['test_acc'] for r in results['mlp']]
        t_stat, p_val = scipy.stats.ttest_ind(main_accs, mlp_accs)
    else:
        t_stat, p_val = 0.0, 1.0
    
    # Correlation
    collapse_res = [r for r in results['main'] if r['collapse_detected']]
    if len(collapse_res) >= 3:
        mags = [r['collapse_magnitude'] for r in collapse_res]
        accs = [r['test_acc'] for r in collapse_res]
        corr, corr_p = scipy.stats.pearsonr(mags, accs)
    else:
        corr, corr_p = 0.0, 1.0
    
    # Final results
    final_results = {
        'metadata': {
            'runtime_seconds': time.time() - experiment_start,
            'n_seeds': {k: len(v) for k, v in results.items()}
        },
        'statistics': stats,
        'per_seed_results': {
            'main': [{'seed': i, 'acc': r['test_acc'], 'collapse': r['collapse_detected']} 
                     for i, r in enumerate(results['main'])]
        },
        'significance_tests': {
            'cnn_vs_mlp': {'t': float(t_stat), 'p': float(p_val)},
            'collapse_correlation': {'r': float(corr), 'p': float(corr_p), 'n': len(collapse_res)}
        },
        'convergence_status': 'CONVERGED' if stats['main'].get('converged', False) else 'NOT_CONVERGED'
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Runtime: {time.time() - experiment_start:.1f}s")
    print(f"CNN: {stats['main']['mean_acc']:.3f} ± {stats['main']['std_acc']:.3f} (n={stats['main']['n']})")
    print(f"MLP: {stats['mlp']['mean_acc']:.3f} ± {stats['mlp']['std_acc']:.3f} (n={stats['mlp']['n']})")
    print(f"Collapse rate: {stats['main']['collapse_rate']*100:.0f}%")
    
    if stats['main']['collapse_rate'] > 0.2 and abs(corr) > 0.3:
        print("SIGNAL_DETECTED: Gradient coherence collapse observed")
    else:
        print("NO_SIGNAL: No clear gradient coherence pattern")
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_publication_experiment()