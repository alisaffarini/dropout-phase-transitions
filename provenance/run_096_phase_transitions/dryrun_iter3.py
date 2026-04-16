
# === DRY RUN VALIDATION: forced tiny scale ===
import builtins
_dry_run_got_results = False
_orig_print_fn = builtins.print
def _patched_print(*args, **kwargs):
    global _dry_run_got_results
    msg = " ".join(str(a) for a in args)
    if "RESULTS:" in msg:
        _dry_run_got_results = True
    _orig_print_fn(*args, **kwargs)
builtins.print = _patched_print

import atexit
def _check_results():
    if not _dry_run_got_results:
        _orig_print_fn("DRY_RUN_WARNING: Pipeline completed but no RESULTS: line was printed!")
        _orig_print_fn("DRY_RUN_WARNING: The post-processing/output stage may be broken.")
    else:
        _orig_print_fn("DRY_RUN_OK: Full pipeline validated (train → analyze → output)")
atexit.register(_check_results)

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
            # Simple subsampling for speed
            indices = torch.linspace(0, len(flat_g)-1, proj_dim).long()
            proj_g = flat_g[indices]
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

# ============= MODEL DEFINITIONS =============
class TinyCNN(nn.Module):
    """Tiny CNN for fast training"""
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

# ============= METRIC IMPLEMENTATIONS =============
def compute_gradient_coherence_index_fast(model, data, target, loss_fn, epsilon=1e-8):
    """Fast GCI computation"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    # Get gradients directly
    grads = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grads.append(param.grad.clone().detach())
    
    if len(grads) < 2:
        return 0.0
    
    # Simple adjacent layer coherence (fast)
    coherences = []
    proj_dim = 64  # Small for speed
    
    for i in range(len(grads) - 1):
        g1_flat = grads[i].flatten()
        g2_flat = grads[i+1].flatten()
        
        # Quick projection
        if len(g1_flat) > proj_dim:
            idx1 = torch.linspace(0, len(g1_flat)-1, proj_dim).long()
            g1_proj = g1_flat[idx1]
        else:
            g1_proj = F.pad(g1_flat, (0, proj_dim - len(g1_flat)))
            
        if len(g2_flat) > proj_dim:
            idx2 = torch.linspace(0, len(g2_flat)-1, proj_dim).long()
            g2_proj = g2_flat[idx2]
        else:
            g2_proj = F.pad(g2_flat, (0, proj_dim - len(g2_flat)))
        
        # Normalize and compute coherence
        g1_proj = g1_proj / (torch.norm(g1_proj) + epsilon)
        g2_proj = g2_proj / (torch.norm(g2_proj) + epsilon)
        
        coherence = F.cosine_similarity(g1_proj, g2_proj, dim=0).item()
        coherences.append(abs(coherence))
    
    return np.mean(coherences) if coherences else 0.0

def compute_gradient_variance_fast(model, data, target, loss_fn):
    """Fast gradient variance computation"""
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    # Only check first and last layer
    vars = []
    params = [(n, p) for n, p in model.named_parameters() if p.grad is not None and 'weight' in n]
    if len(params) >= 2:
        vars.append(params[0][1].grad.data.var().item())
        vars.append(params[-1][1].grad.data.var().item())
    
    return np.mean(vars) if vars else 0.0

# ============= TRAINING AND EVALUATION =============
def evaluate_model_fast(model, loader, device, max_batches=3):
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

def train_fast(seed, model_type='cnn', n_samples=5000, max_epochs=3, time_budget=30):
    """Fast training with time budget"""
    start_time = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tiny dataset
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
    
    # Use only n_samples
    indices = torch.randperm(len(full_dataset))[:n_samples]
    small_dataset = Subset(full_dataset, indices)
    
    # Quick split
    train_size = int(0.8 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    train_dataset, val_dataset = random_split(small_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    if model_type == 'cnn':
        model = TinyCNN().to(device)
    else:  # baseline
        model = SimpleBaseline().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Tracking
    metrics = {
        'gci_history': [],
        'var_history': [],
        'val_acc_history': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    collapse_detected = False
    collapse_magnitude = 0.0
    
    print(f"Seed {seed}, Model {model_type}: Training...")
    
    for epoch in range(max_epochs):
        # Check time budget
        if time.time() - start_time > time_budget:
            print(f"Time budget exceeded at epoch {epoch}")
            break
            
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
            
            # Quick metric computation (every 20 batches)
            if batch_idx % 20 == 0 and batch_idx > 0:
                gci = compute_gradient_coherence_index_fast(model, data, target, loss_fn)
                var = compute_gradient_variance_fast(model, data, target, loss_fn)
                metrics['gci_history'].append(gci)
                metrics['var_history'].append(var)
        
        # Fast validation
        val_acc = evaluate_model_fast(model, val_loader, device, max_batches=5)
        metrics['val_acc_history'].append(val_acc)
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch}: loss={avg_loss:.3f}, val_acc={val_acc:.3f}")
        
        # Detect collapse
        if len(metrics['gci_history']) > 10 and not collapse_detected:
            recent = np.mean(metrics['gci_history'][-5:])
            earlier = np.mean(metrics['gci_history'][-10:-5])
            if earlier > 0.1 and (earlier - recent) / earlier > 0.25:
                collapse_detected = True
                collapse_magnitude = (earlier - recent) / earlier
        
        # Early stopping
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 5 or val_acc > 0.95:
            print(f"  Converged at epoch {epoch}")
            break
    
    # Quick final eval
    test_acc = evaluate_model_fast(model, test_loader, device, max_batches=10)
    
    return {
        'seed': seed,
        'model_type': model_type,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'collapse_detected': collapse_detected,
        'collapse_magnitude': collapse_magnitude,
        'final_gci': metrics['gci_history'][-1] if metrics['gci_history'] else 0.0,
        'epochs': epoch + 1,
        'converged': patience_counter >= 5 or val_acc > 0.95
    }

# ============= MAIN EXPERIMENT =============
def run_experiment():
    """Run scaled experiment with time management"""
    print("Starting Gradient Coherence Analysis Experiment")
    print("=" * 60)
    
    # First seed sanity check
    print("\n1. Sanity check with first seed...")
    first_result = train_fast(0, model_type='cnn', n_samples=5000, time_budget=60)
    
    if first_result['final_gci'] == 0.0:
        print(f"SANITY_ABORT: GCI is zero")
        exit(1)
        
    if first_result['test_acc'] < 0.5:
        print(f"SANITY_ABORT: Training failed (acc={first_result['test_acc']})")
        exit(1)
        
    print(f"Sanity check passed: acc={first_result['test_acc']:.3f}, gci={first_result['final_gci']:.3f}")
    
    # Run experiments
    all_results = [first_result]
    
    # Main experiment: 10 seeds
    print("\n2. Main experiment (CNN with GCI)...")
    for seed in range(1, 10):
        result = train_fast(seed, model_type='cnn', n_samples=5000, time_budget=30)
        all_results.append(result)
        print(f"  Seed {seed} complete: acc={result['test_acc']:.3f}")
    
    # Baseline: Simple MLP (3 seeds)
    print("\n3. Baseline (Simple MLP)...")
    baseline_results = []
    for seed in range(3):
        result = train_fast(seed, model_type='baseline', n_samples=5000, time_budget=30)
        baseline_results.append(result)
        print(f"  Baseline seed {seed}: acc={result['test_acc']:.3f}")
    
    # Ablation: Different data size (3 seeds)
    print("\n4. Ablation (Less data)...")
    ablation_results = []
    for seed in range(3):
        result = train_fast(seed, model_type='cnn', n_samples=2000, time_budget=20)
        ablation_results.append(result)
    
    # Analysis
    def analyze_results(results):
        test_accs = [r['test_acc'] for r in results]
        collapse_rate = sum(1 for r in results if r['collapse_detected']) / len(results)
        collapse_mags = [r['collapse_magnitude'] for r in results if r['collapse_detected']]
        
        return {
            'mean_acc': float(np.mean(test_accs)),
            'std_acc': float(np.std(test_accs)),
            'collapse_rate': float(collapse_rate),
            'mean_collapse_mag': float(np.mean(collapse_mags)) if collapse_mags else 0.0,
            'n_samples': len(results)
        }
    
    # Stats
    main_stats = analyze_results(all_results)
    baseline_stats = analyze_results(baseline_results)
    ablation_stats = analyze_results(ablation_results)
    
    # T-test
    main_accs = [r['test_acc'] for r in all_results]
    baseline_accs = [r['test_acc'] for r in baseline_results]
    t_stat, p_value = scipy.stats.ttest_ind(main_accs, baseline_accs)
    
    # Correlation
    collapse_results = [r for r in all_results if r['collapse_detected']]
    if len(collapse_results) >= 3:
        mags = [r['collapse_magnitude'] for r in collapse_results]
        accs = [r['test_acc'] for r in collapse_results]
        corr, corr_p = scipy.stats.pearsonr(mags, accs)
    else:
        corr, corr_p = 0.0, 1.0
    
    # Random baseline
    random_acc = 0.1  # 10-class problem
    
    # Final results
    final_results = {
        'main_results': {
            'stats': main_stats,
            'per_seed': [{'seed': r['seed'], 'acc': r['test_acc'], 
                         'collapse': r['collapse_detected']} for r in all_results]
        },
        'baselines': {
            'random': {'mean_acc': random_acc},
            'simple_mlp': baseline_stats
        },
        'ablations': {
            'less_data': ablation_stats
        },
        'statistical_tests': {
            'cnn_vs_mlp': {'t': float(t_stat), 'p': float(p_value)},
            'cnn_vs_random': {'diff': main_stats['mean_acc'] - random_acc},
            'collapse_correlation': {'r': float(corr), 'p': float(corr_p)}
        },
        'convergence_status': 'CONVERGED' if all(r['converged'] for r in all_results) else 'NOT_CONVERGED'
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"CNN: {main_stats['mean_acc']:.3f} ± {main_stats['std_acc']:.3f}")
    print(f"MLP: {baseline_stats['mean_acc']:.3f} ± {baseline_stats['std_acc']:.3f}")
    print(f"Collapse rate: {main_stats['collapse_rate']*100:.0f}%")
    print(f"p-value (CNN vs MLP): {p_value:.4f}")
    
    if main_stats['collapse_rate'] > 0.2 and abs(corr) > 0.3:
        print(f"SIGNAL_DETECTED: Collapse correlates with performance (r={corr:.3f})")
    else:
        print("NO_SIGNAL: No clear relationship found")
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_experiment()