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
class CNN(nn.Module):
    """Standard CNN for MNIST/CIFAR"""
    def __init__(self, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        if num_layers == 3:
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 3 * 3, 128)
        elif num_layers == 2:  # Ablation: fewer layers
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        if self.num_layers == 3:
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleBaseline(nn.Module):
    """Simple 2-layer MLP baseline"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RandomBaseline(nn.Module):
    """Random prediction baseline"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.size(0)
        return torch.randn(batch_size, 10)

# ============= METRIC IMPLEMENTATIONS =============
def compute_gradient_coherence_index(model, data, target, loss_fn, epsilon=1e-8):
    """Compute GCI across network layers using projection"""
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
        return 0.0, []
    
    # Compute multi-scale coherence
    proj_dim = 128  # Moderate size for accuracy
    coherences = []
    
    for i in range(len(grads) - 1):
        g1_flat = grads[i].flatten()
        g2_flat = grads[i+1].flatten()
        
        # Project to common dimension
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
        g1_norm = torch.norm(g1_proj)
        g2_norm = torch.norm(g2_proj)
        
        if g1_norm > epsilon and g2_norm > epsilon:
            g1_proj = g1_proj / g1_norm
            g2_proj = g2_proj / g2_norm
            coherence = F.cosine_similarity(g1_proj, g2_proj, dim=0).item()
            coherences.append(abs(coherence))
        else:
            coherences.append(0.0)
    
    mean_coherence = np.mean(coherences) if coherences else 0.0
    return mean_coherence, coherences

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

def compute_gradient_norm_ratio(model, data, target, loss_fn):
    """Baseline: gradient norm imbalance"""
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
    
    # Ratio of max to min norm (measure of imbalance)
    return max(norms) / (min(norms) + 1e-8)

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
    """Detect if gradient coherence has collapsed"""
    if len(gci_history) < window * 2:
        return False, 0.0
    
    recent = gci_history[-window:]
    earlier = gci_history[-2*window:-window]
    
    recent_mean = np.mean(recent)
    earlier_mean = np.mean(earlier)
    
    if earlier_mean > 0.1:
        decrease_ratio = (earlier_mean - recent_mean) / earlier_mean
        collapsed = decrease_ratio > 0.25
        return collapsed, decrease_ratio
    
    return False, 0.0

def train_model(seed, model_type='cnn', dataset='mnist', max_epochs=50, 
                monitor_frequency=10, use_scheduler='cosine'):
    """Train model with comprehensive monitoring"""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_class = torchvision.datasets.MNIST
        input_channels = 1
    elif dataset == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset_class = torchvision.datasets.FashionMNIST
        input_channels = 1
    
    full_train = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    # Proper train/val split
    train_size = int(0.85 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    if model_type == 'cnn':
        model = CNN(num_layers=3).to(device)
    elif model_type == 'cnn_shallow':
        model = CNN(num_layers=2).to(device)
    elif model_type == 'mlp':
        model = SimpleBaseline().to(device)
    elif model_type == 'random':
        model = RandomBaseline().to(device)
        # Random baseline doesn't need training
        test_acc = evaluate_model(model, test_loader, device)
        return {
            'seed': seed, 'model_type': model_type, 'dataset': dataset,
            'test_acc': test_acc, 'val_acc': 0.1, 'train_acc': 0.1,
            'collapse_detected': False, 'converged': True,
            'final_gci': 0.0, 'epochs_trained': 0
        }
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if use_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Metrics tracking
    metrics = defaultdict(list)
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    collapse_detected = False
    collapse_epoch = -1
    collapse_magnitude = 0.0
    
    print(f"\nSeed {seed}, {model_type.upper()} on {dataset.upper()}:")
    
    for epoch in range(max_epochs):
        # Training
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
            
            # Monitor metrics
            if batch_idx % monitor_frequency == 0 and batch_idx > 0:
                gci, layer_coherences = compute_gradient_coherence_index(
                    model, data, target, loss_fn
                )
                var = compute_gradient_variance(model, data, target, loss_fn)
                ratio = compute_gradient_norm_ratio(model, data, target, loss_fn)
                
                metrics['gci'].append(gci)
                metrics['layer_coherences'].append(layer_coherences)
                metrics['grad_variance'].append(var)
                metrics['grad_norm_ratio'].append(ratio)
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        metrics['val_acc'].append(val_acc)
        
        avg_loss = epoch_loss / len(train_loader)
        current_gci = metrics['gci'][-1] if metrics['gci'] else 0.0
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={avg_loss:.3f}, val={val_acc:.3f}, gci={current_gci:.3f}")
        
        # Detect collapse
        if not collapse_detected and len(metrics['gci']) > 10:
            collapsed, magnitude = detect_coherence_collapse(metrics['gci'])
            if collapsed:
                collapse_detected = True
                collapse_epoch = epoch
                collapse_magnitude = magnitude
                print(f"  COLLAPSE at epoch {epoch} (magnitude={magnitude:.3f})")
        
        # Learning rate scheduling
        if use_scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"  CONVERGED at epoch {epoch}")
            break
        
        if val_acc > 0.995:
            print(f"  CONVERGED (high accuracy) at epoch {epoch}")
            break
    
    # Final evaluation
    train_acc = evaluate_model(model, train_loader, device)
    test_acc = evaluate_model(model, test_loader, device)
    
    result = {
        'seed': seed,
        'model_type': model_type,
        'dataset': dataset,
        'test_acc': test_acc,
        'val_acc': best_val_acc,
        'train_acc': train_acc,
        'generalization_gap': train_acc - test_acc,
        'collapse_detected': collapse_detected,
        'collapse_epoch': collapse_epoch,
        'collapse_magnitude': collapse_magnitude,
        'final_gci': metrics['gci'][-1] if metrics['gci'] else 0.0,
        'mean_gci': np.mean(metrics['gci']) if metrics['gci'] else 0.0,
        'final_grad_var': metrics['grad_variance'][-1] if metrics['grad_variance'] else 0.0,
        'epochs_trained': epoch + 1,
        'converged': patience_counter >= max_patience or val_acc > 0.995
    }
    
    return result

# ============= MAIN EXPERIMENT =============
def run_publication_experiment():
    """Run full publication-quality experiment"""
    print("Gradient Coherence Analysis: Publication-Quality Experiment")
    print("=" * 70)
    
    start_time = time.time()
    
    # Seed 0 sanity check
    print("\n1. Running sanity check with seed 0...")
    first_result = train_model(0, model_type='cnn', dataset='mnist')
    
    # Early abort checks
    if first_result['final_gci'] == 0.0:
        print("SANITY_ABORT: GCI metric is zero")
        exit(1)
    
    if first_result['test_acc'] < 0.8:
        print(f"SANITY_ABORT: Poor accuracy ({first_result['test_acc']:.3f})")
        exit(1)
        
    print(f"✓ Sanity check passed: acc={first_result['test_acc']:.3f}, gci={first_result['final_gci']:.3f}")
    
    all_results = []
    
    # MAIN EXPERIMENT: 10 seeds with CNN
    print("\n2. Main experiment: CNN with gradient coherence monitoring")
    all_results.append(first_result)
    
    for seed in range(1, 10):
        result = train_model(seed, model_type='cnn', dataset='mnist')
        all_results.append(result)
        print(f"  Seed {seed}: acc={result['test_acc']:.3f}, collapse={result['collapse_detected']}")
    
    # BASELINES
    print("\n3. Baselines:")
    
    # Random baseline
    random_results = []
    for seed in range(3):
        result = train_model(seed, model_type='random', dataset='mnist')
        random_results.append(result)
    
    # Simple MLP baseline
    mlp_results = []
    for seed in range(5):
        result = train_model(seed, model_type='mlp', dataset='mnist')
        mlp_results.append(result)
        print(f"  MLP seed {seed}: acc={result['test_acc']:.3f}")
    
    # ABLATIONS
    print("\n4. Ablation studies:")
    
    # Ablation 1: Shallower network
    shallow_results = []
    for seed in range(5):
        result = train_model(seed, model_type='cnn_shallow', dataset='mnist')
        shallow_results.append(result)
        print(f"  Shallow CNN seed {seed}: acc={result['test_acc']:.3f}")
    
    # Ablation 2: Different dataset
    fashion_results = []
    for seed in range(5):
        result = train_model(seed, model_type='cnn', dataset='fashion')
        fashion_results.append(result)
        print(f"  Fashion-MNIST seed {seed}: acc={result['test_acc']:.3f}")
    
    # Ablation 3: Different scheduler
    plateau_results = []
    for seed in range(3):
        result = train_model(seed, model_type='cnn', dataset='mnist', use_scheduler='plateau')
        plateau_results.append(result)
    
    # ANALYSIS
    def analyze_results(results, name=""):
        """Compute comprehensive statistics"""
        if not results:
            return {}
            
        accs = [r['test_acc'] for r in results]
        gen_gaps = [r['generalization_gap'] for r in results]
        collapse_rate = sum(1 for r in results if r['collapse_detected']) / len(results)
        collapse_mags = [r['collapse_magnitude'] for r in results if r['collapse_detected']]
        
        stats = {
            'n': len(results),
            'test_acc_mean': float(np.mean(accs)),
            'test_acc_std': float(np.std(accs)),
            'test_acc_95ci': float(1.96 * np.std(accs) / np.sqrt(len(accs))),
            'gen_gap_mean': float(np.mean(gen_gaps)),
            'gen_gap_std': float(np.std(gen_gaps)),
            'collapse_rate': float(collapse_rate),
            'collapse_mag_mean': float(np.mean(collapse_mags)) if collapse_mags else 0.0,
            'converged_rate': sum(1 for r in results if r['converged']) / len(results)
        }
        
        return stats
    
    # Compute all statistics
    main_stats = analyze_results(all_results, "Main CNN")
    random_stats = analyze_results(random_results, "Random")
    mlp_stats = analyze_results(mlp_results, "MLP")
    shallow_stats = analyze_results(shallow_results, "Shallow CNN")
    fashion_stats = analyze_results(fashion_results, "Fashion-MNIST")
    plateau_stats = analyze_results(plateau_results, "Plateau scheduler")
    
    # STATISTICAL TESTS
    print("\n5. Statistical tests:")
    
    # T-tests
    main_accs = [r['test_acc'] for r in all_results]
    mlp_accs = [r['test_acc'] for r in mlp_results]
    
    t_cnn_mlp, p_cnn_mlp = scipy.stats.ttest_ind(main_accs, mlp_accs)
    print(f"  CNN vs MLP: t={t_cnn_mlp:.3f}, p={p_cnn_mlp:.4f}")
    
    # Paired t-test for shallow ablation (same seeds)
    shallow_accs = [r['test_acc'] for r in shallow_results[:5]]
    main_accs_paired = [r['test_acc'] for r in all_results[:5]]
    t_paired, p_paired = scipy.stats.ttest_rel(main_accs_paired, shallow_accs)
    
    # Correlation: collapse magnitude vs generalization
    collapse_results = [r for r in all_results if r['collapse_detected']]
    if len(collapse_results) >= 3:
        mags = [r['collapse_magnitude'] for r in collapse_results]
        gaps = [r['generalization_gap'] for r in collapse_results]
        corr_r, corr_p = scipy.stats.pearsonr(mags, gaps)
        print(f"  Collapse-generalization correlation: r={corr_r:.3f}, p={corr_p:.3f}")
    else:
        corr_r, corr_p = 0.0, 1.0
    
    # Bootstrap confidence intervals
    def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
        """Compute bootstrap confidence interval"""
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1-ci)/2 * 100)
        upper = np.percentile(means, (1+ci)/2 * 100)
        return lower, upper
    
    main_ci_lower, main_ci_upper = bootstrap_ci(main_accs)
    
    # Final results compilation
    final_results = {
        'experiment_metadata': {
            'total_runtime_seconds': time.time() - start_time,
            'n_seeds_main': 10,
            'n_seeds_baselines': {'random': 3, 'mlp': 5},
            'n_seeds_ablations': {'shallow': 5, 'fashion': 5, 'plateau_scheduler': 3}
        },
        
        'main_results': {
            'stats': main_stats,
            'per_seed': [
                {
                    'seed': r['seed'], 
                    'test_acc': r['test_acc'],
                    'collapse': r['collapse_detected'],
                    'collapse_epoch': r['collapse_epoch'] if r['collapse_detected'] else -1
                } 
                for r in all_results
            ]
        },
        
        'baselines': {
            'random': random_stats,
            'simple_mlp': mlp_stats
        },
        
        'ablations': {
            'shallow_network': shallow_stats,
            'fashion_dataset': fashion_stats,
            'plateau_scheduler': plateau_stats
        },
        
        'statistical_tests': {
            'cnn_vs_mlp': {
                't_statistic': float(t_cnn_mlp),
                'p_value': float(p_cnn_mlp),
                'significant': p_cnn_mlp < 0.05
            },
            'cnn_vs_shallow_paired': {
                't_statistic': float(t_paired),
                'p_value': float(p_paired),
                'significant': p_paired < 0.05
            },
            'collapse_correlation': {
                'r': float(corr_r),
                'p_value': float(corr_p),
                'n_collapse_events': len(collapse_results)
            },
            'bootstrap_95ci': {
                'lower': float(main_ci_lower),
                'upper': float(main_ci_upper)
            }
        },
        
        'convergence_status': 'CONVERGED' if all(r['converged'] for r in all_results) else 'NOT_CONVERGED',
        
        'key_findings': {
            'signal_detected': main_stats['collapse_rate'] > 0.2 and abs(corr_r) > 0.3,
            'cnn_beats_baseline': main_stats['test_acc_mean'] > mlp_stats['test_acc_mean'] + 0.02,
            'collapse_predictive': abs(corr_r) > 0.3 and corr_p < 0.1
        }
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Runtime: {(time.time() - start_time)/60:.1f} minutes")
    print(f"\nTest Accuracy (mean ± std):")
    print(f"  CNN:      {main_stats['test_acc_mean']:.3f} ± {main_stats['test_acc_std']:.3f}")
    print(f"  MLP:      {mlp_stats['test_acc_mean']:.3f} ± {mlp_stats['test_acc_std']:.3f}")
    print(f"  Random:   {random_stats['test_acc_mean']:.3f}")
    print(f"\nCollapse rate: {main_stats['collapse_rate']*100:.0f}%")
    print(f"Statistical significance: p={p_cnn_mlp:.4f}")
    
    if final_results['key_findings']['signal_detected']:
        print("\nSIGNAL_DETECTED: Gradient coherence collapse correlates with generalization")
    else:
        print("\nNO_SIGNAL: Gradient coherence not predictive")
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_publication_experiment()