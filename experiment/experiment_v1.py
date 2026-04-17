#!/usr/bin/env python3
"""
Dropout Phase Diagrams: Critical Temperatures and Phase Transitions
in Neural Network Training

Maps the 2D phase diagram (training epoch × dropout rate) for multiple
architectures on CIFAR-10/100, identifying thermodynamic-like phase transitions.

Author: Ali Saffarini
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d

# ============================================================
# Config
# ============================================================
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'
EPOCHS = 200
CHECKPOINT_EVERY = 5  # save checkpoint every N epochs
BATCH_SIZE = 128
LR = 0.1
MC_SAMPLES = 50  # Monte Carlo forward passes per dropout rate
DROPOUT_RATES = np.linspace(0.0, 0.96, 25)  # 25 dropout rates to sweep
NUM_TEST_BATCHES = 10  # number of test batches for phase diagram (speed vs accuracy)
DATASETS = ['cifar10', 'cifar100']
MODEL_NAMES = ['simple_cnn', 'vgg11_nodropout', 'resnet18_nobn', 'mlp']

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Models (trained WITHOUT dropout — dropout applied only at eval)
# ============================================================

class SimpleCNN(nn.Module):
    """4-layer CNN, no dropout, no BN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, drop_rate=0.0):
        x = self.features(x)
        # Apply dropout to feature maps if requested
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = x.view(x.size(0), -1)
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = self.classifier(x)
        return x


class VGG11NoBN(nn.Module):
    """VGG-11 without BatchNorm or Dropout (clean baseline)"""
    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_ch = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([nn.Conv2d(in_ch, v, 3, padding=1), nn.ReLU(True)])
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, drop_rate=0.0):
        x = self.features(x)
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = x.view(x.size(0), -1)
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """ResNet basic block WITHOUT BatchNorm"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18NoBN(nn.Module):
    """ResNet-18 without BatchNorm"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, drop_rate=0.0):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        if drop_rate > 0:
            out = F.dropout(out, p=drop_rate, training=True)
        out = self.layer2(out)
        if drop_rate > 0:
            out = F.dropout(out, p=drop_rate, training=True)
        out = self.layer3(out)
        if drop_rate > 0:
            out = F.dropout(out, p=drop_rate, training=True)
        out = self.layer4(out)
        if drop_rate > 0:
            out = F.dropout(out, p=drop_rate, training=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MLP(nn.Module):
    """3-layer MLP"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self._num_layers = 3  # for dropout insertion

    def forward(self, x, drop_rate=0.0):
        x = x.view(x.size(0), -1)
        x = F.relu(self.net[1](x))
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = F.relu(self.net[3](x))
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
        x = self.net[5](x)
        return x


def build_model(name, num_classes):
    if name == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif name == 'vgg11_nodropout':
        return VGG11NoBN(num_classes)
    elif name == 'resnet18_nobn':
        return ResNet18NoBN(num_classes)
    elif name == 'mlp':
        return MLP(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
# Data
# ============================================================

def get_dataloaders(dataset_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    if dataset_name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, num_classes


# ============================================================
# Phase Diagram Measurement
# ============================================================

@torch.no_grad()
def measure_phase_diagram(model, test_loader, dropout_rates, mc_samples, num_classes, num_batches=10):
    """
    For each dropout rate, run MC forward passes and compute:
    - mean prediction entropy (order parameter)
    - mean prediction confidence
    - accuracy
    Returns dict of arrays indexed by dropout rate.
    """
    model.eval()  # BN in eval mode (not that we use BN), but dropout applied manually

    # Collect test batches
    test_batches = []
    for i, (x, y) in enumerate(test_loader):
        if i >= num_batches:
            break
        test_batches.append((x.to(DEVICE), y.to(DEVICE)))

    results = {
        'entropy': [],      # mean entropy per dropout rate
        'confidence': [],    # mean max-prob per dropout rate
        'accuracy': [],      # accuracy per dropout rate
        'entropy_std': [],   # std of entropy across samples
    }

    for p in dropout_rates:
        all_entropies = []
        all_confidences = []
        correct = 0
        total = 0

        for x, y in test_batches:
            # MC forward passes
            logit_sum = torch.zeros(x.size(0), num_classes, device=DEVICE)
            prob_sum = torch.zeros(x.size(0), num_classes, device=DEVICE)

            for _ in range(mc_samples):
                logits = model(x, drop_rate=p)
                probs = F.softmax(logits, dim=1)
                prob_sum += probs

            # Average prediction
            avg_probs = prob_sum / mc_samples
            avg_probs = torch.clamp(avg_probs, min=1e-8)  # numerical stability

            # Entropy of average prediction
            entropy = -(avg_probs * torch.log(avg_probs)).sum(dim=1)  # per-sample
            all_entropies.append(entropy.cpu().numpy())

            # Confidence
            confidence = avg_probs.max(dim=1)[0]
            all_confidences.append(confidence.cpu().numpy())

            # Accuracy
            preds = avg_probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        all_entropies = np.concatenate(all_entropies)
        all_confidences = np.concatenate(all_confidences)

        results['entropy'].append(float(np.mean(all_entropies)))
        results['entropy_std'].append(float(np.std(all_entropies)))
        results['confidence'].append(float(np.mean(all_confidences)))
        results['accuracy'].append(correct / total)

    return {k: np.array(v) for k, v in results.items()}


def compute_susceptibility(entropies, dropout_rates):
    """Numerical derivative of entropy w.r.t. dropout rate (susceptibility)."""
    dp = np.diff(dropout_rates)
    dH = np.diff(entropies)
    chi = dH / dp
    # Pad to same length
    chi = np.concatenate([chi, [chi[-1]]])
    return chi


def find_critical_point(susceptibility, dropout_rates):
    """Find dropout rate where susceptibility is maximum."""
    idx = np.argmax(susceptibility)
    return dropout_rates[idx], float(susceptibility[idx])


# ============================================================
# Training Loop
# ============================================================

def train_and_measure(model_name, dataset_name):
    """Train model, checkpoint periodically, measure phase diagram at each checkpoint."""
    print(f"\n{'='*70}")
    print(f"  {model_name} on {dataset_name}")
    print(f"{'='*70}")

    set_seed(SEED)
    train_loader, test_loader, num_classes = get_dataloaders(dataset_name)
    model = build_model(model_name, num_classes).to(DEVICE)
    
    # Use lower LR for models without BN (they're harder to train)
    lr = 0.01 if model_name in ['resnet18_nobn', 'vgg11_nodropout'] else LR
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # Storage for phase diagram across training
    phase_data = {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'phase_diagrams': [],  # list of dicts, one per checkpoint
        'critical_points': [],  # (p_c, chi_max) per checkpoint
        'dropout_rates': DROPOUT_RATES.tolist(),
    }

    # Measure at epoch 0 (random init)
    print(f"  [Epoch 0] Measuring phase diagram at initialization...")
    pd_result = measure_phase_diagram(model, test_loader, DROPOUT_RATES, MC_SAMPLES, num_classes, NUM_TEST_BATCHES)
    susceptibility = compute_susceptibility(pd_result['entropy'], DROPOUT_RATES)
    pc, chi_max = find_critical_point(susceptibility, DROPOUT_RATES)
    phase_data['epochs'].append(0)
    phase_data['train_acc'].append(0.0)
    phase_data['test_acc'].append(pd_result['accuracy'][0])
    phase_data['phase_diagrams'].append({k: v.tolist() for k, v in pd_result.items()})
    phase_data['critical_points'].append({'p_c': pc, 'chi_max': chi_max})
    print(f"    p_c={pc:.3f}, chi_max={chi_max:.3f}, acc@0drop={pd_result['accuracy'][0]:.4f}")

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, drop_rate=0.0)  # NO dropout during training
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] NaN/Inf loss at epoch {epoch}, breaking early")
                return phase_data  # return what we have so far
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_acc = correct / total

        if epoch % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [Epoch {epoch}/{EPOCHS}] loss={running_loss/len(train_loader):.4f} "
                  f"train_acc={train_acc:.4f} elapsed={elapsed:.0f}s")

        # Checkpoint & measure phase diagram
        if epoch % CHECKPOINT_EVERY == 0:
            pd_result = measure_phase_diagram(model, test_loader, DROPOUT_RATES, MC_SAMPLES, num_classes, NUM_TEST_BATCHES)
            susceptibility = compute_susceptibility(pd_result['entropy'], DROPOUT_RATES)
            pc, chi_max = find_critical_point(susceptibility, DROPOUT_RATES)

            phase_data['epochs'].append(epoch)
            phase_data['train_acc'].append(train_acc)
            phase_data['test_acc'].append(pd_result['accuracy'][0])
            phase_data['phase_diagrams'].append({k: v.tolist() for k, v in pd_result.items()})
            phase_data['critical_points'].append({'p_c': pc, 'chi_max': chi_max})

            if epoch % CHECKPOINT_EVERY == 0:
                gen_gap = train_acc - pd_result['accuracy'][0]
                print(f"    -> p_c={pc:.3f} | chi_max={chi_max:.3f} | "
                      f"test_acc={pd_result['accuracy'][0]:.4f} | gen_gap={gen_gap:.4f}")

    total_time = time.time() - t_start
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save raw data
    save_path = os.path.join(RESULTS_DIR, f'{model_name}_{dataset_name}.json')
    with open(save_path, 'w') as f:
        json.dump(phase_data, f, indent=2)
    print(f"  Saved: {save_path}")

    return phase_data


# ============================================================
# Visualization
# ============================================================

def plot_phase_diagram_heatmap(phase_data, model_name, dataset_name):
    """Plot 2D heatmap: epoch × dropout rate, colored by entropy."""
    epochs = phase_data['epochs']
    dropout_rates = phase_data['dropout_rates']

    # Build 2D entropy matrix
    entropy_matrix = np.array([pd['entropy'] for pd in phase_data['phase_diagrams']])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Entropy heatmap
    ax = axes[0]
    im = ax.imshow(entropy_matrix.T, aspect='auto', origin='lower',
                   extent=[epochs[0], epochs[-1], dropout_rates[0], dropout_rates[-1]],
                   cmap='magma', interpolation='bilinear')
    # Overlay critical point trajectory
    p_cs = [cp['p_c'] for cp in phase_data['critical_points']]
    ax.plot(epochs, p_cs, 'c-o', markersize=3, linewidth=2, label='Critical p_c')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Dropout Rate')
    ax.set_title(f'Prediction Entropy\n{model_name} / {dataset_name}')
    ax.legend(loc='upper left', fontsize=8)
    plt.colorbar(im, ax=ax, label='H(p̄)')

    # 2. Susceptibility heatmap
    ax = axes[1]
    chi_matrix = np.zeros_like(entropy_matrix)
    for i in range(len(epochs)):
        chi_matrix[i] = compute_susceptibility(entropy_matrix[i], np.array(dropout_rates))
    im2 = ax.imshow(chi_matrix.T, aspect='auto', origin='lower',
                    extent=[epochs[0], epochs[-1], dropout_rates[0], dropout_rates[-1]],
                    cmap='inferno', interpolation='bilinear')
    ax.plot(epochs, p_cs, 'c-o', markersize=3, linewidth=2, label='Critical p_c')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Dropout Rate')
    ax.set_title(f'Susceptibility χ = dH/dp\n{model_name} / {dataset_name}')
    ax.legend(loc='upper left', fontsize=8)
    plt.colorbar(im2, ax=ax, label='χ')

    # 3. Critical point trajectory + generalization gap
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(epochs, p_cs, 'b-o', markersize=4, linewidth=2, label='Critical p_c')
    gen_gaps = [phase_data['train_acc'][i] - phase_data['test_acc'][i] for i in range(len(epochs))]
    ax2.plot(epochs, gen_gaps, 'r-s', markersize=3, linewidth=1.5, label='Gen. Gap', alpha=0.7)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Critical Dropout Rate p_c', color='b')
    ax2.set_ylabel('Generalization Gap', color='r')
    ax.set_title(f'Critical Point vs Generalization\n{model_name} / {dataset_name}')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'phase_diagram_{model_name}_{dataset_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_entropy_curves_at_epochs(phase_data, model_name, dataset_name):
    """Plot entropy vs dropout rate at selected epochs (cross-sections of phase diagram)."""
    epochs = phase_data['epochs']
    dropout_rates = np.array(phase_data['dropout_rates'])
    num_classes_entropy = np.log(10) if 'cifar10' in dataset_name else np.log(100)

    # Select ~8 evenly spaced epochs
    indices = np.linspace(0, len(epochs) - 1, 8, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    for i, idx in enumerate(indices):
        color = cmap(i / len(indices))
        entropies = phase_data['phase_diagrams'][idx]['entropy']
        ax.plot(dropout_rates, entropies, '-o', color=color, markersize=3,
                label=f'Epoch {epochs[idx]}', linewidth=1.5)

    ax.axhline(y=num_classes_entropy, color='gray', linestyle='--', alpha=0.5, label='Max entropy (uniform)')
    ax.set_xlabel('Dropout Rate', fontsize=12)
    ax.set_ylabel('Prediction Entropy H(p̄)', fontsize=12)
    ax.set_title(f'Entropy vs Dropout Rate Across Training\n{model_name} / {dataset_name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, f'entropy_curves_{model_name}_{dataset_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_architecture_comparison(all_results):
    """Compare critical point trajectories across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_idx, dataset_name in enumerate(DATASETS):
        ax = axes[dataset_idx]
        for model_name in MODEL_NAMES:
            key = f'{model_name}_{dataset_name}'
            if key not in all_results:
                continue
            phase_data = all_results[key]
            epochs = phase_data['epochs']
            p_cs = [cp['p_c'] for cp in phase_data['critical_points']]
            ax.plot(epochs, p_cs, '-o', markersize=3, linewidth=2, label=model_name)

        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Critical Dropout Rate p_c', fontsize=12)
        ax.set_title(f'Critical Point Trajectories — {dataset_name.upper()}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'architecture_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_correlation_analysis(all_results):
    """Scatter plot: final p_c vs final generalization gap across all runs."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for key, phase_data in all_results.items():
        final_pc = phase_data['critical_points'][-1]['p_c']
        final_gen_gap = phase_data['train_acc'][-1] - phase_data['test_acc'][-1]
        color = 'blue' if 'cifar10' in key else 'red'
        marker = {'simple_cnn': 'o', 'vgg11_nodropout': 's', 'resnet18_nobn': '^', 'mlp': 'D'}
        model_name = key.rsplit('_cifar', 1)[0]
        ax.scatter(final_pc, final_gen_gap, c=color, marker=marker.get(model_name, 'o'),
                   s=100, label=key, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Final Critical Dropout Rate p_c', fontsize=12)
    ax.set_ylabel('Final Generalization Gap', fontsize=12)
    ax.set_title('Critical Point vs Generalization Gap\n(Each point = one model/dataset combo)', fontsize=13)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'correlation_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


# ============================================================
# Main
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Datasets: {DATASETS}")
    print(f"Epochs: {EPOCHS}, Checkpoint every: {CHECKPOINT_EVERY}")
    print(f"Dropout rates: {len(DROPOUT_RATES)} values from {DROPOUT_RATES[0]:.2f} to {DROPOUT_RATES[-1]:.2f}")
    print(f"MC samples per dropout rate: {MC_SAMPLES}")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    total_start = time.time()
    all_results = {}

    for dataset_name in DATASETS:
        for model_name in MODEL_NAMES:
            phase_data = train_and_measure(model_name, dataset_name)
            key = f'{model_name}_{dataset_name}'
            all_results[key] = phase_data

            # Plot per-model figures
            plot_phase_diagram_heatmap(phase_data, model_name, dataset_name)
            plot_entropy_curves_at_epochs(phase_data, model_name, dataset_name)

    # Cross-model comparisons
    print("\n" + "=" * 70)
    print("  Generating comparison plots...")
    print("=" * 70)
    plot_architecture_comparison(all_results)
    plot_correlation_analysis(all_results)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE. Total runtime: {total_time:.0f}s ({total_time/3600:.2f} hours)")
    print(f"  Results in: {RESULTS_DIR}/")
    print(f"{'='*70}")

    # Print summary
    print("\n  SUMMARY OF CRITICAL POINTS:")
    print(f"  {'Model':<25} {'Dataset':<12} {'Final p_c':<12} {'Final χ_max':<12} {'Gen Gap':<10}")
    print(f"  {'-'*70}")
    for key, pd in all_results.items():
        parts = key.rsplit('_cifar', 1)
        model = parts[0]
        dataset = 'cifar' + parts[1]
        final_cp = pd['critical_points'][-1]
        gen_gap = pd['train_acc'][-1] - pd['test_acc'][-1]
        print(f"  {model:<25} {dataset:<12} {final_cp['p_c']:<12.4f} {final_cp['chi_max']:<12.4f} {gen_gap:<10.4f}")


if __name__ == '__main__':
    main()
