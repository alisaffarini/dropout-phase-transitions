#!/usr/bin/env python3
"""
Dropout Phase Transitions - Multi-seed runner (3 seeds)
Wraps the original experiment to run seeds 42, 43, 44 and aggregate results.
"""

import os
import json
import time
import sys
import importlib.util
import numpy as np

SEEDS = [42, 43, 44]
RESULTS_DIR = 'results_multiseed'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import the original experiment module
spec = importlib.util.spec_from_file_location("experiment", 
    os.path.join(os.path.dirname(__file__), "experiment_v1.py"))
exp = importlib.util.module_from_spec(spec)

# We'll directly use the functions after loading
spec.loader.exec_module(exp)

def run_all_seeds():
    """Run all model-dataset combos for each seed, then aggregate."""
    total_start = time.time()
    
    all_seed_results = {}  # {seed: {model_dataset: phase_data}}
    
    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")
        
        # Override the global seed
        exp.SEED = seed
        seed_dir = os.path.join(RESULTS_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        exp.RESULTS_DIR = seed_dir
        
        seed_results = {}
        for dataset_name in exp.DATASETS:
            for model_name in exp.MODEL_NAMES:
                key = f'{model_name}_{dataset_name}'
                
                # Check if already done
                save_path = os.path.join(seed_dir, f'{key}.json')
                if os.path.exists(save_path):
                    print(f"  [SKIP] {key} seed {seed} - already exists")
                    with open(save_path) as f:
                        seed_results[key] = json.load(f)
                    continue
                
                phase_data = exp.train_and_measure(model_name, dataset_name)
                seed_results[key] = phase_data
        
        all_seed_results[seed] = seed_results
    
    # Aggregate across seeds
    print(f"\n{'='*70}")
    print("  AGGREGATING ACROSS SEEDS")
    print(f"{'='*70}")
    
    aggregated = {}
    
    for dataset_name in exp.DATASETS:
        for model_name in exp.MODEL_NAMES:
            key = f'{model_name}_{dataset_name}'
            
            # Collect per-seed final metrics
            final_pcs = []
            final_chi_maxs = []
            final_test_accs = []
            final_train_accs = []
            final_gen_gaps = []
            
            for seed in SEEDS:
                if key not in all_seed_results.get(seed, {}):
                    print(f"  WARNING: Missing {key} for seed {seed}")
                    continue
                pd = all_seed_results[seed][key]
                final_cp = pd['critical_points'][-1]
                final_pcs.append(final_cp['p_c'])
                final_chi_maxs.append(final_cp['chi_max'])
                final_test_accs.append(pd['test_acc'][-1])
                final_train_accs.append(pd['train_acc'][-1])
                final_gen_gaps.append(pd['train_acc'][-1] - pd['test_acc'][-1])
            
            if len(final_pcs) == 0:
                continue
                
            aggregated[key] = {
                'n_seeds': len(final_pcs),
                'final_pc': {
                    'mean': float(np.mean(final_pcs)),
                    'std': float(np.std(final_pcs)),
                    'values': final_pcs
                },
                'final_chi_max': {
                    'mean': float(np.mean(final_chi_maxs)),
                    'std': float(np.std(final_chi_maxs)),
                    'values': final_chi_maxs
                },
                'final_test_acc': {
                    'mean': float(np.mean(final_test_accs)),
                    'std': float(np.std(final_test_accs)),
                    'values': final_test_accs
                },
                'final_gen_gap': {
                    'mean': float(np.mean(final_gen_gaps)),
                    'std': float(np.std(final_gen_gaps)),
                    'values': final_gen_gaps
                }
            }
    
    # Save aggregated results
    agg_path = os.path.join(RESULTS_DIR, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    total_time = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY ({len(SEEDS)} seeds)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Dataset':<12} {'p_c':<18} {'χ_max':<18} {'Gen Gap':<18}")
    print(f"  {'-'*85}")
    for key, agg in aggregated.items():
        parts = key.rsplit('_cifar', 1)
        model = parts[0]
        dataset = 'cifar' + parts[1]
        pc = agg['final_pc']
        chi = agg['final_chi_max']
        gg = agg['final_gen_gap']
        print(f"  {model:<25} {dataset:<12} {pc['mean']:.3f}±{pc['std']:.3f}      "
              f"{chi['mean']:.2f}±{chi['std']:.2f}      {gg['mean']:.3f}±{gg['std']:.3f}")
    
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time/3600:.2f} hours)")
    print(f"  Results in: {RESULTS_DIR}/")
    print(f"  Aggregated: {agg_path}")

if __name__ == '__main__':
    run_all_seeds()
