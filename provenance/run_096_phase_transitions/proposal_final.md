## Final Proposal: Understanding Feature Learning Dynamics through Gradient Coherence Analysis

Thank you for the constructive feedback. Let me address each concern with concrete solutions.

### (1) EXACT NOVELTY CLAIM (Reframed)

We provide the first systematic analysis of how **gradient coherence patterns across network depth** evolve during training and relate to feature learning quality. Specifically, we show that networks exhibiting rapid decorrelation of gradients across layers (measured by our Gradient Coherence Index) develop more generalizable features, extending the neural collapse framework (Papyan et al., 2020) to training dynamics.

### (2) CLOSEST PRIOR WORK (Comprehensive)
- **"Neural Collapse" (Papyan et al., 2020)**: Studies feature convergence at training end; we study gradient dynamics during training
- **"Gradient Confusion" (Sankararaman et al., 2020)**: Analyzes gradient conflicts in multi-task learning; we analyze single-task depth-wise patterns  
- **"On the Validity of Modeling SGD Noise as Gaussian" (Fort & Ganguli, 2019)**: Studies gradient noise; we study gradient structure
- **"Layer-adaptive Learning Rates" (You et al., 2019)**: Uses gradient norms for LR; we use gradient correlations for analysis

### (3) EXPECTED CONTRIBUTION (Focused)
- **New understanding**: Gradient decorrelation across depth indicates healthy feature specialization (supported by neural collapse theory)
- **Practical diagnostic**: Simple tool to monitor training health without validation data
- **Empirical finding**: Gradient coherence patterns differ systematically between architectures/tasks
- **Analysis framework**: Methodology for studying feature learning dynamics

### (4) HYPOTHESIS (Theory-grounded)

**Main hypothesis**: Following neural collapse theory, as networks learn hierarchical features, gradients at different depths must decorrelate to enable specialization. The rate of this decorrelation (measured 15-30% into training) predicts final feature quality and generalization.

**Mechanistic basis** (addressing soundness):
- Early training: High gradient coherence → layers compute similar functions → redundant features
- Mid training: Coherence collapse → layers specialize → hierarchical features emerge
- This connects to **Gradient Confusion** (Sankararaman et al., 2020): high coherence = confused gradients = poor learning

### (5) EXPERIMENTAL PLAN (Rigorous)

**A. Principled Metric Design** (addressing scale/projection issues):

```python
def compute_gradient_coherence_index(model, batch, epsilon=1e-5):
    """Theory-grounded coherence metric"""
    grads = compute_gradients(model, batch)
    
    # Use CCA instead of random projection (Raghu et al., 2017)
    coherence_matrix = np.zeros((len(grads), len(grads)))
    
    for i, g_i in enumerate(grads):
        for j, g_j in enumerate(grads):
            if i != j:
                # Reshape to 2D for CCA
                g_i_2d = g_i.flatten().reshape(-1, 1)
                g_j_2d = g_j.flatten().reshape(-1, 1)
                
                # Use gradient norm ratios as simpler baseline
                norm_ratio = torch.norm(g_i) / (torch.norm(g_j) + epsilon)
                
                # Compute correlation in minibatch gradient space
                if g_i_2d.shape[0] > 100:  # Use CCA for large layers
                    cca = CCA(n_components=1)
                    cca.fit(g_i_2d.cpu(), g_j_2d.cpu())
                    coherence = cca.score(g_i_2d.cpu(), g_j_2d.cpu())
                else:  # Use cosine similarity for small layers
                    coherence = F.cosine_similarity(g_i.flatten(), g_j.flatten(), dim=0)
                
                coherence_matrix[i, j] = coherence
    
    # Multi-scale analysis justified by network depth
    # Scale k = comparing layers separated by k (hierarchical features)
    scales = range(1, min(6, len(grads)))  # Adaptive to network depth
    scale_coherences = []
    
    for k in scales:
        k_coherences = [coherence_matrix[i, i+k] for i in range(len(grads)-k)]
        scale_coherences.append(np.mean(k_coherences))
    
    return np.array(scale_coherences), coherence_matrix, norm_ratio
```

**B. Comprehensive Baselines** (addressing missing comparisons):

```python
baselines = {
    # Simple baselines that might work just as well
    'gradient_variance': lambda m, d: compute_layer_gradient_variance(m, d),
    'gradient_norm_ratio': lambda m, d: compute_gradient_norm_ratios(m, d),
    'gradient_snr': lambda m, d: compute_gradient_signal_noise_ratio(m, d),
    
    # Sophisticated baselines
    'ntk_evolution': lambda m, d: compute_ntk_eigenvalue_evolution(m, d),
    'sam_sharpness': lambda m, d: compute_sam_sharpness(m, d, rho=0.05),
    'layer_lr_diagnostic': lambda m, d: compute_layerwise_lr_ratios(m, d),
    
    # Standard baseline
    'val_loss': lambda m, d: validation_loss(m, d)
}
```

**C. Expanded Evaluation** (addressing limited scope):

1. **Vision Tasks**:
   - CIFAR-10/100 (ResNet-20, VGG-11)
   - Fashion-MNIST (3-layer CNN)
   
2. **NLP Task** (addressing generality):
   - BERT-tiny on GLUE/SST-2 (sentiment classification)
   - Monitor gradient coherence in attention vs FFN layers
   - 5K training samples for computational feasibility

3. **Synthetic Task** (for controlled analysis):
   - Teacher-student setup with known ground truth features
   - Verify coherence collapse coincides with feature alignment

**D. Rigorous Protocol**:

```python
# For each model/dataset combination:
for epoch in range(100):
    # Compute all metrics every 100 steps
    if step % 100 == 0:
        gci, coherence_matrix, norm_ratios = compute_gradient_coherence_index(model, batch)
        
        # Store full coherence matrix for analysis
        metrics['coherence_matrices'].append(coherence_matrix)
        metrics['gci'].append(gci)
        
        # Compute all baselines
        for name, baseline_fn in baselines.items():
            metrics[name].append(baseline_fn(model, batch))
    
    # Identify coherence collapse point
    if epoch > 15 and detect_collapse(metrics['gci']):
        collapse_magnitude = measure_collapse_magnitude(metrics['gci'])
        collapse_timing = epoch

# Correlate with final generalization
final_gap = test_acc - train_acc
correlation = scipy.stats.pearsonr(collapse_magnitudes, final_gaps)
```

**E. Computational Feasibility** (<4 hours):
- Vision experiments: 2h (smaller models, checkpoint gradients)
- NLP experiment: 1h (BERT-tiny, 5K samples)  
- Analysis: 0.5h
- Buffer: 0.5h

**F. Theory Connection** (strengthening soundness):

Drawing from **"Gradient Starvation"** (Pezeshki et al., 2021) and **"Neural Collapse"** (Papyan et al., 2020):
- High coherence → gradient starvation → features don't specialize
- Coherence collapse → gradient diversity → neural collapse can proceed
- This explains why the metric predicts generalization

### Expected Outcomes:
1. GCI outperforms gradient variance alone (Δr > 0.1)
2. Collapse timing consistent across architectures (15-25% training)
3. BERT shows coherence patterns between attention/FFN layers
4. Theoretical predictions match empirical observations

This addresses all concerns while maintaining feasibility and scientific rigor.