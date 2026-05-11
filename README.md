# Dropout as a Phase Transition

We treat the dropout rate as a control parameter and measure prediction entropy on a held-out set as a function of $p$, building two-dimensional phase diagrams over (epoch, dropout rate). The critical rate $p_c = \arg\max_p \, dH/dp$ and the peak susceptibility $\chi_{\max}$ are read off these diagrams.

## Findings

- Dropout induces a sharp transition between confident (low-entropy) and uniform (high-entropy) prediction regimes.
- $\chi_{\max}$ grows with depth: MLP $\approx 1.4$, SimpleCNN $\approx 4.4$, ResNet-18 / VGG-11 $\approx 7.0$--$7.2$ on CIFAR-10.
- The shape of the diagram separates three model--task regimes: good fit (stable $p_c$, high $\chi_{\max}$), overfitting ($p_c$ drifts down during training), and underfitting (anomalously high $p_c$ from capacity underuse).
- ResNet-18 on CIFAR-10 vs.\ CIFAR-100: $p_c$ drifts $0.84 \to 0.72$ while the generalization gap grows to $39.7\%$.
- Two seeds per configuration; $p_c$ is identical across seeds for 6/7 valid configurations.

## Layout

```
paper/       LaTeX source
results/     JSON results for every model--dataset combination
experiment/  Python training and sweep scripts
figures/     Generated plots used in the paper
```
