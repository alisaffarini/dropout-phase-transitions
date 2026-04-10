# Dropout as a Phase Transition: Critical Points in Neural Network Regularization

**Author:** Ali Saffarini, Harvard University

## Key Results

- Dropout induces **sharp phase transitions** in prediction entropy, analogous to thermodynamic transitions
- Peak susceptibility χ_max scales monotonically with depth: MLP (1.4) < CNN (4.3) < ResNet/VGG (7.2-7.7)
- Three distinct model-task fit regimes: good fit (stable p_c), overfitting (drifting p_c), underfitting (high p_c)
- ResNet-18 on CIFAR-10 vs CIFAR-100: p_c drifts from 0.84 → 0.72 as generalization gap grows to 39.7%

## Structure

```
paper/       LaTeX source
results/     JSON results for all model-dataset combinations
experiment/  Python experiment scripts
```

## Status

Currently single-seed results. Additional seeds would strengthen the statistical claims.

## Citation

Target venue: ICLR 2027
