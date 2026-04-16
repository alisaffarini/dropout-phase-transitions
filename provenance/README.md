# Provenance — Dropout Phase Transitions

Raw run data from the experiment pipeline.

## Run History

| Run | Description | Key Output |
|-----|-------------|------------|
| `run_094_dropout_phase` | Initial dropout experiment | experiment.py, output.log |
| `run_096_phase_transitions` | **Main run** — full phase transition experiments | All iteration code, conversation log, literature review, training outputs |
| `original_data/` | Result JSONs from burn-tokens paper directory | 7 JSON files (4 CIFAR-10 + 3 CIFAR-100, missing MLP CIFAR-100) |

## Key Files

- `run_096_phase_transitions/conversation_log.md` — AI agent conversation during research
- `run_096_phase_transitions/literature_report.md` — Literature review
- `run_096_phase_transitions/experiment_best.py` + `experiment_best_output.txt` — Best iteration
- `run_096_phase_transitions/output.log` — Full training output
- `original_data/*.json` — Result files for each model-dataset combination

## Notes

- MLP on CIFAR-100 was never run (7/8 combinations completed)
- VGG-11 on CIFAR-100 failed to train above chance (0.5%)
- Architecture descriptions in paper have been corrected to match code: MLP=1024-512, SimpleCNN=64-64-128-128
- Run 096 contains downloaded MNIST data (excluded from this provenance copy)
