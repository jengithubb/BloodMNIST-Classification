# Cleanup and validation

## Changes made

- Kept original notebook files and outputs unchanged in `archive/`.
- Cleared outputs in the runnable notebooks to distinguish a future run from the archive.
- Removed the unused, incorrectly constructed class-weight setup. The original
  training cell already used unweighted `CrossEntropyLoss`; this is the retained baseline.
- Made three-channel normalization explicit and flattened integer labels correctly.
- Added seed 123 and CPU thread control to the CNN; kept Random Forest seed 42.
- Removed redundant data loading and duplicate metric print statements.
- Added CSV export of new aggregate results, separate from historical reference values.
- Renamed the public-facing project to BloodMNIST to match the dataset actually used.

## Verification scope

Python notebook code and report scripts were syntax-checked. The report generator
was executed; local links and archive integrity were checked. Historical metric
tables were compared with saved notebook output. No models were retrained:
PyTorch, torchvision and MedMNIST were unavailable in the preparation environment.
Compatibility of an installed training environment remains to be checked locally.

The fixed seed is a new reproducibility aid, not evidence of exact recovery of the
old stochastic run. The old README remains archived as a source artifact; its
inconsistent method/result claims should not be treated as authoritative.
