# DNABERT Experiment Validation

The DNABERT command-line workflow is separate from the Flask/classical ML workflow.
These changes improve experiment integrity; they do not demonstrate predictive
accuracy or clinical validity.

## Training

Example in PowerShell, substituting a verified graph and matching annotations:

```powershell
& .\.venv\Scripts\python.exe main.py --gfa verified.gfa --labels verified_labels.csv --model zhihan1996/DNA_bert_3 --kmer 3 --train --test-split 0.3 --val-split 0.2 --seed 42 --output-dir trained_model_run1
```

- Confirm labels identify the actual extracted paths in their extraction order.
  A VCF or CSV filename alone does not establish valid path-level annotations.
- Inputs are grouped by identical encoded token IDs. Conflicting labels within a
  group stop training, including conflicts caused by truncating distinct sequences.
- Stratified test selection reserves 30% of groups. Validation reserves 20% of the
  remaining groups, giving approximately 56% training / 14% validation / 30% test.
  All partitions must contain both classes. Fractions are group-level, so sample
  fractions can differ when duplicate inputs exist.
- Only training inputs update weights. Epoch count is fixed by `--epochs`.
  Validation inputs select the F1 threshold; the untouched test partition supplies
  final metrics. Internal epoch validation is disabled for this workflow.
- Tiny or insufficiently diverse datasets fail explicitly; training data are never
  substituted for test data. The bundled demonstration inputs may be insufficient.
- Output includes model weights, the checkpoint tokenizer, and `experiment.json`.
  The JSON records configuration, package versions, source-file SHA-256 hashes,
  partition indices/class counts, training history, threshold and held-out metrics.
  Use a new output directory for each experiment to preserve previous artifacts.

## Checkpoints and Inference

```powershell
& .\.venv\Scripts\python.exe main.py --gfa verified.gfa --model trained_model_run1 --kmer 3
```

Match `--kmer` and `--max-len` to the training configuration. The saved validation
threshold is reused for displayed predictions and metrics. Token IDs come from the
checkpoint tokenizer, not the project's generated vocabulary. Loading failures,
incompatible vocabularies, and position-limit violations stop execution. A newly
initialized classification head cannot be used for inference before fine-tuning.

Previously saved models without tokenizer artifacts are not automatically trusted:
their original token-ID mapping must be established, or they must be retrained.
Existing model files are not migrated or overwritten by these code changes.

Without verified labels, evaluation is unavailable and no performance scores are
reported. With labels in inference-only mode, independence from the checkpoint's
training data remains the user's responsibility. Having classifier weights does
not prove that a checkpoint was trained correctly or is suitable for clinical use.

## Metrics and Limitations

The experiment record includes accuracy, precision, recall, binary F1, ROC-AUC,
average precision, and the confusion matrix ordered BENIGN / PATHOGENIC.
Average precision summarizes the precision-recall curve; it is not trapezoidal
PR-AUC. ROC-AUC and average precision are recorded as null for single-class
evaluation inputs. Probabilities are not calibrated confidence estimates.

Long inputs currently use the first `max_length - 2` k-mers, with an explicit
truncation warning. Variant-centred windows require reliable variant coordinates;
they are not implemented here. Exact-input grouping does not exclude related
patients, source sequences, or synthetic mutation families across partitions.
Verified provenance and family-level grouping remain necessary before scientific
generalization claims. Source hashes support reproducibility, not source validity.

Further research work: independent curated data, per-gene evaluation with adequate
sample sizes, uncertainty intervals, and a controlled graph-feature ablation against
sequence-only baselines. No graph-aware accuracy improvement is claimed.

## Regression Checks

```powershell
& .\.venv\Scripts\python.exe -m unittest test_dnabert_pipeline test_validation_pipeline -v
```

The DNABERT tests use mocked boundaries and a tiny locally initialized BERT for a
CPU-only training/save/reload check. They do not download a pretrained checkpoint
or establish accuracy on the project's biological data.