# Saved model evidence

These files are existing training and evaluation outputs, copied without rerunning training.

- Clean YOLO11n detector: 45 completed training epochs; validation-selected threshold 0.35.
- Held-out test: 167 images and 316 annotated objects; precision 75.43%, recall 82.59%, F1 78.85%.
- YOLO11s-seg water model: 200 completed training epochs. Its CSV records training/validation metrics, not an independent field evaluation.
- The detector test has no background-only images; image-level false-alarm rate is unknown. The tyre class has only three test objects.
- `args.yaml` paths refer to the original Linux training container. Model checkpoints and data preparation scripts are documented in the GitHub repository.
- The gallery's tyre photo comes from the training split and demonstrates the interface. The test report uses the held-out test split.
- The clean small detector and stock-inclusive detector reports, configurations, and logs are included for the research report's comparison. The stock-inclusive experiment uses a different test split.
- Nano water training logs and configuration are also included. Both water CSVs contain validation metrics; neither is an independent test report.
