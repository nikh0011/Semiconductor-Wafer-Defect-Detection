

## 📈 Model Evaluation & Results

The trained CNN model was evaluated on a test set containing 902 wafer map images.

### Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **91.57%** |
| Test Loss | **0.3499** |
| Macro F1-score | **0.91** |
| Weighted F1-score | **0.91** |

### Class-wise Performance

| Defect Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Center | 0.90 | 0.98 | 0.94 |
| Donut | 0.95 | 0.94 | 0.95 |
| Edge Local | 0.87 | 0.90 | 0.89 |
| Edge Ring | 0.88 | 0.98 | 0.93 |
| Local | 0.95 | 0.76 | 0.84 |
| Scratch | 0.88 | 0.85 | 0.86 |
| Near Full | 0.96 | 1.00 | 0.98 |
| None | 0.97 | 0.87 | 0.92 |
| Random | 0.91 | 0.96 | 0.93 |

The model achieved **91.57% test accuracy** across the 902-image test set.

> Note: Prediction confidence for an individual image is different from overall test accuracy.
