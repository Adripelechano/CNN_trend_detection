# Financial time series trend classification using 1D-CNNs

## Summary
This project investigates the application of 1D Convolutional Neural Networks (Conv1D) to predict daily equity price direction ($y \in \{0, 1\}$) using rolling windows of normalized return series. 

Rather than presenting an unrealistic "black-box" model, this repository empirically documents the iterative research pipeline: identifying severe financial noise overfitting, adjusting model capacity, and validating results against the Random Walk Hypothesis.

## Repository structure
* **README.md** - Project overview and key findings.
* **requirements.txt** - Python dependencies and environment specifications.
* **notebooks/cnn_trend_classification.ipynb** - Jupyter Notebook containing data preprocessing, model architecture, training loops, and evaluations.

## Research & Iteration Pipeline

### Phase 1: High capacity baseline and Overfitting identification
* **Model:** Deep 1D-CNN (64 filters, 3-day kernels) trained on a small asset sample (`AAPL`).
* **Observation:** Training accuracy scaled to **97.2%** while validation accuracy collapsed to **37.0%** (with validation loss expanding from 0.71 to 1.07).
* **Diagnosis:** The network memorized non-stationary market noise rather than learning generalized structural dynamics.

### Phase 2: Dataset expansion and regularization
* **Strategy:** Scaled input volume using index data (`SPY`) and simplified architecture (8 filters, Dropout = 0.5).
* **Observation:** Training accuracy (~54.6%) converged tightly with Test accuracy (~53.5%), stabilizing evaluation loss at ~0.69.

## Key Theoretical Insights
1. **Local feature scanning:** Using a `kernel_size=3` acts as a temporal scanner, isolating 3-day micro-regimes (momentum, exhaustion, and bounces) across historical rolling windows.
2. **Efficient market hypothesis (EMH):** Predicting price direction solely from isolated 10-day historical returns hits an informational ceiling near ~53-54% accuracy, consistent with low signal-to-noise ratios in liquid asset returns.

## Future Enhancements
* Incorporating volume features and technical indicators (RSI, MACD) to expand input dimensionality beyond univariate return series.
