# Bidirectional LSTM for Light-Curve Redshift Prediction

This project predicts spectroscopic redshift from raw multiband astronomical light curves. It extends the previous MLP-based approach by preserving the temporal structure of each observation sequence and processing it with a bidirectional LSTM.

The complete workflow is available in [`Phase3_485.ipynb`](Phase3_485.ipynb).

## Dataset and input representation

The notebook loads the `MultimodalUniverse/plasticc` dataset from Hugging Face in streaming mode.

It processes 7,848 astronomical objects and uses `hostgal_specz` as the continuous regression target. Unlike the previous feature-engineering approach, the notebook keeps the original sequence of observations instead of reducing each light curve to summary statistics.

The input representation is:

- Up to 2,112 time steps per object
- Six photometric channels: `Y`, `g`, `i`, `r`, `u`, and `z`
- Zero-padding for shorter sequences
- Per-band Z-score normalization fitted using the training data

The data is split reproducibly with random seed `42`:

- Training set: 5,493 samples
- Validation set: 1,177 samples
- Test set: 1,178 samples

## Main model

The primary architecture is an `AdvancedTriadicBiLSTM`:

```text
Input sequence [2112, 6]
        ↓
Bidirectional LSTM, hidden size 96
        ↓
Global max pooling over time
        ↓
Fully connected layer: 192 → 48
        ↓
ReLU
        ↓
Output layer: 48 → 1 redshift prediction
```

The model contains 89,185 trainable parameters. Global max pooling produces a fixed-size representation while reducing the influence of the long padded regions in shorter sequences.

## Experiments

### Raw SGD baseline

The manually trained BiLSTM baseline records:

- Test MSE: `0.1362`
- Test R²: `0.0144`

The notebook compares this result with the previous MLP baseline. The raw BiLSTM does not automatically outperform the simpler model, showing that a more complex architecture requires suitable optimization and regularization.

### Optimizer and learning-rate comparison

The notebook compares SGD with momentum and Adam using learning rates of `0.01` and `0.001`. Adam with a learning rate of `0.001` provides the strongest validation performance among the tested configurations.

### Regularization and early stopping

The notebook evaluates:

- No regularization
- Dropout with probability `0.3`
- Weight decay of `1e-4`
- Dropout combined with weight decay

The recorded ablation results show that dropout is more useful than weight decay for this setup, while combining both regularizers can restrict the model too strongly.

Early stopping is then applied with patience `10`. The best model snapshot is recorded at epoch `31`, training stops at epoch `41`, and the held-out test results are:

- Test MSE: `0.0913`
- Test R²: `0.3392`

### Time-series augmentation

The augmentation experiment simulates realistic observation problems by applying:

- Random masking of time steps to represent missing observations
- Small Gaussian flux jitter to represent measurement noise

In the recorded run, this augmented configuration achieved a test MSE of `0.1184` and an R² of `0.1435`. The notebook also discusses which transformations preserve the physical meaning of the redshift label and which could corrupt it.

## Transfer learning and self-supervised learning

### Sequence autoencoder transfer

The notebook pre-trains a BiLSTM encoder as a sequence autoencoder using the light curves without labels. The encoder is then transferred to the redshift-regression task and fine-tuned with labeled data.

The recorded comparison shows that this transfer setup did not improve the full-data baseline:

| Training scheme | Test MSE | Test R² |
| --- | ---: | ---: |
| From scratch | 0.1362 | 0.0146 |
| Self-supervised transfer | 0.1384 | -0.0013 |

The notebook also evaluates label efficiency. Transfer learning is most helpful when only a small fraction of labels is available, while its advantage largely disappears when more labeled samples are provided.

### Masked-segment reconstruction

The final experiment masks a continuous 15% segment of each light curve and trains an encoder-decoder to reconstruct the missing observations. This pretext task is designed to encourage the encoder to learn temporal continuity and cross-band relationships.

A frozen encoder followed by an MLP probe is compared with a fully supervised model:

| Representation scheme | Test MSE |
| --- | ---: |
| Supervised model trained from scratch | 0.0822 |
| Frozen self-supervised encoder with MLP probe | 0.1345 |

The supervised model performs better in this experiment, suggesting that representations optimized only for reconstruction do not necessarily capture the fine-grained information needed for precise redshift estimation.

## Technologies

- Python
- Jupyter Notebook
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Hugging Face Datasets

## Running the notebook

Install the required packages:

```bash
pip install datasets==3.6.0 pandas numpy matplotlib seaborn scikit-learn torch
pip install git+https://github.com/MultimodalUniverse/MultimodalUniverse.git
```

Open `Phase3_485.ipynb` in Jupyter or Google Colab and execute the cells in order. The dataset is downloaded from Hugging Face, so an internet connection is required. A Hugging Face token is optional for this public dataset but may provide higher request limits.

## Project status

This is an academic deep-learning experiment. The notebook documents sequential data preparation, model development, optimization, regularization, transfer learning, and self-supervised representation learning. It is not packaged as a reusable training pipeline or production inference service.

