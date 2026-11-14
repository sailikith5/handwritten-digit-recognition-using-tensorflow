# Handwritten Digit Recognition with TensorFlow

A simple neural network trained on the MNIST dataset that recognizes handwritten digits (0-9).

## How it works

1. Loads and normalizes the MNIST dataset.
2. Trains a small network: `Flatten -> Dense(128, relu) -> Dense(128, relu) -> Dense(10, softmax)` for 3 epochs.
3. Saves the model to `digits.keras` (later runs load it instead of retraining).
4. Predicts the digits in `digits/1.png` ... `digits/5.png` and shows each image.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Use your own digits

Draw a digit in black on a white background, save it as a 28x28 PNG (e.g. in Paint), and put it in the `digits/` folder as `1.png` to `5.png`.
