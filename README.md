# Teachable-Image-Classifier

An optimized, production-ready implementation to deploy Teachable Machine vision models into modern TensorFlow environments without Keras runtime conflicts.

---

## 🚀 Project Overview

* **The Challenge:** Modern Keras 3 environments in Google Colab break legacy `.h5` workflows due to strict structural validation mismatches (e.g., `DepthwiseConv2D` issues).
* **The Solution:** Bypasses fragile high-level Keras layers completely by running predictions natively through the lightweight **TensorFlow Lite (TFLite) Interpreter engine**.
* **Key Benefits:** Ensures 100% environment stability, cross-platform reproducibility, and ultra-efficient execution metrics.

---

## 📁 Repository Structure

* `model_unquant.tflite` — Optimized raw model weights exported via TFLite floating-point standard.
* `labels.txt` — Class configurations index (`0 cat`, `1 dog`).
* `dogTraining.jpg` — Input sample image used for pipeline evaluation.
* `Teachable_Machine_Inference.ipynb` — Executable Jupyter notebook workspace.

---

## ⚡ Technical Core Pipeline

* **Workspace Validation:** Automated file checks to verify asset presence before initializing layers.
* **Resolution Downscaling:** Resizes input images cleanly to $224 \times 224 \times 3$ target matrices using Bilinear resampling.
* **Data Normalization:** Maps tensor pixels from standard `[0, 255]` to the expected `[-1, 1]` numeric range.
* **Edge Inference Execution:** Binds normalized data into memory and runs fast, native classification.

---

## 📈 Evaluation & Results

* **Target Classification:** `1 dog`
* **Model Confidence Rating:** `93.12%`

### Execution Output Screenshot:
![Inference Dashboard Metrics](./screenshot.png)

---

## 🛠️ How to Run Locally

* **Prerequisites:** Install standard deep learning dependencies before execution:
  ```bash
  pip install numpy tensorflow pillow
