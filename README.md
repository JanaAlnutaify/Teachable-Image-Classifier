# Teachable-Image-Classifier


An optimized, conflict-free pipeline to deploy Teachable Machine vision models into modern TensorFlow environments without Keras runtime exceptions.

---

## 🚀 Overview

* **The Challenge:** Keras 3 environments in modern runtimes break legacy `.h5` model workflows due to structural validation mismatches.
* **The Solution:** Runs predictions natively through the lightweight **TensorFlow Lite (TFLite) Interpreter engine** to bypass Keras abstraction layers.
* **Key Benefits:** Ensures 100% environment stability, cross-platform reproducibility, and highly efficient edge execution metrics.

---

## 📁 Repository Structure

* `model_unquant.tflite` — Optimized raw model weights (Floating-point standard).
* `labels.txt` — Class configuration mapping configuration index (`0 cat`, `1 dog`).
* `dogTraining.jpg` — Evaluation input sample image.
* `Teachable_Machine_Inference.ipynb` — Executable Jupyter notebook workspace.
* `output.png` — Execution output screenshot showing results.

---

## ⚡ Technical Core Pipeline

* **Validation:** Automated workspace integrity checks to verify all core asset presence.
* **Preprocessing:** Scales sample input images cleanly to $224 \times 224 \times 3$ target pixels via Bilinear resampling.
* **Normalization:** Maps raw pixel values from standard `[0, 255]` to the expected `[-1, 1]` numeric tensor range.
* **Execution:** Binds data registers directly into the native interpreter to execute fast classification inference.

---

## 📈 Evaluation & Results

* **Predicted Target Class:** `1 dog`
* **Confidence Metric Rating:** `93.12%`

### Inference Output Screenshot:
![Inference Dashboard Metrics](./output.png) 

---

## 🛠️ How to Run Locally

* **Prerequisites:** Install standard dependencies:
  ```bash
  pip install numpy tensorflow pillow
