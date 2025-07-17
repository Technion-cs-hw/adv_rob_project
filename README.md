# Adversarial Robustness of Transformer-Based Image Captioning Models
This project was done as fulfillment of the requirements of the 236874 Technion course on computer vision, in the winter semester of the academic year 2024/25, under the supervision of Ori Bryt.


It benchmarks state-of-the-art image captioning models and analyzes their robustness against adversarial attacks. We explore how well models such as BLIP and ExpansionNet handle both natural and adversarial perturbations, and propose two defense mechanisms: **Gaussian blur** and a **caption confidence metric**. The project also includes a classification-based sanity check and a human evaluation study.

## 📌 Features

- Benchmarking of popular image captioning models (BLIP, ExpansionNet)
- Evaluation on both natural corruptions (blur, noise, geometric transforms) and adversarial perturbations
- Defense methods:
  - **Gaussian blur** applied to input images
  - **Caption confidence** based on inverse perplexity of output logits
- Human study comparing captioning performance under noise
- Sanity check using CLIP encoder + linear classifier on Caltech-101
- Qualitative and quantitative results

## 📁 Project Structure

```
.
├── adversarial_attacks/        # adversarial attack results
├── benchmarks/                 # basic perturbation results
├── projectCode/                # Core model logic and dataset wrappers
│   ├── CartoonDataset.py       # Cartoon dataset loader (used for OOD evaluation)
│   ├── CocoDataset.py          # COCO dataset wrapper
│   ├── Evaluator.py            # Evaluation pipeline
│   ├── ModelLoader.py          # Unified model loading (BLIP, ExpansionNet, ...)
│   ├── Perturbator.py          # Natural image corruptions (blur, noise, transforms)
│   ├── attack.py               # Attack entry point
│   ├── imagenet100.py          # ImageNet100 class utility (if used)
│   ├── imports.py              # Centralized imports and configs
│   ├── utils.py                # Helper functions
│   ├── __init__.py
│   └── __pycache__/
├── Blip_Benchmark.ipynb          # BLIP model benchmarking notebook
├── Blip_Classification.ipynb     # Sanity check: BLIP classifier on Caltech-101
├── ExpansionNet_Attack.ipynb     # Adversarial attack on ExpansionNet
├── ExpansionNet_benchmark.ipynb  # Benchmarking ExpansionNet
├── ExpansionNet_confidence.ipynb # Confidence-based defense on ExpansionNet
├── Plot_results.ipynb            # Visualization of evaluation metrics and model behavior
├── README.md
```


## 📊 Results

- Adversarial perturbations (even small) cause major semantic degradation in captioning.
- Blur defends surprisingly well when dealing with captioning tasks.
- Caption confidence is useful for filtering unreliable predictions.
- Classifier-based sanity check confirms literature: transformer classifiers are highly vulnerable to adversarial attacks.
- Human study shows performance degrades with noise, but humans are more robust than models.

## 📂 External Resources Required

⚠️ **Important:** The following resources must be downloaded separately and placed in the specified directories:

- **ExpansionNet_v2 weights:**  
  Place in: `models/ExpansionNet_v2/`

- **COCO annotations:**  
  Place in: `annotations/annotations/`

- **Caltech-101 dataset:**  
  Place in: `datasets/caltech101/`

## 🧪 Reproducibility

- All experiments are run with fixed seeds.
- We provide scripts for reproducibility across:
  - Natural corruption evaluation
  - AutoAttack generation
  - Blur and confidence-based defenses
  - Classifier sanity check

## 📝 Citation

If you use this project or its components in your research, please consider citing our work.

---
