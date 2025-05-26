# Arabic Sign Language Recognition – Comparative Analysis

A Visual Information Processing (VIP) capstone that builds and benchmarks deep-learning models for recognising Arabic Sign Language (ArSL) hand–gestures captured in still images.

## Contents

1.  Project overview
2.  Repository layout
3.  Quick start
4.  Detailed workflow
5.  Experimental results
6.  Team & acknowledgements

---

## 1  Project overview

Deaf and hard-of-hearing people rely on sign language to communicate, yet automatic ArSL recognition is still an open research problem.  This project:

* collects and cleans a 32-class ArSL image dataset (≈9 k samples) using a custom **YOLOv5** hand-detector (`best.pt`);
* explores two vision architectures
    * **Transfer-learning:** _ResNet-101 V2_ pretrained on ImageNet;
    * **Custom CNN:** 8-layer model engineered from scratch;
* evaluates the models on accuracy, precision, recall, F1 and loss;
* analyses strengths & weaknesses and suggests future work.

Full methodology, equations and figures live in [`Code/code.ipynb`](Code/code.ipynb) and the final report.

---

## 2  Repository layout

```
├── Code/
│   ├── code.ipynb         – main Jupyter notebook (EDA, training, evaluation)
│   ├── best.pt            – YOLOv5 hand-detector checkpoint
│   ├── in/                – raw images (add yours here before cleaning)
│   └── out/               – cleaned, class-sorted images (generated)
├── Final Report.pdf       – written report (same as `report.md`)
├── Presenation.pptx       – presentation slides
├── report.md             – markdown version of the report
└── README.md             – **you are here**
```

---

## 3  Quick start

### 3.1  Environment

```bash
# clone & enter the repo
git clone <your-fork>
cd vip

# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install core dependencies
pip install -r requirements.txt  # generate via the list below
```

Main Python packages
```
numpy pandas matplotlib scikit-learn tensorflow==2.16.1 torch torchvision opencv-python ultralytics
```

### 3.2  Run the notebook

```
jupyter lab  # or jupyter notebook
# open Code/code.ipynb and execute cells sequentially
```

> **Tip:** The first execution of the YOLO cropping cell downloads `ultralytics/yolov5` (~200 MB).

---

## 4  Detailed workflow

1. **Data cleaning**  (`YOLOv5 → detect_and_crop`)
   • Place all raw `.jpg/.png` files in `Code/in/` then run the **Data-cleaning** cell.  Cropped, labelled images are written to `Code/out/<class>/`.

2. **Data preprocessing**
   • Images are resized to 64 × 64, rescaled to [0,1] and split 80 / 10 / 10 into train/val/test using `image_dataset_from_directory`.

3. **Modelling**
   * _ResNet-101 V2_ head replaced by GAP + Dense(32) soft-max; fine-tuned with Adam, LR=1e-4.
   * _Custom CNN_ 8 conv-blocks + FC; trained from scratch.

4. **Evaluation**  – metrics computed with `sklearn`, confusion matrices plotted.

5. **Export**  – best weights stored (`*.h5`) and learning curves saved for the report.

---

## 5  Experimental results (high-level)

| Model            | Val accuracy | Test accuracy | Params |
|------------------|-------------|--------------|--------|
| ResNet-101 V2    | **97 %**    | 96 %          | 44 M   |
| Custom CNN (8-L) | 92 %        | 90 %          | 2 M    |

> Exact numbers may vary; see notebook for full tables & plots.

Key findings
* Transfer learning converges 4× faster and outperforms the scratch model by ~6 %.
* Most confusion occurs between visually similar gestures (e.g. _ain_ vs _ghain_).
* Data imbalance affects minority classes; future work includes augmentation & attention models.

---

## 6  Team & acknowledgements

Group 10 – Multimedia University

| ID        | Name            | Specialisation |
|-----------|-----------------|---------------|
| 1191301456| Kamel Mojtaba   | Game Dev       |
| 1191302190| Amin Ahmed      | Data Science   |
| 1171103208| Obai Ali        | Data Science   |
| 1211300174| Ahmed Al-Nasri  | Data Science   |

*Advisors:* Dr … (add supervisor names)

---

### License

This repository is released for academic, non-commercial use under the MIT License – see `LICENSE`. 