# Reliable Calibration Transfer for Lightweight YOLO Detectors Under Unseen Image Corruptions

[![Target: IEEE Transactions](https://img.shields.io/badge/Target-IEEE%20Transactions%20%2F%20ICCV%2FW-blue?style=for-the-badge)](https://github.com/Upasana-Mukherjee-777/YOLO_Calibration)
[![Outcome: B (Equivalence Shown)](https://img.shields.io/badge/Outcome-B%20(Equivalence%20Shown)-emerald?style=for-the-badge)](https://github.com/Upasana-Mukherjee-777/YOLO_Calibration)
[![Models: YOLOv8n & YOLO11n](https://img.shields.io/badge/Models-YOLOv8n%20%7C%20YOLO11n-purple?style=for-the-badge)](https://github.com/Upasana-Mukherjee-777/YOLO_Calibration)
[![Zero Retraining](https://img.shields.io/badge/Zero--Retraining-0.51%20ms%2F1k%20dets-cyan?style=for-the-badge)](https://github.com/Upasana-Mukherjee-777/YOLO_Calibration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Author:** Upasana Mukherjee, Mouli Mukherjee, Adi Ratnam, Prem Kumar Jha, Rudranil Chakraborty 
> **Project Repository:** [https://github.com/Upasana-Mukherjee-777/YOLO_Calibration](https://github.com/Upasana-Mukherjee-777/YOLO_Calibration)  
> **Interactive Demonstration Dashboard:** [https://upasana-mukherjee-777.github.io/YOLO_Calibration/](https://upasana-mukherjee-777.github.io/YOLO_Calibration/)  

---

## 📌 1. Project Overview & Research Proposal

Lightweight object detectors such as **YOLOv8n** (3.2M parameters) and **YOLO11n** (2.6M parameters) are widely deployed at the edge in autonomous driving, drone surveillance, and robotic inspection. However, real-world deployment frequently introduces environmental corruptions—such as **sensor noise, defocus blur, motion blur, brightness variations, fog, and snow**.

While detection accuracy (mAP, recall) degrades under distribution shifts, a far more dangerous failure mode is **confidence miscalibration**: detectors emit high-confidence predictions on hallucinated or poorly localized boxes, rendering confidence scores untrustworthy for safety-critical edge systems. Retraining detectors for every deployment environment is computationally prohibitive.

### Central Research Question:
$$\mathbf{Can\ post\text{-}hoc\ confidence\ calibration\ fitted\ on\ observed\ corruptions\ reliably\ transfer\ to\ previously\ unseen\ corruption\ families\ without\ retraining\ or\ degrading\ localization\ quality?}$$

### Four Core Research Hypotheses:
1. **$H_1$ (Calibration Transferability):** Post-hoc calibration (Platt scaling, Isotonic regression, Multivariate Logistic) fitted on known corruption families transfers to held-out unseen families, achieving detection ECE statistically equivalent to an in-distribution Oracle calibrator within a margin $\epsilon = 0.01$.
2. **$H_2$ (The Localization-Aware Paradox):** Standard calibrators fitted only on binary detection correctness ($\mathbb{I}[\text{IoU} \ge 0.5 \land \hat{y} = y]$) severely degrade Localization-Aware Calibration Error (LaECE) by overconfidently scaling detections whose bounding boxes remain loose.
3. **$H_3$ (Localization-Aware & Conformal Correction):** Jointly optimizing calibrators on detection correctness scaled by IoU ($\text{correct} \times \text{IoU}$) resolves the LaECE paradox, while Conformal Risk Control (CRC) guarantees finite-sample coverage under severe corruption regimes.
4. **$H_4$ (Edge Efficiency Trade-Off):** Test-time stochastic sampling (Monte Carlo Dropout) incurs a prohibitive $>2.3\times$ latency penalty on edge hardware while offering negligible calibration transfer gains over ultra-lightweight ($<0.6$ ms/1k detections) post-hoc mappings.

---

## 🔍 2. The 2025–2026 Novelty Matrix

Our research gap is verified against state-of-the-art publications (2020–2026):

| Dimension | Huseljic et al. (IJCV 2025) | Ries et al. (PMLR 2026) | Zeevi et al. (CVPR 2025) | Yang et al. (CVPRW 2026) | Li et al. (COUNTS, CVPR 2025) | Kuang et al. (PR 2024) | Oksuz et al. (CVPR 2023) | **Our Work (Mukherjee 2026)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Detector Architecture** | Faster R-CNN, Cascade (Two-stage) | YOLOv5, Faster R-CNN | ResNet, ViT (Classifier) | Vision Transformers | 20+ Heavy Detectors | YOLOv3, Faster R-CNN | Faster R-CNN, RetinaNet | **YOLOv8n & YOLO11n (Nano 1-Stage)** |
| **Shift Paradigm** | OOD / In-domain | Autonomous Driving Shift | Feature-Map Shift | Test-Time Stress | 14 Natural Shifts | In-domain / Generalization | Clean COCO / Pascal | **Held-Out Corruption Families (19 conds)** |
| **Transfer Evaluated** | In-domain recalibration | Cross-city domain shift | No transfer evaluation | Dynamic test compute | Benchmark ranking | In-domain stochastic | In-domain calibration | **Held-Out Family Transfer vs. Severity** |
| **Uncertainty Target** | Class Probability Vector | Class Sets & Box Intervals | Feature Entropy | Predictive Entropy | mAP / Brier Score | DropBlock Spatial Variance | Joint Conf & Localization | **Joint D-ECE, LaECE & Conformal CRC** |
| **Retraining?** | Yes (Fine-tuning layers) | No (Quantile split) | No (Test-time feature) | No (Compute scaling) | Yes (Retrained models) | Yes (DropBlock training) | No (Post-hoc) | **No (Zero-Retraining Post-Hoc)** |
| **Inference Overhead** | Low (<1 ms) | Low (<1 ms) | Moderate (1.4x) | High (2.0x - 5.0x) | N/A (Benchmark) | High (5x - 10x passes) | Negligible (<1 ms) | **Ultra-Low (0.51 ms/1k) vs MC (2.31x)** |
| **Equivalence Tested?**| None (Point metrics) | Exact coverage ($1-\alpha$)| None (AUROC / Brier) | None (Compute trade-off) | None (Benchmark) | None (Variance metrics) | None (Class-average ECE) | **Formal 95% Bootstrap CI ($\pm 0.01$)** |
| **Localization Paradox**| Excluded 1-stage heads | Addressed via box intervals| N/A (Classification) | N/A (Classification) | No (Accuracy focus) | Box variance measured | Addressed in-domain | **Uncovered & Resolved in Cross-Family** |

---

## 🧪 3. Disjoint Experimental Protocol & Splits

To strictly eliminate data leakage, we evaluate **5,000 COCO val2017 images** partitioned into:
* **Fit Split:** 40% (2,000 images) — Calibrator training.
* **Validation Split:** 20% (1,000 images) — Calibrator model selection.
* **Test Split:** 40% (2,000 images) — Final evaluation (never seen during fitting or selection).

```
                               ┌────────────────────────────────────────────────────────┐
                               │       COCO val2017 Benchmark (5,000 Images)            │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                               ┌──────────────────────────┴─────────────────────────────┐
                               │  19 ImageNet-C Conditions (Clean + 6 types x 3 sevs)   │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
          ┌──────────────────────┬────────────────────────┼────────────────────────┬──────────────────────┐
          ▼                      ▼                        ▼                        ▼                      ▼
  Split S1 (Weather)     Split S2 (Noise)         Split S3 (Brightness)    Split S4 (Blur)        Split S5 (Severity)
  Fit: Clean,Noise,Brigt Fit: Clean,Blur,Weather  Fit: Clean,Noise,Weather Fit: Clean,Bright,Weat Fit: Sev 1, 5 (All)
  Val: Blur              Val: Brightness          Val: Blur                Val: Noise             Val: Cross-Image
  Test: Fog & Snow       Test: Gaussian Noise     Test: Brightness         Test: Defocus/Motion   Test: Sev 3 (Held-Out)
```

---

## 📊 4. Key Scientific Discoveries

### 1. Post-Hoc Calibration Transfer Works (Outcome B Verified)
* Across all 4 family splits and both YOLOv8n and YOLO11n, Platt scaling and Box-Aware Multivariate Logistic drop test ECE from **0.1143 to 0.0098**.
* The transfer gap against an in-distribution Oracle is:
  $$\Delta_{\text{ECE}} = +0.0002, \quad 95\% \text{ CI: } [-0.0050, +0.0060]$$
* Because the 95% CI falls strictly within the practical equivalence margin of **$[-\epsilon, +\epsilon] = [-0.01, +0.01]$**, **Outcome B (Equivalence) is formally demonstrated**. Post-hoc calibration does transfer across unseen corruptions without detector retraining.

### 2. The Localization-Aware Calibration Paradox Uncovered
* Standard calibrators minimize classification log loss on binary correctness ($\text{correct} \in \{0, 1\}$).
* While detection ECE improves by **91.4%**, **Localization-Aware Calibration Error (LaECE) explodes by +566%** (increasing from **0.0152 to 0.1013**, $p < 0.001$).
* Calibrated detectors assign high confidence to poorly localized bounding boxes.
* **The Solution:** Our proposed **Localization-Aware Isotonic Calibrator (`loc_isotonic`)** fits directly to the joint target $\text{correct} \times \text{IoU}$, preserving LaECE at **0.0097** ($\Delta = -0.0055$, no degradation).

### 3. Edge Latency Profiling: Why MC-Dropout Fails on Micro-Hardware
* **YOLOv8n Baseline Latency:** $21.12$ ms per image (88.16 MB RAM).
* **Platt / Loc-Isotonic Calibration:** Adds **$0.51$ ms per 1,000 detections** ($1.00002\times$ overhead).
* **Monte Carlo Dropout ($T=6$):** Increases latency to **$48.83$ ms per image ($2.31\times$ overhead)** and RAM to 142.3 MB.
* MC-Dropout violates edge real-time deadlines (30 FPS = 33 ms budget) without beating a 0.51 ms post-hoc mapping in calibration transfer.

---

## 📑 5. Complete 23-Point Experimental Audit Scorecard

| ID | Objective / Requirement | Status | Empirical Evidence (Full 5,000 Image Run) |
| :---: | :--- | :---: | :--- |
| **0** | G1: Detection quality measured per condition | **Done** | YOLOv8n mAP50-95 drops $0.366 \to 0.005$; recall $0.502 \to 0.007$. |
| **1** | G2: Calibration improves detection ECE | **Met** | Verified in $4/4$ splits for YOLOv8n and $4/4$ splits for YOLO11n. |
| **2** | G2: Standard calibrators keep LaECE | **Not Met** | Degrades LaECE by $+0.0861$ in $4/4$ splits (**The Localization Paradox**). |
| **3** | G2: Loc-aware calibrators retain LaECE | **Met** | `loc_isotonic` retains LaECE within margin ($\Delta = -0.0055$, $4/4$ splits). |
| **4** | G3: Transfer to held-out families established | **Met (Equivalence)** | Outcome B proven: 95% CI for gap vs Oracle is $[-0.0050, +0.0060] \subset [\pm 0.01]$. |
| **5** | G3: Test is adequately powered | **Met** | Median CI half-width is $0.0053 \le 0.01$ significance margin ($2,000$ test images). |
| **6** | G3: Each held-out family shifts calibration | **Not Met** | Synthetic noise/brightness show weak calibration shift ($<0.01$); blur/weather shift strongly. |
| **7** | G3: Severity separated from family | **Done** | Evaluated independently in Split S5 (Fit on s1, s5; Test on held-out s3). |
| **8** | G3: Test family excluded from fit/val | **Done** | Disjoint data assertions verified in data loader; selection on validation family only. |
| **9** | G4: No suppression artefact (coverage drift) | **Resolved** | Fixed threshold drift was $0.056 > 0.05$; Conformal Risk Control bounds coverage to $1 - \alpha$. |
| **10**| G4: Latency and memory reported | **Done** | Platt requires $0.51$ ms/1k detections; MC-Dropout adds $2.31\times$ latency tax. |
| **11**| Error detection: MC beats raw AUROC | **Met** | MC reliability feature outperforms raw confidence in error AUROC across $4/4$ splits. |
| **12**| Correction justified by evidence | **Clarified** | Unjustified for classification ECE (Outcome B holds); strictly justified for localization LaECE. |
| **13**| Baselines: Raw, Temp, Platt, Iso, Hist, MC | **Done** | All 6 standard uncertainty baselines implemented and benchmarked. |
| **14**| Baseline: Detector-specific (Küppers et al.)| **Done** | Implemented box-aware multivariate logistic calibration (scale & aspect ratio). |
| **15**| Baseline: Published retrained methods | **Scoped Out** | Retraining violates zero-retraining edge efficiency objective. |
| **16**| Baseline: Conformal prediction | **Now Integrated**| Formulated Conformal Risk Control (CRC) and prediction set coverage module. |
| **17**| Evaluated on both YOLOv8n & YOLO11n | **Done** | Both models evaluated across all 19 conditions and 5 splits. |
| **18**| Repeated image-split seeds | **Done** | Evaluated across 3 independent random image partition seeds ($42, 43, 44$). |
| **19**| Second dataset / Natural distribution shift | **Prepared** | Evaluation harness prepared for external benchmarks (COUNTS / BDD100k-C). |
| **20**| Real corruptions vs. synthetic | **Limitation** | Documented as ImageNet-C synthetic approximations (OpenCV kernels). |
| **21**| Test set $\ge 2,000$ images | **Met** | Exactly $2,000$ test images per split ($40\%$ of $5,000$ COCO val2017). |
| **22**| Reproducibility record | **Done** | Run config, seeds, library versions, CSV tables, and master JSON archived. |

---

## 💻 6. Interactive Web Showcase & Demonstration Dashboard

The repository includes a standalone interactive dashboard designed for faculty presentations and thesis defense committees:
* **One-Page Proposal View:** Executive summary and research questions with KaTeX rendering.
* **Novelty Matrix View:** Searchable, filterable comparative matrix against 9 state-of-the-art papers.
* **Interactive Lab & Simulator:** Adjust models, corruptions (noise, blur, weather), and calibrators; view live corrupted detections and reliability curves.
* **Benchmark Scorecard & Visualizer:** Interactive Chart.js graphs of ECE transfer, the LaECE Paradox, mAP degradation, and edge latency.
* **Professor Presentation Mode:** An 8-slide guided presentation deck for committee walkthroughs.

To run the web app locally:
```bash
# Open directly in any browser:
open index.html
# Or serve via Python:
python -m http.server 8000
```
Or access online via GitHub Pages: [https://upasana-mukherjee-777.github.io/YOLO_Calibration/](https://upasana-mukherjee-777.github.io/YOLO_Calibration/)

---

## 🚀 7. Reproduction & Execution Guide

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/Upasana-Mukherjee-777/YOLO_Calibration.git
cd YOLO_Calibration
pip install -r requirements.txt
```

### Step 2: Set Up Dataset (Automated Utility)
The project evaluates on the COCO val2017 dataset. We provide an automated downloader utility:

* **Option A — Fast Verification (Zero-Wait, 50 Images):**
  Generates a synthetic quick-test subset in 2 seconds to verify the full pipeline:
  ```bash
  python download_dataset.py --quick
  ```

* **Option B — Official Full COCO val2017 (~1.2 GB):**
  Automatically downloads and unpacks official COCO images and annotations into `./data/coco`:
  ```bash
  python download_dataset.py --full
  ```

### Step 3: Run the Benchmark

#### Mode A: Standalone CLI Python Script
Run the automated benchmark directly from terminal:
```bash
# Fast test (50 images):
python run_calibration.py --data_dir ./data/coco --n_images 50

# Full benchmark run (5,000 images):
python run_calibration.py --data_dir ./data/coco --n_images 5000 --device 0
```

#### Mode B: Interactive Master Jupyter Notebook
Open the notebook in VS Code, JupyterLab, or upload to Google Colab / Kaggle:
* `YOLO_Calibration_Transfer.ipynb`
* All outputs, tables, and figures are pre-rendered so you can view the results immediately without re-running on GPU.

### Step 4: Launch Interactive Demonstration Dashboard
Open the interactive dashboard locally:
```bash
python -m http.server 8000
# Open http://localhost:8000 in your browser
```
Or view the live cloud version on GitHub Pages:  
👉 **[https://upasana-mukherjee-777.github.io/YOLO_Calibration/](https://upasana-mukherjee-777.github.io/YOLO_Calibration/)**

---

## 📚 8. Citation & Academic Reference

If you find this research proposal, empirical audit, or code helpful in your work, please cite:

```bibtex
@article{mukherjee2026yolocalibration,
  title={Reliable Calibration Transfer for Lightweight YOLO Detectors Under Unseen Image Corruptions},
  author={Mukherjee, Upasana},
  journal={arXiv preprint / IEEE Submission},
  year={2026},
  url={https://github.com/Upasana-Mukherjee-777/YOLO_Calibration}
}
```
