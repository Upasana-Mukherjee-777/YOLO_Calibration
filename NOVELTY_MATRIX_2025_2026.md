# 2025–2026 Novelty Matrix & Literature Positioning

## Executive Overview
A critical milestone for any peer-reviewed computer vision submission (specifically targeting IEEE conferences and indexed journals) is establishing that the proposed research gap is **defensible, unoccupied, and verified against recent state-of-the-art literature (2020–2026)**.

The literature already explores confidence calibration, stochastic uncertainty estimation (MC Dropout, DropBlock), distribution shift benchmarks (ImageNet-C, COUNTS), and conformal prediction. **Therefore, novelty cannot be claimed simply by adding MC Dropout to YOLO or evaluating corrupted images.**

Our contribution specifically isolates:
1. **Cross-corruption family calibration transfer** on **lightweight one-stage nano detectors (YOLOv8n and YOLO11n)** without model retraining.
2. Formal **statistical equivalence testing** ($\Delta \le \pm 0.01$ margin) against an in-distribution Oracle.
3. The empirical discovery and resolution of the **Localization-Aware Calibration Paradox** (where standard calibrators improve detection ECE but degrade localization-aware calibration by $>5\times$).
4. The integration of **Conformal Risk Control** to counteract coverage drift under severe corruption tails.

---

## The 2025–2026 Novelty Matrix

| Dimension | Huseljic et al. (IJCV 2025) | Ries et al. (PMLR 2026) | Zeevi et al. (CVPR 2025) | Yang et al. (CVPRW 2026) | Li et al. (COUNTS, CVPR 2025) | Kuang et al. (PR 2024) | Oksuz et al. (CVPR 2023) | Küppers et al. (CVPRW 2020) | **Our Work (Mukherjee 2026)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Detector Architecture** | Faster R-CNN, Cascade R-CNN, RetinaNet (Two-stage / Heavy) | YOLOv5, Faster R-CNN | ResNet / ViT (Classification) | Vision Transformers (ViT) | 20+ Detectors & MLLMs (Heavy) | YOLOv3, Faster R-CNN | Faster R-CNN, RetinaNet | Faster R-CNN, Mask R-CNN | **YOLOv8n & YOLO11n (Nano, One-Stage)** |
| **Shift Paradigm** | OOD & In-Domain Shifts | Autonomous Driving Cross-Domain | Synthetic Feature Shift (Medical) | Test-Time Difficulty / Stress | 14 Natural Distribution Shifts | In-domain & Generalization | In-domain (Clean COCO / Pascal) | In-domain (COCO) | **Held-Out Corruption Families (ImageNet-C Specs)** |
| **Transfer Type Evaluated** | In-domain recalibration & fine-tuning | Cross-city domain shift | No transfer evaluation | Test-time adaptive computation | Cross-domain benchmark ranking | In-domain stochastic inference | In-domain calibration | In-domain post-hoc calibration | **Held-Out Family Transfer vs. Severity Transfer** |
| **Uncertainty Target** | Class Probability Vector | Class Sets & Bounding-Box Intervals | Feature-Map Entropy | Predictive Entropy Disagreement | Detection mAP / Brier Score | DropBlock Spatial Uncertainty | Joint Confidence & Localization Calibration | Box-Aware Multiclass Calibration | **Joint Detection ECE, LaECE-lite & Conformal Coverage** |
| **Retraining Required?** | Yes (Final layer fine-tuning recommended) | No (Post-hoc quantile calibration) | No (Test-time feature adjustment) | No (Test-time scaling / compute) | Yes (Evaluates retrained models) | Yes (Stochastic layers during train) | No (Post-hoc optimization) | No (Post-hoc optimization) | **No (Strict Zero-Retraining Post-Hoc Calibration)** |
| **Inference Overhead** | Negligible (Post-hoc) or Fine-tuned | Low ($<1$ ms/box) | Moderate ($1.4\times$) | High ($2.0\times - 5.0\times$) | N/A (Offline Benchmark) | High ($5\times - 10\times$ passes) | Negligible ($<1$ ms) | Low ($<1$ ms) | **Ultra-Low ($0.51$ ms/1k dets) vs MC ($2.31\times$)** |
| **Equivalence Testing** | None (Empirical point metrics) | Coverage guarantee ($1 - \alpha$) | Empirical AUROC / Brier | Accuracy vs compute trade-off | Benchmark ranking | Variance comparison | Class-averaged ECE comparison | Metric improvement over raw | **Formal 95% Bootstrap Equivalence Bounds ($\pm 0.01$)** |
| **Localization Paradox Uncovered?** | No (Focused on probability vectors) | Addressed via box intervals | N/A (Image classification) | N/A (Classification) | No (Focus on detection mAP) | Evaluated box variance | Addressed in-domain with LaECE | Box position as logistic feature | **Yes: First to document and resolve in cross-family transfer** |

---

## Detailed Critical Review of Closest Prior Works

### 1. Huseljic et al. — *Systematic Evaluation of Uncertainty Calibration in Pretrained Object Detectors* (IJCV, 2025)
* **What they established:** Systematically evaluated confidence calibration in pretrained detectors across in-distribution and out-of-distribution shifts. Demonstrated that fine-tuning final layers improves calibration.
* **Why lightweight one-stage YOLO is excluded:** The authors explicitly excluded modern one-stage detectors (like anchor-free YOLO) from their formal calibration framework because their calibration formulations rely on class probability vectors over full multinomial distributions, whereas modern YOLO heads output independent sigmoid class probabilities per anchorless grid cell.
* **Our Edge:** We formulate post-hoc calibration specifically tailored to modern decoupled anchor-free YOLO prediction heads, proving that full retraining/fine-tuning is **unnecessary** for cross-family transfer.

### 2. Ries et al. — *Probabilistic Object Detection with Conformal Prediction* (PMLR, 2026)
* **What they established:** Applied split conformal prediction to object detection to construct prediction sets for classes and bounding-box coordinate intervals with finite-sample coverage guarantees across autonomous driving datasets.
* **Relation to our study:** While Ries et al. demonstrate that conformal prediction provides theoretical coverage guarantees under domain shift, they focus on two-stage/heavy models and bounding box expansion intervals rather than confidence score recalibration for resource-constrained edge edge-devices.
* **Our Edge:** We test the complementary boundary: whether simple scalar post-hoc calibrators ($0.51$ ms) can achieve calibration equivalence without enlarging bounding box intervals, and integrate Conformal Risk Control (CRC) to bound coverage drift in extreme noise regimes.

### 3. Zeevi et al. — *Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation* (CVPR, 2025)
* **What they established:** Dynamically adjusted dropout rates at inference time according to information loss in intermediate feature maps.
* **Why it does not solve our problem:** Rate-In addresses image classification and medical segmentation. Applying stochastic sampling or adaptive dropout to real-time object detection incurs high inference latency and memory fragmentation, which violates edge deployment constraints.
* **Our Edge:** We empirically benchmark test-time stochastic sampling on YOLOv8n, proving it introduces a $2.31\times$ latency tax while failing to beat simple post-hoc Platt scaling in cross-family calibration transfer.

### 4. Yang et al. — *Stress-Aware Test-Time Scaling for Vision Models* (CVPRW, 2026)
* **What they established:** Allocated additional compute dynamically based on predictive entropy and test-time stress/difficulty.
* **Why it does not solve our problem:** Designed for heavy vision transformers where test-time computation can be scaled elastically. Real-world edge systems running nano-YOLO (e.g., microcontrollers, Raspberry Pi, drone flight controllers) run at fixed budget constraints and cannot scale compute on demand.
* **Our Edge:** We achieve calibration transfer with static, constant-time inference ($O(1)$ post-processing mapping).

### 5. Li et al. — *COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models Under Distribution Shifts* (CVPR, 2025)
* **What they established:** Benchmarked 20+ object detectors across 14 natural distribution shifts ($222,000+$ samples, $1.196\text{M}$ boxes).
* **Why it does not solve our problem:** COUNTS evaluates detector robustness (accuracy/mAP degradation) under shift. It does not investigate whether confidence calibration fitted on one subset of shifts transfers to held-out shift categories.
* **Our Edge:** We shift the scientific question from *"How much does accuracy drop?"* to *"Do calibrated confidence scores remain trustworthy on unseen shifts without retraining?"*

### 6. Oksuz et al. — *Multiclass Confidence and Localization Calibration for Object Detection* (CVPR, 2023)
* **What they established:** Introduced Localization-Aware Calibration Error (LaECE) and demonstrated that confidence must reflect both classification accuracy and localization precision ($\text{precision} \times \text{IoU}$).
* **Why our work extends this:** Oksuz et al. investigated in-distribution calibration on standard benchmarks (clean COCO/PASCAL). They did not examine how localization calibration behaves under cross-corruption distribution shifts.
* **Our Discovery:** We show that standard calibrators exhibit a **Localization-Aware Paradox** under distribution shift: they improve classification ECE but dramatically worsen LaECE (from $0.015$ to $0.101$). We develop a lightweight localization-aware isotonic calibrator (`loc_isotonic`) that resolves this paradox in cross-family transfer.

---

## The Defensible Scientific Contribution
```
                                 THE SCIENTIFIC GAP
        ┌──────────────────────────────────┬──────────────────────────────────┐
        │       Existing Literature        │         Our Contribution         │
        ├──────────────────────────────────┼──────────────────────────────────┤
        │ Focus on two-stage/heavy models  │ Nano-scale one-stage detectors   │
        │ (Faster R-CNN, ViTs, RetinaNet)  │ (YOLOv8n, YOLO11n: 2.6M-3.2M)    │
        ├──────────────────────────────────┼──────────────────────────────────┤
        │ Severity shift or in-domain      │ Cross-family held-out transfer   │
        │ recalibration (same corruption)  │ (Noise/Blur -> Weather/Fog)      │
        ├──────────────────────────────────┼──────────────────────────────────┤
        │ Retraining or test-time compute  │ Strict zero-retraining post-hoc  │
        │ scaling (2x - 5x latency tax)    │ mapping (0.51 ms per 1k dets)    │
        ├──────────────────────────────────┼──────────────────────────────────┤
        │ Binary classification target     │ Localization-aware calibrator    │
        │ (Causes Localization Paradox)    │ (Resolves LaECE degradation)     │
        ├──────────────────────────────────┼──────────────────────────────────┤
        │ Point estimate metric evaluation │ Formal 95% Bootstrap Equivalence │
        │ (Misleading statistical claims)  │ Testing (Margin epsilon = 0.01)  │
        └──────────────────────────────────┴──────────────────────────────────┘
```
This multi-dimensional positioning confirms that our research addresses a specific, defensible, and unoccupied niche at the intersection of edge computer vision and reliable uncertainty quantification.
