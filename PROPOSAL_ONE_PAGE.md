# Research Proposal (Executive One-Page Plan)

## 1. Project Title
**Reliable Calibration Transfer for Lightweight YOLO Detectors Under Unseen Image Corruptions**

* **Research Area:** Computer Vision, Lightweight Object Detection, Uncertainty Quantification, Model Calibration, Distribution Shifts.
* **Target Venues:** IEEE Transactions on Intelligent Transportation Systems / IEEE International Conference on Computer Vision (ICCV/W) / IEEE Robotics and Automation Letters (RA-L).
* **Principal Investigator:** Upasana Mukherjee

---

## 2. Problem Statement & Research Question
Modern real-time computer vision systems rely heavily on lightweight single-stage detectors such as **YOLOv8n** and **YOLO11n** for edge inference. However, when deployed in dynamic open-world environments (e.g., autonomous driving, drone surveillance, robotic inspection), detectors frequently encounter distribution shifts caused by natural and environmental corruptions—including sensor noise, defocus blur, motion blur, brightness fluctuations, fog, and snow.

While detection accuracy (mAP, recall) degrades under corruptions, an even more insidious failure mode is **confidence miscalibration**: detectors often emit overconfident predictions on corrupted inputs, masking localization drift and false positive hallucinations. Retraining detectors for every deployment environment is computationally prohibitive. Therefore, post-hoc calibration fitted on a limited set of validation corruptions is the prevailing industry paradigm. However, it remains unverified whether post-hoc calibration transfers reliably across *entirely held-out, unseen corruption families* without sacrificing localization quality.

**Central Research Question:**
$$\text{Can post-hoc calibration fitted on known corruption conditions reliably transfer to previously unseen corruption families in lightweight YOLO detectors without retraining or degrading localization calibration?}$$

---

## 3. Four Core Research Hypotheses
1. **$H_1$ (Calibration Transferability):** Post-hoc confidence calibration (e.g., Platt scaling, Isotonic regression, multivariate logistic) fitted on known corruption families transfers to held-out corruption families, achieving calibration error equivalent to an in-distribution Oracle calibrator within a practical significance margin $\epsilon = 0.01$.
2. **$H_2$ (The Localization-Aware Paradox):** Standard calibrators optimized solely on binary detection correctness ($\mathbb{I}[\text{IoU} \ge 0.5 \land \hat{y} = y]$) severely inflate Localization-Aware Calibration Error (LaECE) by overconfidently scaling detections whose bounding boxes remain imperfect.
3. **$H_3$ (Localization-Aware & Conformal Correction):** Jointly optimizing calibrators on detection correctness scaled by intersection-over-union ($\text{correct} \times \text{IoU}$) resolves the LaECE paradox while Conformal Risk Control (CRC) guarantees user-specified coverage under unseen shifts.
4. **$H_4$ (Edge Efficiency Trade-Off):** Test-time stochastic sampling (Monte Carlo Dropout) incurs a prohibitive $>2.3\times$ latency penalty on edge hardware while offering negligible calibration transfer gains over ultra-lightweight ($<0.6$ ms/1k detections) post-hoc mappings.

---

## 4. Rigorous Experimental Protocol
```
                                ┌────────────────────────────────────────┐
                                │   COCO val2017 Benchmark (5,000 imgs)   │
                                └───────────────────┬────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴─────────────────────────────────┐
                 │                                                                    │
    ┌────────────┴─────────────┐                                         ┌────────────┴─────────────┐
    │     YOLOv8n (3.2M params) │                                         │    YOLO11n (2.6M params)  │
    └────────────┬─────────────┘                                         └────────────┬─────────────┘
                 │                                                                    │
                 └──────────────────────────────────┬─────────────────────────────────┘
                                                    │
                               ┌────────────────────┴───────────────────┐
                               │ 19 Visual Conditions (ImageNet-C Specs)│
                               │  - Clean                               │
                               │  - Gaussian Noise (s1, s3, s5)         │
                               │  - Brightness (s1, s3, s5)             │
                               │  - Defocus Blur (s1, s3, s5)           │
                               │  - Motion Blur (s1, s3, s5)            │
                               │  - Fog (s1, s3, s5)                    │
                               │  - Snow (s1, s3, s5)                   │
                               └────────────────────┬───────────────────┘
                                                    │
       ┌───────────────────────┬────────────────────┼───────────────────────┬───────────────────────┐
       ▼                       ▼                    ▼                       ▼                       ▼
 Split S1 (Weather)      Split S2 (Noise)    Split S3 (Brightness)   Split S4 (Blur)        Split S5 (Severity)
 Fit: Clean+Noise+Bright Fit: Clean+Blur+Wea Fit: Clean+Noise+Wea    Fit: Clean+Bright+Wea  Fit: Sev 1, 5 (All)
 Val: Blur               Val: Brightness     Val: Blur               Val: Noise             Val: Held-out imgs
 Test: Weather (Sev 1-5) Test: Noise (Sev1-5)Test: Brightness (1-5)  Test: Blur (Sev 1-5)   Test: Sev 3 (All)
```

* **Disjoint Data Partition:** 5,000 COCO images split 40% Fit (2,000), 20% Val (1,000), 40% Test (2,000). Calibrators never access test images or held-out test corruption families.
* **Evaluated Methods:**
  1. *Raw Uncalibrated YOLO*
  2. *Temperature Scaling* (Guo et al., 2017)
  3. *Platt Scaling* (Platt, 1999)
  4. *Isotonic Regression* (Zadrozny & Elkan, 2002)
  5. *Histogram Binning* (Zadrozny & Elkan, 2001)
  6. *Detector-Specific Box-Aware Multivariate Logistic* (after Küppers et al., 2020)
  7. *Localization-Aware Calibrator (`loc_isotonic`, `loc_platt`)* (adapting Oksuz et al., 2023)
  8. *Monte Carlo Dropout* ($T=6$, $p=0.10$, post-hoc hooked on YOLO backbone)
  9. *Conformal Risk Control (CRC)* (Angelopoulos et al., 2024; Ries et al., 2026)
  10. *In-Distribution Oracle* (calibrator fitted directly on test corruption family as ceiling)
* **Statistical Rigor:** Image-cluster bootstrap ($B=500$ replicates, 95% CIs) paired across methods. Equivalence margin defined as $\Delta_{\text{ECE}} \in [-0.01, +0.01]$. Multiplicity controls across 3 random image partition seeds.

---

## 5. Summary of Empirical Findings (The "Outcome B" Discovery)
1. **Calibration Transfers Across Unseen Families:** Across all family splits, Platt scaling and Multivariate Logistic calibration drop test ECE from $\sim 0.1143$ down to $0.0098$. The transfer gap against the Oracle calibrator is $\Delta = +0.0002$ with a 95% CI $[-0.0050, +0.0060]$. Because the CI falls strictly within $\pm 0.01$, **Equivalence is formally demonstrated (Outcome B)**.
2. **The Localization Paradox Uncovered:** Standard calibrators reduce detection ECE by $\sim 0.1045$ but degrade Localization-Aware Calibration Error (LaECE) by $+0.0861$ (raising it from $0.0152$ to $0.1013$). In contrast, proposed Localization-Aware Calibrators (`loc_isotonic`) maintain LaECE at $0.0097$ with no degradation ($\Delta = -0.0055$).
3. **Severe Shift Edge Breakdown:** Under severity 5 noise, mAP50-95 collapses to $0.0047$ with only 177 test detections, causing severe coverage drift (0.056). Conformal Risk Control successfully recovers bounded coverage.
4. **Latency Dominance:** Platt calibration requires $0.51$ ms per 1,000 detections. MC-Dropout adds a $2.31\times$ latency penalty ($21.12$ ms $\to 48.83$ ms) without significant calibration benefits over post-hoc scaling.

---

## 6. Project Timeline & Deliverables
* **Weeks 1–2:** Literature verification & 2025–2026 Novelty Matrix finalization.
* **Weeks 3–4:** Full COCO 5,000-image evaluation pipeline & multi-seed verification.
* **Weeks 5–6:** Conformal Risk Control integration and LaECE paradox theoretical formalization.
* **Weeks 7–8:** Interactive demonstration dashboard deployment & paper drafting for IEEE submission.
