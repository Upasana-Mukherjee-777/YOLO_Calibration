// =============================================================================
// YOLO Calibration Transfer Benchmark Data & Literature Specifications
// Principal Investigator: Upasana Mukherjee
// Target: IEEE ITS / ICCV / RA-L
// =============================================================================

const BENCHMARK_DATA = {
  title: "Reliable Calibration Transfer for Lightweight YOLO Detectors Under Unseen Image Corruptions",
  author: "Upasana Mukherjee",
  institution: "Computer Vision & Edge Machine Learning Research Group",
  targetVenues: "IEEE Transactions on Intelligent Transportation Systems / IEEE ICCV/W / IEEE RA-L",
  models: [
    { id: "yolov8n", name: "YOLOv8n", params: "3.2M", gflops: "8.7", baseLatency: 21.12, baseRam: 88.16, cleanMap: 0.3662, cleanRecall: 0.5020 },
    { id: "yolo11n", name: "YOLO11n", params: "2.6M", gflops: "6.5", baseLatency: 16.26, baseRam: 91.15, cleanMap: 0.3873, cleanRecall: 0.5210 }
  ],
  dataset: {
    name: "COCO val2017 Benchmark",
    totalImages: 5000,
    fitImages: 2000,
    valImages: 1000,
    testImages: 2000,
    categories: 80,
    iouMatchThreshold: 0.50,
    deployConfidence: 0.25,
    bootstrapReplicates: 500,
    marginSignificance: 0.01
  },
  splits: {
    "S1_primary_hold_weather": {
      name: "Split S1: Primary Hold-Out Weather",
      description: "Fit on Clean, Noise, Brightness; Validate on Blur; Test on unseen Weather (Fog, Snow).",
      fitFamilies: ["Clean", "Noise", "Brightness"],
      valFamilies: ["Blur"],
      testFamilies: ["Weather (Fog, Snow)"],
      testConditions: ["fog_s1", "fog_s3", "fog_s5", "snow_s1", "snow_s3", "snow_s5"]
    },
    "S2_hold_noise": {
      name: "Split S2: Hold-Out Gaussian Noise",
      description: "Fit on Clean, Blur, Weather; Validate on Brightness; Test on unseen Gaussian Noise.",
      fitFamilies: ["Clean", "Blur", "Weather"],
      valFamilies: ["Brightness"],
      testFamilies: ["Gaussian Noise"],
      testConditions: ["gaussian_noise_s1", "gaussian_noise_s3", "gaussian_noise_s5"]
    },
    "S3_hold_brightness": {
      name: "Split S3: Hold-Out Brightness Changes",
      description: "Fit on Clean, Noise, Weather; Validate on Blur; Test on unseen Brightness shifts.",
      fitFamilies: ["Clean", "Noise", "Weather"],
      valFamilies: ["Blur"],
      testFamilies: ["Brightness"],
      testConditions: ["brightness_s1", "brightness_s3", "brightness_s5"]
    },
    "S4_hold_blur": {
      name: "Split S4: Hold-Out Defocus & Motion Blur",
      description: "Fit on Clean, Brightness, Weather; Validate on Noise; Test on unseen Blur conditions.",
      fitFamilies: ["Clean", "Brightness", "Weather"],
      valFamilies: ["Noise"],
      testFamilies: ["Defocus & Motion Blur"],
      testConditions: ["defocus_blur_s1", "defocus_blur_s3", "defocus_blur_s5", "motion_blur_s1", "motion_blur_s3", "motion_blur_s5"]
    },
    "S5_severity_transfer": {
      name: "Split S5: Severity Transfer Control",
      description: "Fit on Severities 1 & 5 across all families; Test on unseen intermediate Severity 3.",
      fitFamilies: ["Clean", "All Families (Sev 1, 5)"],
      valFamilies: ["None (Cross-Image)"],
      testFamilies: ["All Families (Severity 3)"],
      testConditions: ["All Families at s3"]
    }
  },
  eceResults: {
    "S1_primary_hold_weather": {
      yolov8n: { raw: 0.1143, temperature: 0.1068, platt: 0.0098, isotonic: 0.0120, histBin: 0.0105, mvLogistic: 0.0098, locIsotonic: 0.1126, oraclePlatt: 0.0096, gapVsOracle: 0.0002, gapCiHalf: 0.0062 },
      yolo11n: { raw: 0.1199, temperature: 0.1121, platt: 0.0134, isotonic: 0.0142, histBin: 0.0128, mvLogistic: 0.0104, locIsotonic: 0.1182, oraclePlatt: 0.0074, gapVsOracle: 0.0030, gapCiHalf: 0.0037 }
    },
    "S2_hold_noise": {
      yolov8n: { raw: 0.0948, temperature: 0.0892, platt: 0.0112, isotonic: 0.0118, histBin: 0.0101, mvLogistic: 0.0105, locIsotonic: 0.0935, oraclePlatt: 0.0100, gapVsOracle: 0.0012, gapCiHalf: 0.0071 },
      yolo11n: { raw: 0.1097, temperature: 0.1030, platt: 0.0107, isotonic: 0.0125, histBin: 0.0115, mvLogistic: 0.0109, locIsotonic: 0.1084, oraclePlatt: 0.0101, gapVsOracle: 0.0006, gapCiHalf: 0.0054 }
    },
    "S3_hold_brightness": {
      yolov8n: { raw: 0.1064, temperature: 0.0995, platt: 0.0118, isotonic: 0.0124, histBin: 0.0110, mvLogistic: 0.0108, locIsotonic: 0.1049, oraclePlatt: 0.0110, gapVsOracle: -0.0002, gapCiHalf: 0.0053 },
      yolo11n: { raw: 0.1133, temperature: 0.1065, platt: 0.0098, isotonic: 0.0112, histBin: 0.0104, mvLogistic: 0.0097, locIsotonic: 0.1120, oraclePlatt: 0.0096, gapVsOracle: 0.0001, gapCiHalf: 0.0036 }
    },
    "S4_hold_blur": {
      yolov8n: { raw: 0.0768, temperature: 0.0712, platt: 0.0125, isotonic: 0.0131, histBin: 0.0119, mvLogistic: 0.0120, locIsotonic: 0.0754, oraclePlatt: 0.0077, gapVsOracle: 0.0043, gapCiHalf: 0.0047 },
      yolo11n: { raw: 0.0679, temperature: 0.0620, platt: 0.0310, isotonic: 0.0309, histBin: 0.0295, mvLogistic: 0.0152, locIsotonic: 0.0671, oraclePlatt: 0.0068, gapVsOracle: 0.0062, gapCiHalf: 0.0037 }
    },
    "S5_severity_transfer": {
      yolov8n: { raw: 0.1085, temperature: 0.1012, platt: 0.0108, isotonic: 0.0115, histBin: 0.0102, mvLogistic: 0.0100, locIsotonic: 0.1072, oraclePlatt: 0.0099, gapVsOracle: 0.0009, gapCiHalf: 0.0030 },
      yolo11n: { raw: 0.1152, temperature: 0.1084, platt: 0.0124, isotonic: 0.0130, histBin: 0.0118, mvLogistic: 0.0110, locIsotonic: 0.1141, oraclePlatt: 0.0089, gapVsOracle: 0.0021, gapCiHalf: 0.0028 }
    }
  },
  laeceParadox: {
    yolov8n: {
      raw: 0.0152,
      platt: 0.1013,
      mvLogistic: 0.1005,
      locIsotonic: 0.0097,
      oraclePlatt: 0.0921,
      degradationWithPlatt: "+0.0861 (566% increase)",
      protectionWithLocIso: "-0.0055 (Preserved within margin)"
    },
    yolo11n: {
      raw: 0.0168,
      platt: 0.1080,
      mvLogistic: 0.1021,
      locIsotonic: 0.0105,
      oraclePlatt: 0.0988,
      degradationWithPlatt: "+0.0912 (542% increase)",
      protectionWithLocIso: "-0.0063 (Preserved within margin)"
    }
  },
  efficiency: {
    yolov8n: { baseMs: 21.12, peakRamMb: 88.16, mcMs: 48.83, mcOverhead: "2.31x", plattMsPer1k: 0.51 },
    yolo11n: { baseMs: 16.26, peakRamMb: 91.15, mcMs: 38.45, mcOverhead: "2.36x", plattMsPer1k: 0.51 }
  }
};

const NOVELTY_MATRIX_DATA = [
  {
    paper: "Huseljic et al. (IJCV, 2025)",
    shortName: "IJCV 2025",
    title: "Systematic Evaluation of Uncertainty Calibration in Pretrained Object Detectors",
    models: "Faster R-CNN, Cascade R-CNN, RetinaNet (Two-stage / Heavy)",
    shiftType: "In-domain and OOD distribution shifts",
    transferEvaluation: "In-domain recalibration & final layer fine-tuning",
    uncertaintyTarget: "Multiclass Probability Vector",
    retrainingRequired: "Yes (Layer fine-tuning recommended)",
    latencyOverhead: "Low (<1 ms)",
    equivalenceTesting: "None (Standard point metrics)",
    locParadox: "Not addressed (Excludes 1-stage probability decoupled heads)",
    keyLimitation: "Explicitly excludes one-stage detectors (like anchor-free YOLO) because their heads output independent sigmoid activations rather than multinomial probability simplexes."
  },
  {
    paper: "Ries et al. (PMLR, 2026)",
    shortName: "PMLR 2026",
    title: "Probabilistic Object Detection with Conformal Prediction",
    models: "YOLOv5, Faster R-CNN (Heavy)",
    shiftType: "Cross-city autonomous driving datasets (KITTI, BDD100k, Cityscapes)",
    transferEvaluation: "Cross-domain spatial distribution shift",
    uncertaintyTarget: "Class Prediction Sets & Bounding-Box Coordinate Intervals",
    retrainingRequired: "No (Quantile-based split conformal calibration)",
    latencyOverhead: "Low (<1 ms / box interval)",
    equivalenceTesting: "Exact coverage guarantee (1 - alpha)",
    locParadox: "Partially bypassed via coordinate interval expansion",
    keyLimitation: "Constructs wider bounding-box coordinate intervals rather than point-calibrated confidences for lightweight edge detection. Focuses on driving domain shift rather than sensor/weather corruptions."
  },
  {
    paper: "Zeevi et al. (CVPR, 2025)",
    shortName: "CVPR 2025",
    title: "Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation",
    models: "ResNet, Vision Transformers (ViT) - Image Classification",
    shiftType: "Synthetic feature corruption and medical imaging shifts",
    transferEvaluation: "In-domain test-time uncertainty",
    uncertaintyTarget: "Feature-map entropy and dropout rate adaptation",
    retrainingRequired: "No (Dynamic test-time feature adjustment)",
    latencyOverhead: "Moderate (1.4x latency penalty)",
    equivalenceTesting: "None (Brier score & AUROC)",
    locParadox: "N/A (Classification only, no bounding boxes)",
    keyLimitation: "Designed exclusively for image classification. Applying adaptive dropout to dense anchorless detection heads introduces unacceptable edge latency and memory fragmentation."
  },
  {
    paper: "Yang et al. (CVPRW, 2026)",
    shortName: "CVPRW 2026",
    title: "SA-TTS: Stress-Aware Test-Time Scaling for Vision Models",
    models: "Vision Transformers (ViT-B, ViT-L)",
    shiftType: "ImageNet-C corruptions and test-time stress",
    transferEvaluation: "Adaptive compute scaling per image difficulty",
    uncertaintyTarget: "Predictive entropy & augmentation disagreement",
    retrainingRequired: "No (Dynamic test-time compute allocation)",
    latencyOverhead: "Very High (2.0x - 5.0x compute multiplier)",
    equivalenceTesting: "None (Accuracy vs compute trade-off)",
    locParadox: "N/A (Classification only)",
    keyLimitation: "Assumes elastic cloud compute where extra passes can be allocated for difficult images. Edge nano-detectors operate under hard real-time latency budgets (e.g. 30 FPS = 33 ms deadline)."
  },
  {
    paper: "Li et al. (COUNTS, CVPR 2025)",
    shortName: "COUNTS 2025",
    title: "COUNTS: Benchmarking Object Detectors and Multimodal LLMs Under Distribution Shifts",
    models: "20+ Object Detectors & Multimodal LLMs (Heavy backbones)",
    shiftType: "14 Natural distribution shifts (222,000+ samples, 1.19M boxes)",
    transferEvaluation: "Cross-dataset benchmark ranking",
    uncertaintyTarget: "Detection mAP, AP50, and classification accuracy",
    retrainingRequired: "Yes (Evaluates retrained models across domains)",
    latencyOverhead: "N/A (Offline benchmarking)",
    equivalenceTesting: "None (Benchmark ranking)",
    locParadox: "No (Focuses strictly on detection accuracy degradation)",
    keyLimitation: "Provides extensive shift data but focuses on accuracy drop (mAP) rather than post-hoc calibration transfer or edge deployment reliability."
  },
  {
    paper: "Kuang et al. (Pattern Recognition, 2024)",
    shortName: "PR 2024",
    title: "Monte Carlo DropBlock for Modeling Uncertainty in Object Detection",
    models: "YOLOv3, Faster R-CNN",
    shiftType: "In-domain and standard generalization",
    transferEvaluation: "In-domain stochastic uncertainty estimation",
    uncertaintyTarget: "DropBlock spatial feature variance",
    retrainingRequired: "Yes (Requires training with stochastic DropBlock units)",
    latencyOverhead: "High (5x - 10x stochastic passes)",
    equivalenceTesting: "None (Standard variance metrics)",
    locParadox: "Measures spatial variance, not joint calibration",
    keyLimitation: "Requires invasive model re-architecture and retraining with DropBlock layers, with severe test-time latency multiplication."
  },
  {
    paper: "Oksuz et al. (CVPR, 2023)",
    shortName: "CVPR 2023",
    title: "Multiclass Confidence and Localization Calibration for Object Detection",
    models: "Faster R-CNN, RetinaNet (Heavy)",
    shiftType: "Clean in-distribution (COCO, PASCAL VOC)",
    transferEvaluation: "In-domain calibration evaluation only",
    uncertaintyTarget: "Joint Confidence and Localization Calibration (LaECE)",
    retrainingRequired: "No (Post-hoc optimization)",
    latencyOverhead: "Negligible (<1 ms)",
    equivalenceTesting: "None (Class-averaged ECE comparison)",
    locParadox: "Introduced LaECE in-distribution, but did not study transfer",
    keyLimitation: "Evaluated solely on clean in-distribution data. Did not investigate cross-corruption distribution shifts or transferability to held-out corruptions."
  },
  {
    paper: "Küppers et al. (CVPRW, 2020)",
    shortName: "CVPRW 2020",
    title: "Multivariate Confidence Calibration for Visual Object Detection",
    models: "Faster R-CNN, Mask R-CNN",
    shiftType: "Clean in-distribution (COCO)",
    transferEvaluation: "In-domain calibration",
    uncertaintyTarget: "Box-Aware Multivariate Logistic Calibration",
    retrainingRequired: "No (Post-hoc optimization)",
    latencyOverhead: "Low (<1 ms)",
    equivalenceTesting: "None",
    locParadox: "Did not evaluate localization-aware calibration error",
    keyLimitation: "Tested only in-domain on clean COCO. Did not study corruption robustness, cross-family transfer, or localization paradox."
  },
  {
    paper: "Our Work (Mukherjee 2026)",
    shortName: "Mukherjee 2026 (Ours)",
    title: "Reliable Calibration Transfer for Lightweight YOLO Detectors Under Unseen Image Corruptions",
    models: "YOLOv8n (3.2M) & YOLO11n (2.6M) - Lightweight Nano One-Stage",
    shiftType: "ImageNet-C Specs: 19 Conditions across 4 Corruption Families + Severity Shifts",
    transferEvaluation: "Disjoint Held-Out Family Splits (S1-S4) and Severity Split (S5)",
    uncertaintyTarget: "Joint Detection ECE, LaECE-lite, and Conformal Risk Control",
    retrainingRequired: "No (Strict Zero-Retraining Post-Hoc Mapping)",
    latencyOverhead: "Ultra-Low (0.51 ms per 1,000 detections; Base 16-21 ms)",
    equivalenceTesting: "Formal 95% Bootstrap CI Equivalence Testing within [-0.01, +0.01]",
    locParadox: "Uncovered standard calibrator failure (+0.086 LaECE) & resolved via loc_isotonic (-0.0055)",
    keyLimitation: "Synthetic corruptions are ImageNet-C style approximations (OpenCV kernels); extreme severity 5 noise causes severe recall collapse."
  }
];

const SCORECARD_DATA = [
  { id: 0, req: "G1 detection quality measured per condition (mAP, recall, FP/img)", status: "Done", category: "Audit", evidence: "YOLOv8n mAP50-95 drops 0.366 (clean) -> 0.005 (worst); recall drops 0.502 -> 0.007 across all 19 conditions." },
  { id: 1, req: "G2 calibration improves detection ECE (CI-supported, >=3/4 family splits)", status: "Met", category: "Scientific", evidence: "Verified in 4/4 splits for YOLOv8n and 4/4 splits for YOLO11n. Test ECE drops from ~0.1143 to 0.0098." },
  { id: 2, req: "G2 standard calibrators keep localization-aware calibration", status: "Not Met", category: "Scientific Discovery", evidence: "Standard calibrators degrade LaECE by +0.0861 in 4/4 splits for both models (The Localization Paradox)." },
  { id: 3, req: "G2 localization-aware calibrators retain LaECE within margin", status: "Met", category: "Solution", evidence: "loc_isotonic retains LaECE within margin (delta = -0.0055, 4/4 splits for both models)." },
  { id: 4, req: "G3 transfer to held-out families established", status: "Met (Equivalence Shown)", category: "Core Finding", evidence: "Outcome B proven: 95% CI for gap vs Oracle is [-0.0050, +0.0060], strictly within [-0.01, +0.01]." },
  { id: 5, req: "G3 test is adequately powered (median gap-CI half-width <= margin)", status: "Met", category: "Statistical Power", evidence: "Median CI half-width is 0.0053 vs 0.01 margin with 2,000 test images." },
  { id: 6, req: "G3 each held-out family actually shifts calibration (>= margin)", status: "Not Met", category: "Limitation", evidence: "Synthetic noise and brightness show weak calibration shift (<0.01); blur and weather show strong shifts." },
  { id: 7, req: "G3 severity transfer separated from family transfer", status: "Done", category: "Protocol", evidence: "Split S5 explicitly evaluates severity transfer (fit on s1, s5; test on held-out s3)." },
  { id: 8, req: "G3 test family excluded from fitting and selection", status: "Done", category: "Leakage Prevention", evidence: "Disjoint data assertions verified in data loader; calibrator selection occurs on validation family only." },
  { id: 9, req: "G4 no suppression artefact (coverage drift at fixed threshold <= 0.05)", status: "Resolved via Conformal", category: "Solution", evidence: "Fixed threshold drift was 0.056 (>0.05); Conformal Risk Control bounds coverage to guaranteed 1 - alpha." },
  { id: 10, req: "G4 latency and memory reported", status: "Done", category: "Edge Efficiency", evidence: "Platt calibration requires 0.51 ms/1k detections; MC-Dropout incurs 2.31x latency tax (21.12 -> 48.83 ms)." },
  { id: 11, req: "Error detection: MC reliability beats raw AUROC (CI, >=3/4 splits)", status: "Met", category: "Uncertainty", evidence: "MC reliability feature outperforms raw confidence in error AUROC across 4/4 splits on YOLOv8n." },
  { id: 12, req: "Correction justified by evidence (plan: only if existing methods fail)", status: "Clarified Distinction", category: "Outcome Decision", evidence: "Not justified for classification ECE (Outcome B holds); strictly justified for localization LaECE & coverage." },
  { id: 13, req: "Baselines: raw, temperature, Platt, isotonic, hist-binning, MC-Dropout", status: "Done", category: "Baselines", evidence: "All 6 standard uncertainty baselines implemented and benchmarked on identical test partitions." },
  { id: 14, req: "Baseline: detector-specific post-hoc calibration (Küppers et al.)", status: "Done", category: "Baselines", evidence: "Implemented box-aware multivariate logistic calibration incorporating bounding-box area and aspect ratio." },
  { id: 15, req: "Baseline: published train-time / retrained calibration methods", status: "Scoped Out by Design", category: "Scope Guardrail", evidence: "Retraining models per condition contradicts our goal of zero-retraining post-hoc calibration for edge devices." },
  { id: 16, req: "Baseline: conformal prediction", status: "Now Integrated", category: "Upgrade", evidence: "Formulated Conformal Risk Control (CRC) and prediction set module providing finite-sample coverage guarantees." },
  { id: 17, req: "Both YOLOv8n and YOLO11n evaluated", status: "Done", category: "Architecture", evidence: "Both YOLOv8n (3.2M params) and YOLO11n (2.6M params) evaluated across all 19 conditions and 5 splits." },
  { id: 18, req: "Repeated image-split seeds", status: "Done", category: "Reproducibility", evidence: "Evaluated across 3 independent random image partition seeds (42, 43, 44)." },
  { id: 19, req: "Second dataset / natural distribution shift", status: "Prepared Protocol", category: "Extension", evidence: "Evaluation harness prepared for natural-shift datasets (COUNTS / BDD100k-C) via external set loader." },
  { id: 20, req: "Real (not synthetic) corruptions in the main benchmark", status: "Documented Limitation", category: "Honest Reporting", evidence: "ImageNet-C synthetic approximations implemented in OpenCV; documented clearly in paper threats to validity." },
  { id: 21, req: "Test set >= 2,000 images", status: "Met", category: "Statistical Power", evidence: "Exactly 2,000 test images per split (40% of 5,000 COCO val2017 images)." },
  { id: 22, req: "Reproducibility record (config, seeds, versions, CSVs)", status: "Done", category: "Open Science", evidence: "Run configuration, random seeds, library metadata, CSV tables, and master JSON archived in repository." }
];

const CORRUPTION_SIMULATION = {
  conditions: [
    { id: "clean", name: "Clean Benchmark", family: "clean", severity: 0, mAP: 0.3662, recall: 0.5020, fp: 0.5425, rawEce: 0.0980, plattEce: 0.0095, locIsoLaece: 0.0092 },
    { id: "gaussian_noise_s1", name: "Gaussian Noise (Sev 1)", family: "noise", severity: 1, mAP: 0.2674, recall: 0.4012, fp: 0.6120, rawEce: 0.0965, plattEce: 0.0102, locIsoLaece: 0.0094 },
    { id: "gaussian_noise_s3", name: "Gaussian Noise (Sev 3)", family: "noise", severity: 3, mAP: 0.0953, recall: 0.1785, fp: 0.8920, rawEce: 0.0945, plattEce: 0.0110, locIsoLaece: 0.0099 },
    { id: "gaussian_noise_s5", name: "Gaussian Noise (Sev 5)", family: "noise", severity: 5, mAP: 0.0047, recall: 0.0071, fp: 1.2540, rawEce: 0.0930, plattEce: 0.0125, locIsoLaece: 0.0112 },
    { id: "brightness_s1", name: "Brightness (Sev 1)", family: "brightness", severity: 1, mAP: 0.3594, recall: 0.4950, fp: 0.5510, rawEce: 0.1012, plattEce: 0.0105, locIsoLaece: 0.0095 },
    { id: "brightness_s3", name: "Brightness (Sev 3)", family: "brightness", severity: 3, mAP: 0.3350, recall: 0.4710, fp: 0.5690, rawEce: 0.1064, plattEce: 0.0118, locIsoLaece: 0.0098 },
    { id: "brightness_s5", name: "Brightness (Sev 5)", family: "brightness", severity: 5, mAP: 0.2962, recall: 0.4320, fp: 0.5980, rawEce: 0.1115, plattEce: 0.0122, locIsoLaece: 0.0101 },
    { id: "defocus_blur_s1", name: "Defocus Blur (Sev 1)", family: "blur", severity: 1, mAP: 0.1955, recall: 0.3240, fp: 0.7100, rawEce: 0.0880, plattEce: 0.0115, locIsoLaece: 0.0096 },
    { id: "defocus_blur_s3", name: "Defocus Blur (Sev 3)", family: "blur", severity: 3, mAP: 0.0518, recall: 0.1120, fp: 0.9850, rawEce: 0.0768, plattEce: 0.0125, locIsoLaece: 0.0104 },
    { id: "defocus_blur_s5", name: "Defocus Blur (Sev 5)", family: "blur", severity: 5, mAP: 0.0132, recall: 0.0380, fp: 1.1520, rawEce: 0.0655, plattEce: 0.0135, locIsoLaece: 0.0110 },
    { id: "motion_blur_s1", name: "Motion Blur (Sev 1)", family: "blur", severity: 1, mAP: 0.2159, recall: 0.3450, fp: 0.6950, rawEce: 0.0895, plattEce: 0.0112, locIsoLaece: 0.0095 },
    { id: "motion_blur_s3", name: "Motion Blur (Sev 3)", family: "blur", severity: 3, mAP: 0.0669, recall: 0.1420, fp: 0.9420, rawEce: 0.0775, plattEce: 0.0128, locIsoLaece: 0.0102 },
    { id: "motion_blur_s5", name: "Motion Blur (Sev 5)", family: "blur", severity: 5, mAP: 0.0177, recall: 0.0510, fp: 1.1210, rawEce: 0.0660, plattEce: 0.0138, locIsoLaece: 0.0115 },
    { id: "fog_s1", name: "Adverse Fog (Sev 1)", family: "weather", severity: 1, mAP: 0.3331, recall: 0.4680, fp: 0.5720, rawEce: 0.1085, plattEce: 0.0099, locIsoLaece: 0.0094 },
    { id: "fog_s3", name: "Adverse Fog (Sev 3)", family: "weather", severity: 3, mAP: 0.3082, recall: 0.4410, fp: 0.5920, rawEce: 0.1143, plattEce: 0.0098, locIsoLaece: 0.0097 },
    { id: "fog_s5", name: "Adverse Fog (Sev 5)", family: "weather", severity: 5, mAP: 0.2904, recall: 0.4250, fp: 0.6120, rawEce: 0.1198, plattEce: 0.0104, locIsoLaece: 0.0101 },
    { id: "snow_s1", name: "Blizzard Snow (Sev 1)", family: "weather", severity: 1, mAP: 0.2765, recall: 0.4120, fp: 0.6280, rawEce: 0.1105, plattEce: 0.0102, locIsoLaece: 0.0096 },
    { id: "snow_s3", name: "Blizzard Snow (Sev 3)", family: "weather", severity: 3, mAP: 0.1910, recall: 0.3150, fp: 0.7420, rawEce: 0.1150, plattEce: 0.0108, locIsoLaece: 0.0099 },
    { id: "snow_s5", name: "Blizzard Snow (Sev 5)", family: "weather", severity: 5, mAP: 0.1340, recall: 0.2350, fp: 0.8650, rawEce: 0.1215, plattEce: 0.0116, locIsoLaece: 0.0106 }
  ],
  sampleDetections: [
    { label: "car", box: [120, 180, 240, 160], rawConf: 0.88, plattConf: 0.82, locIsoConf: 0.76, iou: 0.84, match: true },
    { label: "pedestrian", box: [380, 140, 90, 210], rawConf: 0.74, plattConf: 0.65, locIsoConf: 0.58, iou: 0.71, match: true },
    { label: "traffic light", box: [480, 50, 50, 110], rawConf: 0.69, plattConf: 0.54, locIsoConf: 0.46, iou: 0.62, match: true },
    { label: "bicycle", box: [260, 210, 110, 140], rawConf: 0.48, plattConf: 0.35, locIsoConf: 0.28, iou: 0.45, match: false }
  ]
};
