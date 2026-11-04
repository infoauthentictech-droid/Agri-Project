# Groundnut AI

An agricultural AI prototype for **groundnut (peanut) crop monitoring**: capture
or upload a field photo, run YOLO object detection + patch classification on a
FastAPI backend, view an XAI explanation, and get an (honestly-labelled)
herbicide recommendation.

Frontend: Flutter (Android / Web).
Backend: FastAPI + Ultralytics YOLO + PyTorch classifiers.
ML: `ml/` pipeline (data prep, YOLO training, segmentation, classifier zoo, XAI).

## Algorithms Implemented

This project implements the following state-of-the-art algorithms for agricultural AI:

### Data Augmentation
- **Mosaic & Mixup Augmentation**: Forces the model to learn overlapping leaf edges, occlusion, and varied light exposures.

### Synthesis
- **CycleGAN**: Converts normal crop images to weed-infested scenarios without needing paired training data.

### Preprocessing
- **Gaussian Filtering + ExG Thresholding**: Instantly scrubs sensor noise and isolates vegetation from bare dirt.

### Semantic & Instance Segmentation
- **YOLOv8-Seg / YOLO11-Seg**: The absolute best option combining both semantic and instance segmentation into a single, high-speed, pixel-precise mask engine.

### Indexing
- **NDVI (Normalized Difference Vegetation Index)**: For multispectral imagery
- **ExG (Excess Green)**: For standard RGB cameras

### Classification
- **Single module**: CNN, ResNet50, VGG16, MobileNet, EfficientNet
- **Hybrid module**: ResNet50+MobileNet, ResNet50+EfficientNet, ResNet50+EfficientNet+MobileNet

### Detection
- **YOLOv8**: The industry standard for real-time, high-accuracy bounding box detection on agricultural edge-computing devices.

### Herbicide Recommendation
- **Fuzzy Logic Rule Engine**: Translates multi-variable field data—such as detected weed species, density, and growth stage—into precise chemical dosages without relying on a black-box AI model.

### Feature Extraction
- **ResNet Residual Backbones**: Automatically extracts hierarchical plant features like vein structure and leaf texture while preventing gradient vanishing.

### Explainable AI (XAI)
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Generates an intuitive, real-time visual heatmap over the exact leaf structures that triggered the weed classification.

Detection: Uses the best performing YOLOv8n model (agri_yolov8n) for consistency
(test mAP@50 **0.8651**, mAP@50-95 **0.6957**, F1 **0.8400**; 6.25 MB, ~88.6 FPS)。
The detector is a binary **groundnut (0) vs weed (1)** model — any non-groundnut
plant/crop/weed is mapped to the weed class.

---

## IMPORTANT — research-validity status (read before using for a thesis or in
a real agricultural context)

This is a **research prototype**. Several components are deliberately
labelled/handled conservatively because they are **not yet scientifically
validated**:

1. **Detector labels are CONFIRMED (binary groundnut-vs-weed).** The YOLO
   dataset config
   (`datasets/processed/groundnut_weed_detection/data.yaml`) maps the two
   classes to `groundnut` (Arachis hypogaea crop/plant) and `weed` (any other
   plant/crop/weed). Confirmed 2026-09-17. The app therefore reports detections
   honestly: class 0 = groundnut, class 1 = weed. See
   [`docs/DETECTOR_LABELS.md`](docs/DETECTOR_LABELS.md).
2. **Herbicide recommendations now use agronomically reviewed records.** The
   knowledge base (`herbicide/processed/herbicide_dataset_normalized.json`) has
   **37 `validation_status: passed`** records (cited doses/PHI from official
   regulatory, extension and research sources) and 5 records deliberately kept
   `pending` (conflict flags, missing dose, manual-only). The engine recommends
   ONLY from `passed` records; `GROUNDNUT_ALLOW_PENDING_RECORDS=1` additionally
   includes the pending records for research previews. Review status can be
   re-applied with `python herbicide/scripts/mark_records_passed.py`.
3. **Non-plant images are rejected before analysis.** A dedicated trained
   **plant-vs-other gate** (`backend/app/utils/plant_gate.py`,
   `ml/gate/train_plant_gate.py`) combines HSV colour, texture and ImageNet
   semantics to decide if an upload is a crop-field photo at all. Trained on
   6,956 plant images + 985 real negatives + 1,600 synthetic negatives
   (held-out **AUC 0.998**; conservative two-tier decision: clearly-plant
   accepted, clearly-non-plant rejected, ambiguous images resolved with
   detector + colour evidence — verified **0 of 60 real field images rejected**
   in sampling). `/analyze` returns `error_code=not_plant_image` with a message,
   and the app shows a "Retake Photo / Upload Image" exception screen instead of
   fake detections. Keep improving it by adding photos to
   `datasets/negatives_real/` and re-running
   `python ml/gate/train_plant_gate.py --negatives-dir datasets/negatives_real`.
4. **Density/coverage is an uncalibrated estimate.** Percent coverage is
   computed from detection-box area (union of boxes of the configured weed
   class) and the 5%/15% thresholds are **not field-calibrated**. The UI marks
   it "estimate (uncalibrated)".
5. **Grad-CAM is now a genuine gradient-based Grad-CAM** (class-score gradients
   w.r.t. the last neck conv; the input tensor is placed on the model's device).
   When the gradient path cannot run it falls back to a **clearly labelled**
   non-gradient activation overlay.
6. **The % shown in the app is model confidence, not accuracy.** Detections
   show YOLO box scores and the classification chip shows a 2-class softmax
   probability — these are not field-tested accuracy numbers. Real test-set
   accuracies are published in `docs/MODEL_PERFORMANCE_REPORT.md` and shown on
   the model-selection screen ("accuracy basis (held-out test set)").
7. The classifier zoo was trained on `datasets/processed/classification_groundnut_weed` (**3,738 patches**; 2,658/528/552 train/val/test, balanced, seed 42).
   groundnut-vs-weed classification set (Arachis hypogaea images + Peanut
   crop/weed mask patches + agri_data whole images), patch-level crop/weed
   discrimination, not field-image weed-species identification. Best classifier
   (hybrid ensemble) test accuracy **0.9946**; per-model results in
   `docs/MODEL_PERFORMANCE_REPORT.md`.

## Final model performance (rebuilt datasets, seed 42)

**Classifier zoo - groundnut vs weed** - test split (552 patches, 128x128):

|Model|Test accuracy|Precision|Recall|F1 (macro)|Inference (ms/img)|
|---|---:|---:|---:|---:|---:|
|efficientnet|0.9873|0.9856|0.9891|0.9873|4.2|
|mobilenet|0.9873|0.9927|0.9819|0.9873|3.6|
|resnet50|0.9873|0.9891|0.9855|0.9873|4.4|
|vgg16|0.9873|0.9963|0.9783|0.9873|5.1|
|cnn|0.9638|0.9741|0.9529|0.9638|3.2|

- **Detector** - `agri_yolov8n` (YOLOv8n, 484 test images): test mAP@50 **0.8651**, mAP@50-95 **0.6957**, F1 **0.8400**, precision 0.8867, recall 0.7980, 6.25 MB, ~88.6 FPS.

## Quick start (development)

```
# 1) Backend
cd backend
pip install -r requirements.txt
python run_server.py dev-my-secret-key 8000

# 2) App (Android phone on the same Wi-Fi)
.\run_project.ps1 -ApiKey dev-my-secret-key -AllowFirewall
```

For details see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Go live (free hosting — no credit card)

Backend on **Hugging Face Spaces** (4 vCPU / 15 GiB free CPU Docker Space),
frontend on **GitHub Pages** + a signed Android **APK**. Everything is scripted
in [`deploy/`](deploy/):

```powershell
.\deploy\package_space.ps1                                  # build the backend bundle
.\deploy\push_to_hf.ps1 -HfUser <hf_user> -Token hf_xxx     # push to a FREE Space
.\deploy\build_apps.ps1 -HfUser <hf_user> -WebBaseHref /Agri-Project/
.\deploy\deploy_web_ghpages.ps1                             # web app -> GitHub Pages
```

Step-by-step instructions, the Docker tuning notes and the troubleshooting
table are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `backend/` | FastAPI application (services, schemas, endpoints, tests) |
| `lib/` | Flutter app (screens, services, models, widgets) |
| `ml/` | Python ML pipeline (data, YOLO, segmentation, classifier zoo, XAI, density, synthesis, preprocessing, indexing) |
| `herbicide/` | Herbicide knowledge-base normalizer, validator, recommendation engine |
| `models/` | Production-inference wrapper |
| `runs/` `experiments/` | Trained models / experiment outputs |
| `datasets/` | Prepared datasets |
| `docs/` | Deployment, research-validity, algorithms, and reproducibility documentation |

For detailed information on all implemented algorithms, see [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## License / attribution

Research code. Sources for the herbicide knowledge base are recorded inline in
the normalized dataset; confirm with an agronomist before any field use.
