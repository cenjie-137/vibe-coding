# AI-Assisted Medical Image Segmentation: Standardized Prompt Template v1.1

> This template ensures reproducible AI-assisted experiments across multiple
> medical image segmentation competitions. Every experiment MUST use this
> template with task-specific variables filled in.

---

## Metadata Header (Logged for Every Experiment)

```yaml
experiment_id: "EXP-XXX"
timestamp: "2026-MM-DD HH:MM:SS"
competition: "COMPETITION_NAME"

# LLM Configuration
llm_platform: "TRAE"                   # or "Claude Desktop", "VS Code Copilot"
llm_model: "DeepSeek-V4-Pro"          # FIXED for this project (open-source, versioned)
llm_model_version: "V4-Pro"           # FIXED for this project
llm_mode: "interactive"               # NOT "auto" — must be specified model
temperature: 0.7
top_p: 1.0
random_seed: 42                       # if platform supports it

# Prompt Configuration
prompt_template_version: "v1.1"
prompt_template_hash: "sha256:COMPUTED_HASH"
human_intervention_count: 0           # incremented on each manual code edit
total_interaction_turns: 0            # number of LLM round-trips

# Environment
platform_version: "TRAE vX.X.X"
os: "Windows 11 Pro"
python_version: "3.10.x"
pytorch_version: "2.x.x"
gpu: "RTX 4090 / A100"               # server GPU
cuda_version: "12.1"

# Reproducibility Checklist
checklist:
  model_version_logged: true
  prompt_template_archived: true
  random_seed_fixed: true
  data_split_documented: true
  environment_pinned: true            # Dockerfile or requirements.txt
  code_committed: true                # git commit hash
  weights_archived: true              # model weights saved
```

---

## SOTA Research Protocol (Step 0 — Run Before ANY Code)

> **Mandatory**: Every competition starts with SOTA research. No code is written
> before this step completes and results are saved to RESEARCH_LOG.md.

### Step 0 Prompt (Fixed — Run First, Research Only)

```
## SOTA Research Task

Research the following competition/dataset. Do NOT write code — research only.

## Research Target
- Name: [COMPETITION_NAME]
- URL: [COMPETITION_URL]
- Task: [TASK_TYPE]

## Required Dimensions (must cover ALL):
1. Dataset details — size, image dims, modality, class distribution, splits
2. Evaluation protocol — primary metric, how ranking works
3. SOTA methods — list 5-10, with year, architecture, reported score, contribution
4. Winning strategies — pretraining, foundation model usage, loss, augmentation,
   post-processing, ensemble/TTA
5. Failure modes — which classes/structures are hard, common mistakes

## Output (fill this YAML, save to RESEARCH_LOG.md):
```

### Step 0 Result Format (Saved to RESEARCH_LOG.md)

```yaml
SOTA_RESEARCH:
  dataset:
    size: "[SIZE]"
    modality: "[MODALITY]"
    splits: "[TRAIN/TEST]"
  evaluation:
    primary_metric: "[METRIC]"
  sota_methods:
    - name: "[METHOD]"
      year: "[YEAR]"
      score: "[SCORE]"
      contribution: "[KEY_IDEA]"
  winning_strategies:
    pretraining: "[ImageNet vs domain-specific]"
    foundation_model: "[UNI/Virchow/CONCH; freeze vs fine-tune]"
    loss: "[loss choice]"
    augmentation: "[modality-specific aug]"
    postprocess: "[CRF, min-size, morph ops]"
    ensemble: "[k-fold, TTA, multi-seed]"
  failure_modes:
    - "[hard class/structure]"
  lessons_for_our_pipeline:
    - "[what to adopt in our standardized workflow]"
```

### Step 0 → Step 1 Transition Rule

Only after `SOTA_RESEARCH` is saved to RESEARCH_LOG.md may coding begin.
The Task Prompt below MUST reference the findings (e.g., set `AUGMENTATION`
and `FAILURE_MODES` from the research, not from memory).

---

## Standard Prompt Template

The following text is the FIXED prompt given to the LLM. Only the
`[VARIABLE]` placeholders change per competition.

---

### System Prompt (Fixed — Never Modified)

```
You are an expert medical image analysis researcher and PyTorch engineer
specializing in semantic segmentation. Your task is to write complete,
executable Python code for a medical image segmentation competition.

Rules:
1. Write production-quality PyTorch code with type hints
2. Use 5-fold stratified cross-validation
3. Set random seed = 42 for all stochastic operations
4. Include data augmentation appropriate for the imaging modality
5. Use mixed precision training (AMP) for efficiency
6. Log all hyperparameters and results to a JSON file
7. Save best model weights per fold
8. Implement Test-Time Augmentation (horizontal + vertical flip)
9. Pin all library versions in requirements.txt
10. Do NOT add error handling for impossible scenarios
11. Do NOT add comments unless explaining a non-obvious constraint
12. Output files: models.py, dataset.py, train.py, inference.py
13. Apply SOTA findings from RESEARCH_LOG.md (loss, augmentation, postprocess,
    encoder) — do NOT invent strategies from generic knowledge
```

### Task Prompt (Variable — Filled Per Competition)

```
## Competition: [COMPETITION_NAME]
## URL: [COMPETITION_URL]

## Task Description
[TASK_DESCRIPTION]

## Dataset
- Name: [DATASET_NAME]
- Modality: [MODALITY] (e.g., H&E histopathology, fundus photography)
- Image format: [FORMAT] (e.g., .tif, .png, .nii.gz)
- Image size: [WIDTH]x[HEIGHT]
- Training samples: [N_TRAIN]
- Test samples: [N_TEST]
- Classes: [CLASS_NAMES_AND_IDS]
- Class distribution: [CLASS_PERCENTAGES]
- Annotation format: [ANNOTATION_FORMAT]

## Evaluation
- Metric: [METRIC_NAME] (e.g., Micro Dice, Macro Dice, mIoU)
- Metric formula: [FORMULA_OR_URL]
- Submission format: [SUBMISSION_FORMAT]

## Competition Constraints
- External data allowed: [YES/NO]
- Pretrained models allowed: [YES/NO]
- GPU memory: [GPU_MEMORY_GB]
- Inference time limit: [TIME_LIMIT]

## Model Architecture
- Architecture: [ARCHITECTURE_NAME] (e.g., U-Net, SwinUNETR, FoundationUNet)
- Encoder: [ENCODER_NAME] (e.g., ResNet34, UNI, ConvNeXt V2 Base)
- Pretrained weights: [WEIGHTS_SOURCE] (e.g., ImageNet, pathology foundation model)
- Input size: [INPUT_SIZE]
- Num classes: [NUM_CLASSES]

## Training Configuration
- Batch size: [BATCH_SIZE]
- Epochs: [EPOCHS]
- Learning rate: [LR]
- Optimizer: [OPTIMIZER]
- Loss function: [LOSS_FUNCTION]
- LR schedule: [SCHEDULE]
- Data augmentation: [AUGMENTATION_LIST]

## Output Requirements
1. models.py — model architecture definition
2. dataset.py — data loading, preprocessing, augmentation
3. train.py — training loop with 5-fold CV, logs, checkpointing
4. inference.py — inference with TTA, submission generation
5. requirements.txt — pinned dependencies
6. config.json — all hyperparameters in JSON

## Reproducibility Requirements
- Fixed random seed = 42
- Log all intermediate results
- Save training curves
- Document any manual interventions (code edits, prompt modifications)
```

---

## Per-Competition Variable Fill-in

### PUMA (Completed)
```yaml
COMPETITION_NAME: "PUMA Challenge"
COMPETITION_URL: "https://puma.grand-challenge.org/"
TASK_DESCRIPTION: "5-class tissue semantic segmentation in H&E stained melanoma histopathology"
MODALITY: "H&E histopathology"
FORMAT: ".tif (1024x1024)"
N_TRAIN: 206
CLASSES: "0=background, 1=stroma, 2=blood_vessel, 3=tumor, 4=epidermis, 5=necrosis"
ANNOTATION_FORMAT: "GeoJSON polygons → label masks"
METRIC: "Micro Dice (per-class averaged, excluding background)"
ARCHITECTURE: "FoundationUNet (U-Net decoder + UNI encoder)"
ENCODER: "UNI ViT-L/16 (MahmoodLab, pathology pretrained)"
INPUT_SIZE: "512x512"
BATCH_SIZE: 4
EPOCHS: 100
LR: 1e-4
LOSS: "Focal Loss (gamma=2, label_smoothing=0.1)"
AUGMENTATION: "HED staining, horizontal/vertical flip, rotation ±10°"
```

### GLaS (Template — Fill When Ready)
```yaml
COMPETITION_NAME: "GLaS Gland Segmentation"
COMPETITION_URL: "https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/"
TASK_DESCRIPTION: "Binary gland segmentation in H&E stained colorectal histology"
MODALITY: "H&E histopathology"
FORMAT: ".bmp (~775x522)"
N_TRAIN: 85
N_TEST: 80
CLASSES: "0=background, 1=gland (object-level evaluation)"
METRIC: "Object-level Dice and F1 (benign + malignant separately)"
ARCHITECTURE: "FoundationUNet (U-Net decoder + UNI encoder)"
ENCODER: "UNI ViT-L/16 (pathology pretrained)"
INPUT_SIZE: "512x512"
# ... fill remaining fields from SOTA research
```

### DRIVE (Completed)
```yaml
COMPETITION_NAME: "DRIVE Retinal Vessel Segmentation"
COMPETITION_URL: "https://drive.grand-challenge.org/"
TASK_DESCRIPTION: "Binary retinal vessel segmentation in fundus images"
MODALITY: "Fundus photography"
FORMAT: ".tif (565x584)"
N_TRAIN: 20
CLASSES: "0=background, 1=vessel"
METRIC: "Dice coefficient"
ARCHITECTURE: "MultiResUNet + SE attention"
ENCODER: "ResNet34 (ImageNet pretrained)"
INPUT_SIZE: "512x512"
```

---

## Manual Intervention Log

Every time the human manually edits AI-generated code, log it:

```yaml
intervention_001:
  timestamp: "2026-09-01 14:30"
  file: "models.py"
  lines: "181-200"
  reason: "Added dynamic_img_size=True for ViT position encoding"
  diff: "git diff hash abc123"

intervention_002:
  timestamp: "2026-09-01 15:00"
  file: "train_puma.py"
  lines: "45-50"
  reason: "Added differential learning rate for encoder vs decoder"
  diff: "git diff hash def456"
```

---

## Reproducibility Checklist (Must Pass Before Submission)

- [ ] LLM model name and version documented
- [ ] LLM mode is NOT "auto" (specific model selected)
- [ ] Prompt template version archived (this file)
- [ ] Prompt template hash computed and logged
- [ ] Random seed = 42 set in all scripts
- [ ] 5-fold data split file saved (not regenerated)
- [ ] All hyperparameters logged to JSON
- [ ] requirements.txt with pinned versions
- [ ] Dockerfile or environment snapshot saved
- [ ] Git commit hash recorded for final code
- [ ] Model weights saved per fold
- [ ] Training logs saved (loss curves, metrics)
- [ ] Manual interventions logged with diffs
- [ ] Inference script produces deterministic output (same input → same output)
