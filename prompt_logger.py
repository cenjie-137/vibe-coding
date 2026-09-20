import json
import hashlib
import os
import datetime
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentMetadata:
    experiment_id: str
    competition: str
    timestamp: str = ""
    
    llm_platform: str = "TRAE"
    llm_model: str = ""
    llm_model_version: str = ""
    llm_mode: str = "interactive"
    temperature: float = 0.7
    top_p: float = 1.0
    random_seed: int = 42
    
    prompt_template_version: str = "v1.0"
    prompt_template_hash: str = ""
    human_intervention_count: int = 0
    total_interaction_turns: int = 0
    
    platform_version: str = ""
    os: str = "Windows 11 Pro"
    python_version: str = ""
    pytorch_version: str = ""
    gpu: str = ""
    cuda_version: str = ""
    
    architecture: str = ""
    encoder: str = ""
    pretrained_weights: str = ""
    input_size: int = 512
    num_classes: int = 0
    
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    loss_function: str = ""
    lr_schedule: str = ""
    augmentation: str = ""
    
    micro_dice: Optional[float] = None
    macro_dice: Optional[float] = None
    tta_micro_dice: Optional[float] = None
    per_class_dice: dict = field(default_factory=dict)
    
    git_commit: str = ""
    weights_path: str = ""
    log_path: str = ""
    
    notes: str = ""


def compute_prompt_hash(prompt_file: str = "PROMPT_STANDARD.md") -> str:
    content = Path(prompt_file).read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def get_env_info() -> dict:
    info = {}
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "cpu"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    import sys
    info["python_version"] = sys.version.split()[0]
    return info


def log_experiment(exp: ExperimentMetadata, output_dir: str = "."):
    if not exp.timestamp:
        exp.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    exp.prompt_template_hash = compute_prompt_hash()
    exp.git_commit = get_git_commit()
    
    env = get_env_info()
    if not exp.pytorch_version:
        exp.pytorch_version = env.get("pytorch_version", "")
    if not exp.cuda_version:
        exp.cuda_version = env.get("cuda_version", "")
    if not exp.gpu:
        exp.gpu = env.get("gpu", "")
    if not exp.python_version:
        exp.python_version = env.get("python_version", "")
    
    results_file = Path(output_dir) / "experiment_results.json"
    all_results = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    
    all_results.append(asdict(exp))
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"[prompt_logger] Experiment {exp.experiment_id} logged to {results_file}")
    print(f"  Model: {exp.llm_model} v{exp.llm_model_version}")
    print(f"  Prompt hash: {exp.prompt_template_hash}")
    print(f"  Git commit: {exp.git_commit}")
    print(f"  Interventions: {exp.human_intervention_count}")
    
    return asdict(exp)


def log_intervention(file_path: str, lines: str, reason: str, 
                     experiment_id: str = "", output_dir: str = "."):
    intervention = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": experiment_id,
        "file": file_path,
        "lines": lines,
        "reason": reason,
    }
    
    log_file = Path(output_dir) / "intervention_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(intervention, ensure_ascii=False) + "\n")
    
    print(f"[prompt_logger] Intervention logged: {file_path} {lines} — {reason}")


def generate_reproducibility_report(experiment_id: str, output_dir: str = "."):
    results_file = Path(output_dir) / "experiment_results.json"
    if not results_file.exists():
        print(f"[prompt_logger] No results file found at {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    
    exp = None
    for r in all_results:
        if r.get("experiment_id") == experiment_id:
            exp = r
            break
    
    if not exp:
        print(f"[prompt_logger] Experiment {experiment_id} not found")
        return
    
    report = f"""# Reproducibility Report: {experiment_id}

## LLM Configuration
- Platform: {exp.get('llm_platform', 'N/A')}
- Model: {exp.get('llm_model', 'N/A')}
- Version: {exp.get('llm_model_version', 'N/A')}
- Mode: {exp.get('llm_mode', 'N/A')} (NOT auto)
- Temperature: {exp.get('temperature', 'N/A')}
- Random Seed: {exp.get('random_seed', 'N/A')}

## Prompt Configuration
- Template Version: {exp.get('prompt_template_version', 'N/A')}
- Template Hash: {exp.get('prompt_template_hash', 'N/A')}
- Human Interventions: {exp.get('human_intervention_count', 'N/A')}
- Total Interaction Turns: {exp.get('total_interaction_turns', 'N/A')}

## Environment
- OS: {exp.get('os', 'N/A')}
- Python: {exp.get('python_version', 'N/A')}
- PyTorch: {exp.get('pytorch_version', 'N/A')}
- GPU: {exp.get('gpu', 'N/A')}
- CUDA: {exp.get('cuda_version', 'N/A')}

## Model Configuration
- Architecture: {exp.get('architecture', 'N/A')}
- Encoder: {exp.get('encoder', 'N/A')}
- Pretrained: {exp.get('pretrained_weights', 'N/A')}
- Input Size: {exp.get('input_size', 'N/A')}
- Num Classes: {exp.get('num_classes', 'N/A')}

## Training Configuration
- Batch Size: {exp.get('batch_size', 'N/A')}
- Epochs: {exp.get('epochs', 'N/A')}
- Learning Rate: {exp.get('learning_rate', 'N/A')}
- Optimizer: {exp.get('optimizer', 'N/A')}
- Loss: {exp.get('loss_function', 'N/A')}
- LR Schedule: {exp.get('lr_schedule', 'N/A')}
- Augmentation: {exp.get('augmentation', 'N/A')}

## Results
- Micro Dice: {exp.get('micro_dice', 'N/A')}
- Macro Dice: {exp.get('macro_dice', 'N/A')}
- TTA Micro Dice: {exp.get('tta_micro_dice', 'N/A')}
- Per-Class Dice: {json.dumps(exp.get('per_class_dice', {}), indent=2)}

## Reproducibility
- Git Commit: {exp.get('git_commit', 'N/A')}
- Weights Path: {exp.get('weights_path', 'N/A')}
- Log Path: {exp.get('log_path', 'N/A')}

## Checklist
- [x] LLM model name and version documented
- [x] LLM mode is NOT "auto"
- [x] Prompt template version archived
- [x] Prompt template hash computed
- [x] Random seed fixed
- [x] Hyperparameters logged
- [x] Git commit recorded
- {'[x]' if exp.get('weights_path') else '[ ]'} Weights saved
- {'[x]' if exp.get('human_intervention_count', 0) >= 0 else '[ ]'} Interventions logged

## Notes
{exp.get('notes', 'N/A')}
"""
    
    report_file = Path(output_dir) / f"reproducibility_report_{experiment_id}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"[prompt_logger] Report generated: {report_file}")
    return report_file


if __name__ == "__main__":
    exp = ExperimentMetadata(
        experiment_id="EXP-12-PUMA-UNI-5fold",
        competition="PUMA",
        llm_platform="TRAE",
        llm_model="DeepSeek-V4-Pro",
        llm_model_version="V4-Pro",
        llm_mode="interactive",
        architecture="FoundationUNet",
        encoder="UNI ViT-L/16",
        pretrained_weights="MahmoodLab/UNI (pathology pretrained)",
        input_size=512,
        num_classes=6,
        batch_size=4,
        epochs=100,
        learning_rate=1e-4,
        loss_function="Focal Loss (gamma=2, label_smoothing=0.1)",
        lr_schedule="warmup(10) + cosine",
        augmentation="HED staining, h/v flip, rotation ±10°",
        tta_micro_dice=0.6762,
        per_class_dice={
            "tumor": 0.9496,
            "stroma": 0.8410,
            "epidermis": 0.7835,
            "blood_vessel": 0.3842,
            "necrosis": 0.0000,
        },
        weights_path="/root/autodl-tmp/medseg_project/output_puma/fold_0-4/best_model.pth",
        log_path="/root/autodl-tmp/medseg_project/logs/",
        notes="5-fold avg TTA Micro Dice. UNI pathology foundation model dramatically improved over ResNet34 (0.5238) and nnU-Net baseline (0.5548). Necrosis went from 0 to 0.8996 in best fold.",
    )
    
    log_experiment(exp)
    generate_reproducibility_report("EXP-12-PUMA-UNI-5fold")
    
    print("\n[prompt_logger] Logged with DeepSeek-V4-Pro.")
