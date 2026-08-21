# API Documentation

Reference for the main programmatic entry points across StrokeRehabAI's
packages. This project does not expose a network API (it's a local
desktop application) — "API" here means the stable Python interfaces
each package exposes for reuse (e.g. from a Jupyter notebook, a custom
script, or a future research extension).

## Configuration

```python
from configs.config_loader import load_config

cfg = load_config()               # loads and caches all configs/*.yaml
cfg.training.batch_size            # dot-access
cfg["training"]["batch_size"]      # dict-access, equivalent
load_config(force_reload=True)     # bypass the cache, reparse YAML
```

## Real-time inference engine

```python
from inference.realtime_pipeline import RealtimeInferencePipeline

pipeline = RealtimeInferencePipeline(patient_id=3, skip_calibration=False)
pipeline.run()   # blocks; runs the full camera -> feedback -> dashboard loop
```

```python
from inference.movement_analyzer import MovementAnalyzer
from inference.exercise_library import ExerciseLibrary
from configs.config_loader import load_config

cfg = load_config()
analyzer = MovementAnalyzer(ExerciseLibrary(cfg.exercises), fps=30, exercises_cfg=cfg.exercises)
result = analyzer.analyze_frame(landmarks_xyz, angles, pose_confidence=0.9)
# result.exercise_key, result.phase, result.movement_quality, result.errors,
# result.repetition_assessment (on rep completion), result.explanation (every frame)
```

```python
from inference.explainable_ai import ExplainableAIEngine

explanation = ExplainableAIEngine().explain(result, definition, assessment=result.repetition_assessment)
print(explanation.to_display_string())   # Exercise / Confidence / Movement Quality / Reason / Recommendation
explanation.to_dict()                     # same, as a dict
```

## Clinical assessment & reasoning

```python
from inference.clinical_assessment import ClinicalAssessmentEngine, RepFrameSample

engine = ClinicalAssessmentEngine()
assessment = engine.assess_repetition(frames, exercise_definition, rep_completed=True, peak_progress_fraction=1.0)
assessment.to_dict()  # 8 scores, 0-100: quality, ROM, smoothness, stability, consistency, compensation, completion, overall
```

```python
from inference.clinical_reasoning import ClinicalReasoningEngine

reasoning = ClinicalReasoningEngine().evaluate(definition, assessment, errors, achieved_angle_deg=90)
for r in reasoning:
    print(r.condition, "->", r.recommendation, "|", r.explanation)
```

## Physiotherapy knowledge base

```python
from inference.knowledge_base import PhysiotherapyKnowledgeBase

kb = PhysiotherapyKnowledgeBase()  # loads knowledge_base/exercises_knowledge.yaml
entry = kb.get("shoulder_flexion")
entry.normal_rom_deg, entry.functional_rom_deg, entry.compensation_patterns, entry.correction_rules
```

## Dashboard / clinical data layer

```python
from dashboard.patient_manager import PatientManager, Patient
from dashboard.session_manager import SessionManager
from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.personalization_engine import PersonalizationEngine
from dashboard.recovery_tracking import RecoveryTracker
from dashboard.recommendation_engine import RecommendationEngine
from dashboard.alert_system import AlertSystem
from dashboard.report_generator import ReportGenerator

db_path = "outputs/database/stroke_rehab.db"

patient_manager = PatientManager(db_path)
patient_id = patient_manager.register(Patient(patient_id=None, full_name="Jane Doe"))

analytics = RecoveryAnalyticsEngine(db_path).compute_patient_analytics(patient_id)
plan = PersonalizationEngine(db_path).build_plan(patient_id)
recovery = RecoveryTracker(db_path).track(patient_id)
recommendations = RecommendationEngine().generate(analytics, exercise_display_name="Elbow Flexion")
alerts = AlertSystem(db_path).check_patient(patient_id)

report_path = ReportGenerator(db_path).generate(patient_id, report_format="pdf")
```

## Models

```python
from models.model_factory import build_model, data_representation_for
from configs.config_loader import load_config

cfg = load_config()
model = build_model(cfg.model)                       # ST-GCN (default) or LSTM
data_representation_for(cfg.model)                     # "graph" | "flat"
```

```python
from models.export_utils import export_torchscript, export_onnx, export_quantized

export_torchscript(model, cfg.model, "weights/exported/model.ts.pt")
export_onnx(model, cfg.model, "weights/exported/model.onnx")
export_quantized(model, cfg.model, "weights/exported/model_quantized.pt")
```

## Training

```python
from training.trainer import Trainer

trainer = Trainer()
trainer.fit()                          # builds dataloaders, trains, checkpoints, early-stops
trainer.fit(resume_path="last")         # resume from weights/checkpoints/last.pt
metrics = trainer.test()                 # accuracy/precision/recall/F1/ROC-AUC/confusion matrix
```

## Datasets

```python
from datasets.dataset_converter import DatasetConverter

converter = DatasetConverter(output_dir="data/processed")
converter.convert_custom(raw_dir, labels_csv=labels_csv_path)   # this project's own recordings
```

## Demo mode

```python
from inference.demo_mode import DemoModeRunner, DemoSourceMode, generate_demo_dataset

patient_id = generate_demo_dataset("outputs/database/stroke_rehab.db", num_sessions=8)

runner = DemoModeRunner()
runner.run(DemoSourceMode.WEBCAM)
runner.run(DemoSourceMode.VIDEO, source_path="sample.mp4")
runner.run(DemoSourceMode.SESSION, source_path=str(session_id))
```

## Error handling

```python
from utils.error_handling import (
    CameraError, CUDAUnavailableError, DatasetError, ModelWeightsError,
    CorruptedFileError, InvalidConfigurationError, LowMemoryError,
    format_user_facing_message, handle_gracefully, check_available_memory_mb,
)

@handle_gracefully(default_return=None, context="loading calibration")
def risky_operation():
    ...
```

## Validation

```python
from scripts.validate_project import run_validation

ok = run_validation()   # prints a full pass/fail report; returns bool
```
