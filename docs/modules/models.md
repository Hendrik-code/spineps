# Models & Labeling

## Loading models

Models are referenced by name and resolved against the configured models directory (or downloaded
automatically). The convenience loaders live in [`spineps.get_models`](../api/models.md):

```python
from spineps import get_semantic_model, get_instance_model, get_labeling_model

semantic = get_semantic_model("t2w")
instance = get_instance_model("instance")
labeling = get_labeling_model("labeling")   # optional, VERIDAH
```

Each loader looks the name up in a model-id-to-folder map, resolves remote (HTTP) entries by downloading
the weights if needed, reads the model's `inference_config.json` and instantiates the right model class.

## Model types

The concrete model classes are defined in [`spineps.seg_model`](../api/models.md) and
[`spineps.lab_model`](../api/models.md):

- **`Segmentation_Model`** — abstract base wrapping a network plus its inference configuration. It handles
  input preparation (reorientation, rescaling to the recommended zoom, padding), running the model and
  mapping outputs back into the input space.
- **`Segmentation_Model_NNunet`** — an nnU-Net backend.
- **`Segmentation_Model_Unet3D`** — a 3D U-Net backend.
- **`VertLabelingClassifier`** — the VERIDAH vertebra-labeling classifier.

The model type is selected from the `inference_config.json` via `ModelType` (see
[Enums & Config](../api/enums.md)).

## Inference configuration

Every model ships an `inference_config.json` describing its expected inputs, modality and acquisition,
recommended resolution range, label mapping and processing thresholds. It is parsed into a
`Segmentation_Inference_Config` (see [`spineps.utils.seg_modelconfig`](../api/enums.md)).

## VERIDAH labeling

VERIDAH ("solving Enumeration Anomaly Aware Vertebra Labeling across Imaging Sequences") assigns
anatomical labels to the detected vertebra instances. It combines a per-vertebra classifier with a
min-cost path solver that enforces a plausible ordering and is aware of enumeration anomalies such as a
13th thoracic vertebra (T13) or a sixth lumbar vertebra (L6). Enable it by passing a labeling model to
the pipeline (`-model_labeling` on the CLI, or `model_labeling=` in Python).
