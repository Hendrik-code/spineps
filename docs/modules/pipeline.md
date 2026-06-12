# Pipeline

SPINEPS processes a scan in a sequence of phases orchestrated by the functions in
[`spineps.seg_run`](../api/pipeline.md). There are two top-level entry points:

- **`segment_image`** — process a single image.
- **`process_dataset`** — discover and process every suitable scan in a [BIDS](https://bids-specification.readthedocs.io/en/stable/) dataset.

## Two-phase approach

The core idea is to split spine segmentation into two complementary tasks:

1. **Semantic phase** — a multi-class network labels every voxel with its anatomical *subregion*
   (vertebra subregions, spinal cord, spinal canal, discs, endplate, sacrum).
2. **Instance phase** — using the vertebra subregions from the semantic mask, individual vertebrae are
   separated into a per-vertebra *instance* mask.

A combined **post-processing** step then cleans both masks against each other, assigns intervertebral
discs and endplates to their parent vertebra, and (optionally) runs the **VERIDAH labeling** model to give
each instance an anatomical vertebra label.

```text
input scan
   │  pre-processing (normalize, optional N4 bias correction, crop, pad)
   ▼
semantic phase ──► subregion (semantic) mask
   │
   ▼
instance phase ──► per-vertebra instance mask
   │
   ▼
post-processing (clean, assign IVD/endplate, optional VERIDAH labeling)
   │
   ▼
seg-spine mask · seg-vert mask · centroids (.json) · snapshot (.png)
```

## Outputs

For each processed scan SPINEPS writes a derivatives folder next to the input containing:

- a `seg-spine` mask (semantic / subregion segmentation),
- a `seg-vert` mask (vertebra instance segmentation),
- a centroid file (`.json`) with points of interest for each vertebra, endplate and disc,
- a snapshot `.png` visualizing the result,
- optionally: an uncertainty image, the model-resolution masks, softmax logits and debug data.

## Calling from Python

The high-level `spineps.segment` API loads the models and runs the pipeline in one call:

```python
import spineps

result = spineps.segment("sub-test_T2w.nii.gz")          # saves a derivatives folder next to the input
result = spineps.segment(nii, output_in_memory=True)     # or get the masks back in memory
```

Use `SpinepsPipeline` to load the models once and segment many images, and the `SemanticConfig` / `InstanceConfig`
/ `LabelingConfig` / `PostConfig` objects to group processing options:

```python
from spineps import SpinepsPipeline, InstanceConfig

pipe = SpinepsPipeline(model_semantic="t2w", model_instance="instance")
for path in paths:
    pipe.segment(path, instance=InstanceConfig(batch_size=8))
```

For full control, call the underlying `segment_image` directly with already-loaded models:

```python
from TPTBox import BIDS_FILE
from spineps import get_semantic_model, get_instance_model, segment_image

semantic = get_semantic_model("t2w")
instance = get_instance_model("instance")

segment_image(
    BIDS_FILE("sub-test_T2w.nii.gz", dataset="/path/to/dataset"),
    model_semantic=semantic,
    model_instance=instance,
)
```

`segment_image` exposes many `proc_*` flags to toggle individual processing steps (pre-processing,
semantic/instance cleaning, hole filling, labeling, …). See the
[Pipeline & Run API reference](../api/pipeline.md) for the full signature.

For batch processing, `process_dataset` accepts the same processing flags and applies them to every
matching scan it finds.
