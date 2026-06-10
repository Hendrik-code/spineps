# Migrating to SPINEPS 2.0

SPINEPS 2.0 is a clean-break release: CLI flags, public functions and classes were renamed for clarity and
consistency, and a new one-call Python API was added. **No backward-compatible aliases are kept** — update your
scripts using the tables below.

## TL;DR — the new one-call API

The biggest change is that you no longer have to load three models and pass dozens of flags:

```python
import spineps

# Save a BIDS derivatives folder next to the input:
result = spineps.segment("sub-01_T2w.nii.gz")

# Or get the masks in memory:
result = spineps.segment(nii, output_in_memory=True)
if result.success:
    semantic, vertebra = result.semantic, result.vertebra

# Segmenting many images? Load the models once:
from spineps import SpinepsPipeline
pipe = SpinepsPipeline(model_semantic="t2w", model_instance="instance")
for path in paths:
    pipe.segment(path)
```

## CLI flags

All long flags are now `--kebab-case` (short aliases are unchanged). Negative flags became positive on/off pairs.

| Old | New |
| --- | --- |
| `-nocrop` / `-nc` | `--no-crop` (cropping is on by default; pass `--no-crop` to disable) |
| `-non4` | `--no-n4` (N4 is on by default; pass `--no-n4` to disable) |
| `-no_tltv_labeling` / `-ntl` | `--enforce-12-thoracic` |
| `-input` / `-i` | `--input` / `-i` |
| `-directory` / `-d` | `--directory` / `-d` |
| `-model_semantic` / `-ms` | `--model-semantic` / `-ms` |
| `-model_instance` / `-mv` | `--model-instance` / `-mv` / `-mi` |
| `-model_labeling` / `-ml` | `--model-labeling` / `-ml` |
| `-der_name` / `-dn` | `--derivative-name` / `-dn` |
| `-raw_name` / `-rn` | `--rawdata-name` / `-rn` |
| `-save_debug` / `-sd` | `--save-debug` / `-sd` |
| `-save_softmax_logits` / `-ssl` | `--save-softmax-logits` / `-ssl` |
| `-save_modelres_mask` / `-smrm` | `--save-modelres-mask` / `-smrm` |
| `-save_log` / `-sl` | `--save-log` / `-sl` |
| `-save_snaps_folder` / `-ssf` | `--save-snaps-folder` / `-ssf` |
| `-override_semantic` / `-os` | `--override-semantic` / `-os` |
| `-override_instance` / `-oi` | `--override-instance` / `-oi` |
| `-override_postpair` / `-opp` | `--override-postpair` / `-opp` |
| `-override_ctd` / `-oc` | `--override-ctd` / `-oc` |
| `-ignore_inference_compatibility` / `-iic` | `--ignore-inference-compatibility` / `-iic` |
| `-ignore_bids_filter` / `-ibf` | `--ignore-bids-filter` / `-ibf` |
| `-ignore_model_compatibility` / `-imc` | `--ignore-model-compatibility` / `-imc` |
| `-run_cprofiler` / `-rcp` | `--run-cprofiler` / `-rcp` |
| `-cpu` | `--cpu` |
| `-verbose` / `-v` | `--verbose` / `-v` |
| *(new)* | `--batch-size` / `-bs` — vertebra cutouts per batched forward pass (faster; default 4) |

Example:

```bash
# 1.x
spineps sample -i scan.nii.gz -model_semantic t2w -model_instance instance -nocrop -non4

# 2.0
spineps sample -i scan.nii.gz --model-semantic t2w --model-instance instance --no-crop --no-n4
```

## Python API

| Old | New |
| --- | --- |
| `process_img_nii(...)` | `segment_image(...)` |
| `Segmentation_Model` | `SegmentationModel` |
| `Segmentation_Model_NNunet` | `SegmentationModelNNunet` |
| `Segmentation_Model_Unet3D` | `SegmentationModelUnet3D` |

`process_dataset`, `get_semantic_model`, `get_instance_model`, `get_labeling_model`, `predict_semantic_mask`,
`predict_instance_mask` and the other phase functions keep their names.

## New in 2.0

- **`spineps.segment(...)`**, **`SpinepsPipeline`**, **`SpinepsResult`** — the high-level API shown above.
- **Config objects** `SemanticConfig`, `InstanceConfig`, `LabelingConfig`, `PostConfig` group the many `proc_*`
  flags. Pass them to `segment(...)`, e.g. `spineps.segment(path, instance=InstanceConfig(batch_size=8))`.
- **`--batch-size`** / `InstanceConfig.batch_size` — the instance model now runs cutouts in batched forward passes
  (much faster on GPU); falls back to one-by-one on out-of-memory.
- Clearer errors: invalid paths / missing models now raise `FileNotFoundError` / `ValueError` instead of bare
  `AssertionError`, and `spineps sample -h` / `dataset -h` no longer crash.
