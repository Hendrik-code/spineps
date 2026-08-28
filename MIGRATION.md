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

All long flags are now `--kebab-case`; short aliases are unchanged wherever the flag kept its meaning. Negative flags became positive on/off pairs, and because their polarity flipped their old short aliases (`-nc`, `-ntl`) were dropped rather than silently reused for the opposite behaviour.

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
| *(new)* | `--amp` — run the instance forward pass under CUDA autocast (faster, may slightly change the output) |
| *(new)* | `--step-size` — semantic model sliding-window tile step size (larger = faster, less accurate) |
| *(new)* | `--tta` / `--no-tta` — force test-time augmentation (mirroring) on/off for the semantic model |

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

### Removed

| Removed | Why / what to do instead |
| --- | --- |
| `--model-semantic auto` (CLI) and `process_dataset(model_semantic=None)` | The auto-selection it depended on (`spineps.seg_utils.find_best_matching_model`) was never implemented and always raised `NotImplementedError`. Name a model explicitly. |
| `spineps.seg_utils.find_best_matching_model` | See above. |
| `spineps.utils.image` (vendored spinalcordtoolbox `Image`) and `spineps.utils.generate_disc_labels` | Standalone disc-label export, wired to no entry point. Derive disc labels from the vertebra mask with `TPTBox` instead. |
| `spineps.architectures_new.unet2D` | The instance model is 3D only; `PLNet(do2D=True)` now raises. |
| `spineps.example` scripts | Never shipped in the wheel; see the README for usage examples. |
| `spineps.seg_pipeline.pipeline_revision` | Centroid metadata no longer records a git revision (see below). |
| `SPINEPS_TURN_OF_CITATION_REMINDER` | Renamed to `SPINEPS_NO_CITATION_REMINDER` (and it now actually works). |

`process_dataset`, `get_semantic_model`, `get_instance_model`, `get_labeling_model`, `predict_semantic_mask`,
`predict_instance_mask` and the other phase functions keep their names.

## New in 2.0

- **`spineps.segment(...)`**, **`SpinepsPipeline`**, **`SpinepsResult`** — the high-level API shown above.
- **Config objects** `SemanticConfig`, `InstanceConfig`, `LabelingConfig`, `PostConfig` group the many `proc_*`
  flags. Pass them to `segment(...)`, e.g. `spineps.segment(path, instance=InstanceConfig(batch_size=8))`.
- **`--batch-size`** / `InstanceConfig.batch_size` — the instance model now runs cutouts in batched forward passes
  (much faster on GPU); falls back to one-by-one on out-of-memory.
- **More speed knobs**: `--amp` / `InstanceConfig.amp` (instance autocast), `--step-size` / `SemanticConfig.step_size`
  (semantic sliding-window step), and `--tta` / `--no-tta` (toggle test-time mirroring; `SegmentationModel.set_test_time_augmentation(...)` in Python).
- Clearer errors: invalid paths / missing models now raise `FileNotFoundError` / `ValueError` instead of bare
  `AssertionError`, and `spineps sample -h` / `dataset -h` no longer crash.

## Fixed in 2.0 — output may change

These are bug fixes, so results can differ from 1.x. Each was wrong before.

- **Endplate labels reach the semantic mask.** The superior/inferior split was computed and then thrown away
  by a binarising `extract_label`, so the `msk` output carried endplate voxels labelled `1` instead of
  `Vertebral_Body_Endplate_Superior` (52) / `_Inferior` (53). If you worked around this, remove the workaround.
- **The semantic bounding-box clean keeps what it incorporates.** It grew a region to take in nearby connected
  components and then cropped to the *largest* component's box anyway, deleting the rest. Spines split across
  components (gaps, implants) keep more of the mask now.
- **No phantom disc.** `detect_and_solve_merged_vertebra` offset every voxel including the background, adding a
  volume-sized fake IVD to the height-sorted list the split-C2 heuristic reads.
- **Small connected-component cleaning actually runs.** With `only_delete=False`, the neighbourhood mask was
  destroyed before it was used, so nothing was ever deleted or relabelled despite the log saying otherwise.
- **Merged-corpus splitting.** `get_separating_components` returned its two parts *after* dilating them in
  place, so they overlapped and the separating plane was derived from smeared centers of mass.
- **Incompatible models stop the run.** `process_dataset` logged "stop program" and then carried on; it now
  raises `ValueError` unless `ignore_model_compatibility=True` / `--ignore-model-compatibility`.
- **Non-overlapping instance partners.** The couple search detected two partners that agree with the anchor but
  not with each other, logged that it was skipping them, and used both anyway.
- **Version metadata.** `ctd.info["version"]` is the installed package version. It used to shell out to `git`
  with no working directory, so it recorded whatever repository you happened to be standing in (or
  `"Version not found"`), and `ctd.info["revision"]` is gone.
- **Labeling no longer crashes** on an empty instance mask, or with `disable_c1=False` and no subregion mask.
- **`import spineps` no longer creates a directory** inside the installed package; the fallback models folder is
  created on demand.

## Faster and smaller

No output change -- pinned by voxel-identical regression tests.

- The instance phase keeps each cutout prediction where it lives instead of in a dense
  `(n_vertebrae, 3, *volume)` array, and compares candidates on their overlapping region only.
- The endplate splitter runs per vertebra on numpy arrays and grows each dilation by one voxel per round
  instead of re-dilating from scratch.
- `clean_cc_artifacts` builds connected components one label at a time and works inside each component's
  bounding box.
- The input volume is read from disk once per image instead of up to three times.

On a 24-vertebra, 8.1M-voxel whole-spine volume: instance phase 2.30s -> 1.00s (peak RSS +313MB -> +157MB),
combined post-processing 3.34s -> 0.59s.
