# Processing Phases

Each scan flows through the following phases. The API for every phase is documented under
[Processing Phases (API)](../api/phases.md).

## Pre-processing — [`spineps.phase_pre`](../api/phases.md)

Prepares the input image: intensity normalization into a fixed range, optional N4 bias-field
correction, cropping to the non-zero region (optionally auto-cropping to the spine using VIBESeg), and
edge padding.

## Semantic phase — [`spineps.phase_semantic`](../api/phases.md)

Runs the semantic model to produce the subregion mask, then post-processes it: removing small
connected-component artifacts, optionally restricting to structures near the spinal canal, keeping only
components inside the largest spine bounding box, and filling 3D holes.

The semantic mask uses these labels:

| Label | Structure                |
| :---: | ------------------------ |
| 41    | Arcus_Vertebrae          |
| 42    | Spinosus_Process         |
| 43    | Costal_Process_Left      |
| 44    | Costal_Process_Right     |
| 45    | Superior_Articular_Left  |
| 46    | Superior_Articular_Right |
| 47    | Inferior_Articular_Left  |
| 48    | Inferior_Articular_Right |
| 49    | Vertebra_Corpus_border   |
| 60    | Spinal_Cord              |
| 61    | Spinal_Canal             |
| 62    | Endplate                 |
| 100   | Vertebra_Disc            |
| 26    | Sacrum                   |

## Instance phase — [`spineps.phase_instance`](../api/phases.md)

Derives a per-vertebra instance mask from the vertebra subregions. For each corpus center of mass it
extracts a cutout, runs the instance model, and merges the overlapping per-vertebra predictions into a
single label map. It also detects and splits merged vertebral bodies.

In the instance mask, each label `X` in `[1, 25]` is a unique vertebra; `100 + X` is that vertebra's
intervertebral disc and `200 + X` its endplate.

## Labeling phase (VERIDAH) — [`spineps.phase_labeling`](../api/phases.md)

Optional. Assigns each detected vertebra an anatomical label using a classifier and a min-cost path
solver ([`find_most_probably_sequence`](../api/utils.md)) that enforces a plausible cranio-caudal
ordering and handles transitional-vertebra anomalies (e.g. T13, L6).

| Label   | Structure  |
| :-----: | ---------- |
| 1       | C1         |
| 2 – 7   | C2 – C7    |
| 8 – 19  | T1 – T12   |
| 28      | T13        |
| 20      | L1         |
| 21 – 25 | L2 – L6    |
| 26      | Sacrum     |

As in the instance mask, `100 + X` is a vertebra's IVD and `200 + X` its endplate (e.g. label 119 is the
IVD below T12).

## Post-processing — [`spineps.phase_post`](../api/phases.md)

Combines the semantic and instance masks: cleans them against each other, assigns intervertebral discs
and endplates to their parent vertebra, resolves merged vertebrae, fixes mislabeled posterior elements,
and labels the instances top-to-bottom (or via VERIDAH when a labeling model is supplied).
