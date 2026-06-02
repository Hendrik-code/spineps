# SPINEPS

**SPINEPS** is a framework for out-of-the-box **whole-spine segmentation of MR images**. It segments the
spine in sagittal MR images (T2w, T1w and others) using a two-phase approach to multi-class **semantic**
and **instance** segmentation, and can additionally assign anatomical **vertebra labels** via the
**VERIDAH** labeling model.

[![Paper](https://img.shields.io/badge/Paper-10.1007-blue)](https://link.springer.com/article/10.1007/s00330-024-11155-y)
[![PyPI version](https://badge.fury.io/py/spineps.svg)](https://pypi.python.org/pypi/spineps/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Pipeline process flow](https://github.com/Hendrik-code/spineps/raw/main/spineps/example/figures/pipeline_processflow.png)

## What it does

Given a sagittal MR scan, the pipeline:

1. **Semantically** segments 14 spinal structures (nine vertebra subregions, spinal cord, spinal canal,
   intervertebral discs, endplate and sacrum).
2. Derives a per-vertebra **instance** mask from the vertebra subregions.
3. Optionally assigns each instance an anatomical **vertebra label** (C1–L6, sacrum) with VERIDAH.
4. Computes **centroids** (points of interest) for each vertebra, endplate and disc.
5. Renders a **snapshot** visualizing the result.

## Quick links

- [Getting Started](getting-started.md) — installation and first run.
- [Pipeline](modules/pipeline.md) — how the two-phase pipeline is structured.
- [Processing Phases](modules/phases.md) — pre-processing, semantic, instance, labeling and post-processing.
- [Models & Labeling](modules/models.md) — model loading and the VERIDAH labeling model.
- [API Reference](api/pipeline.md) — full auto-generated API documentation.

## Quick start

```bash
# Install
pip install spineps

# Segment a single T2w sagittal scan
spineps sample -i /path/sub-test_T2w.nii.gz -model_semantic t2w -model_instance instance
```

See [Getting Started](getting-started.md) for the full installation guide (including PyTorch setup and
model weights), and the [Pipeline](modules/pipeline.md) page for calling SPINEPS from Python.

## Citation

If you use SPINEPS, please cite:

```bibtex
@article{moller_spinepsautomatic_2024,
    title   = {{SPINEPS}—automatic whole spine segmentation of T2-weighted {MR} images using a two-phase approach to multi-class semantic and instance segmentation},
    doi     = {10.1007/s00330-024-11155-y},
    journal = {European Radiology},
    author  = {Möller, Hendrik and Graf, Robert and Schmitt, Joachim and Keinert, Benjamin and Schön, Hanna and Atad, Matan and Sekuboyina, Anjany and Streckenbach, Felix and Kofler, Florian and Kroencke, Thomas and Bette, Stefanie and Willich, Stefan N. and Keil, Thomas and Niendorf, Thoralf and Pischon, Tobias and Endemann, Beate and Menze, Bjoern and Rueckert, Daniel and Kirschke, Jan S.},
    date    = {2024-10-29},
}
```

## License

SPINEPS is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
Copyright 2023 Hendrik Möller.
