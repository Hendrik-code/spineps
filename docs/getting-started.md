# Getting Started

This guide walks you through installing SPINEPS and running your first segmentation.

## Installation (Ubuntu)

This installation assumes you are comfortable with conda and virtual environments. **The order of the
following steps matters.**

### 1. Create a virtual environment

```bash
conda create --name spineps python=3.11
conda activate spineps
conda install pip
```

### 2. Install PyTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and install a PyTorch
build that matches your machine. Then confirm the install works:

```bash
nvidia-smi                                    # should show your GPU
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

### 3. Install SPINEPS

From PyPI:

```bash
pip install spineps
```

Or, for local development, clone the repository, `cd` into it and run:

```bash
pip install -e .
```

## Model weights

SPINEPS automatically downloads the latest model weights on first use, so no manual setup is required.

If you prefer to manage weights manually, download them from the GitHub release page and extract them into
a models folder with the following structure:

```text
<models_folder>
├── <model_name 1>
│   ├── inference_config.json
│   └── <other model-specific files>
├── <model_name 2>
│   ├── inference_config.json
│   └── ...
```

Point SPINEPS at that folder via an environment variable (otherwise it defaults to `spineps/spineps/models/`):

```bash
export SPINEPS_SEGMENTOR_MODELS=<PATH-to-your-folder>
echo ${SPINEPS_SEGMENTOR_MODELS}   # verify it is set
```

## Usage

### Command line

```bash
spineps -h            # top-level help
spineps sample -h     # options for a single file
spineps dataset -h    # options for a whole dataset
```

Segment a single scan:

```bash
# T2w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility \
    -i /path/sub-testsample_T2w.nii.gz -model_semantic t2w -model_instance instance

# T1w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility \
    -i /path/sub-testsample_T1w.nii.gz -model_semantic t1w -model_instance instance
```

Process a whole [BIDS](https://bids-specification.readthedocs.io/en/stable/) dataset:

```bash
spineps dataset -i /path/to/dataset -model_semantic t2w -model_instance instance
```

### Adding vertebra labels (VERIDAH)

To assign anatomical vertebra labels after segmentation, additionally pass a labeling model:

```bash
spineps sample -i /path/sub-test_T2w.nii.gz \
    -model_semantic t2w -model_instance instance -model_labeling labeling
```

### From Python

```python
from TPTBox import BIDS_FILE
from spineps import get_semantic_model, get_instance_model, process_img_nii

semantic = get_semantic_model("t2w")
instance = get_instance_model("instance")

# img_ref is a TPTBox BIDS_FILE pointing at the input scan
img_ref = BIDS_FILE("sub-test_T2w.nii.gz", dataset="/path/to/dataset")

process_img_nii(
    img_ref,
    model_semantic=semantic,
    model_instance=instance,
    derivative_name="derivatives_seg",
)
```

See the [Pipeline](modules/pipeline.md) page for more detail on the Python entry points.

## Troubleshooting

- **Import issues**: re-run the install; sometimes not every dependency installs the first time.
- **PyTorch / CUDA issues**: make sure the PyTorch build matches your CUDA version (step 2 above).
