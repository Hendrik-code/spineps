[![PyPI version spineps](https://badge.fury.io/py/spineps.svg)](https://pypi.python.org/pypi/spineps/)
[![tests](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml)

# SPINEPS – Automatic Whole Spine Segmentation of T2w MR images using a Two-Phase Approach to Multi-class Semantic and Instance Segmentation.

This is a segmentation pipeline to automatically, and robustly, segment the whole spine in T2w sagittal images.

![workflow_figure](https://github.com/Hendrik-code/spineps/tree/main/spineps/example/figures/pipeline_processflow.png?raw=true)

## Citation

If you are using SPINEPS for scientific research, please cite the following:

`arXiv Citation`

## Installation (Ubuntu)

This installation assumes you know your way around conda and virtual environments.

### Setup Venv

The order of the following instructions is important!

1. Use Conda or Pip to create a venv for python 3.11, we are using conda for this example:
```bash
conda create --name spineps python=3.11
conda activate spineps
conda install pip
```
2. Go to <a href="https://pytorch.org/get-started/locally/">https://pytorch.org/get-started/locally/</a> and install a correct pytorch version for your machine in your venv
3. Confirm that your pytorch package is working! Try calling these commands:
```bash
nvidia-smi
```
This should show your GPU and it's usage.
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This should throw no errors and return True

4. Then install all packages according to the `requirements.txt`:
```bash
pip install -r requirements.txt
```
5. Install our BIDS toolbox: `cd` into the BIDS folder (the one with the `setup.py` in it) and call 
```bash
pip install -e .
```
6. Install NNUnetV2 version 2.2:
```bash
pip install nnunetv2==2.2
```

### Setup this package
If you want to install this as package, then `cd` into the `mri_segmentor` folder and install it by running `pip install -e .` or using the `setup.py` inside of the project folder.

1. Download the model weights from https://syncandshare.lrz.de/getlink/fi16bYYmqpwPQZRGd1M4G6/
2. Extract the downloaded modelweights folders into a folder of your choice (the "mri_segmentor/models" folders will be used as default), from now on referred to as your models folder.
The models folder should only contain the individual model folders (with weights), and in those, the `inference_config.json`
3. You need to specify this models folder as argument when running. If you want to set it permanently, set the according environment variable in your `.bashrc` or `.zshrc` (whatever you are using).
```bash
export spineps_segmentor_models=<PATH-to-your-folder>
```
You can also execute the above line whenever you run this segmentation pipeline.

To check that you set the environment variable correctly, call:
```bash
echo ${spineps_segmentor_models}
```

For Windows, this might help: https://phoenixnap.com/kb/windows-set-environment-variable

If you **don't** set the environment variable, the pipeline will look into `spineps/models/` by default.


## Usage

### Installed as package:

1. Activate your venv
2. Run `mriseg -h` to see the arguments

### Installed as local clone:

1. Activate your venv
2. Run `python entrypoint.py -h` to see the arguments.
3. For example, for a sample, run `python entrypoint.py sample -i <path-to-nifty> -model_semantic <model_name> -model_instance <model_name>`
(replacing <model_name> with the name of the model you want to use)

### Issues

- import issues: try installing via the requirements again, somethings it doesn't install everything
- pytorch / cuda issues: good luck! :3


## SPINEPS Capabilities

The pipeline can process either:
- Single Nifty (.nii.gz) files
- Whole Datasets

### Single nifty

`spineps sample <args>`:

Processes a single nifty file, will create a derivatves folder next to the nifty, and write all outputs into that folder

| argument | explanation |
| :--- | --------- |
| -i   | Absolute path to the single nifty file (.nii.gz) to be processed |
| -model_semantic , -ms  | The model used for the semantic segmentation |
| -model_instance , -mv  | The model used for the vertebra instance segmentation |
| -der_name , -dn  | Name of the derivatives folder (default: derivatives_seg) |
| -save_debug, -sd  | Saves a lot of debug data and intermediate results in a separate debug-labeled folder (default: False) |
| -save_unc_img, -sui  | Saves a uncertainty image from the subreg prediction (default: False) |
| -override_semantic, -os  | Will override existing seg-spine files (default: False) |
| -override_instance, -ov  | Will override existing seg-vert files (default: False) |
| -override_ctd, -oc  | Will override existing centroid files (default: False) |
| -verbose, -v  | Prints much more stuff, may fully clutter your terminal (default: False) |

There are a lot more arguments, do `spineps sample -h` to see them.


### Dataset

`spineps dataset <args>`:

Processes all "suitable" niftys it finds in the specified dataset folder.

A dataset folder must have the following structure:
```
dataset-folder
├── <rawdata>
    ├── subfolders (optionally, any number of them)
        ├── One or multiple target files
    ├── One or multiple target files
├── <derivatives>
    ├── The results are saved/loaded here
```

A target file in a dataset must look like the following:
```
sub-<subjectid>_*_T2w.nii.gz
```
where `*` depicts any number of key-value pairs of characters.
Some examples are:
```
sub-0001_T2w.nii.gz
sub-awesomedataset_sequ-HWS_part-inphase_T2w.nii.gz
```
Anything that follows the BIDS-nomenclature is also supported (see https://bids-specification.readthedocs.io/en/stable/)
Meaning you can have some key-value pairs (like `sub-<id>`) in the name. Those key-value pairs are always separated by `_` and combined with `-` (see second example above). Those will be used in creating the filename of the created segmentations.

It supports the same arguments as in sample mode (see table above), and additionally:
| argument | explanation |
| :--- | --------- |
| -raw_name, -rn | Sets the name of the rawdata folder of the dataset (default: "rawdata")
| -ignore_bids_filter, -ibf   | If true, will search the BIDS dataset without the strict filters. Use with care! (default: False) |
| -ignore_model_compatibility, -imc  | If true, will not stop the pipeline to use the given models on unfitting input modalities (default: False) |
| -save_log, -sl  | If true, saves the log into a separate folder in the dataset directory (default: False) |
| -save_snaps_folder, -ssf  | If true, additionally saves the snapshots in a separate folder in the dataset directory (default: False) |

For a full list of arguments, call `spineps dataset -h`


## Segmentation

The pipeline segments in multiple steps:
1. Semantically segments 14 spinal structures (9 regions for vertebrae, Spinal Cord, Spinal Canal, Intervertebral Discs, Endplate, Sacrum)
2. From the vertebra regions, segment the different vertebrae as instance mask
3. Save the first as `seg-spine` mask, the second as `seg-vert` mask
4. It can save an uncertainty image for the semantic segmentation
5. From the two segmentations, calculates centroids for each vertebrae center point, endplate, and IVD and saves that into a .json
6. From the centroid and the segmentations, makes a snapshot showcasing the result as a .png

![example_semantic](https://github.com/Hendrik-code/spineps/tree/main/spineps/example/figures/example_semantic.png?raw=true)

### Labels:

In the subregion segmentation:

| Label | Structure |
| :---: | --------- |
| 41  | Arcus_Vertebrae |
| 42  | Spinosus_Process |
| 43  | Costal_Process_Left |
| 44  | Costal_Process_Right |
| 45  | Superior_Articular_Left |
| 46  | Superior_Articular_Right |
| 47  | Inferior_Articular_Left |
| 48  | Inferior_Articular_Right |
| 49  | Vertebra_Corpus_border |
| 60  | Spinal_Cord |
| 61  | Spinal_Canal |
| 62  | Endplate |
| 100 | Vertebra_Disc |
| 26  | Sacrum |

In the vertebra instance segmentation mask, each label X in [1, 25] are the unique vertebrae, while 100+X are their corresponding IVD and 200+X their endplates.


## Using the Code

If you want to call the code snippets yourself, start by initializing your models using `seg_model.get_segmentation_model()` giving it the absolute path to your model folder.

Depending on whether you want to process a single sample or a whole dataset, go into `seg_run.py` and run either `process_img_nii()` or `process_dataset()`.




## Authorship

This pipeline was created by Hendrik Möller, M.Sc. (he/him)<br>
PhD Researcher at Department for Interventional and Diagnostic Neuroradiology

Developed within an ERC Grant at<br>
University Hospital rechts der Isar at Technical University of Munich<br>
Ismaninger Street 22, 81675 Munich

https://deep-spine.de/<br>
https://aim-lab.io/author/hendrik-moller/




## License

Copyright 2023 Hendrik Möller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.