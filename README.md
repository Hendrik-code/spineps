[![arXiv](https://img.shields.io/badge/Paper-10.1007-blue)](https://link.springer.com/article/10.1007/s00330-024-11155-y)
[![Python Versions](https://img.shields.io/pypi/pyversions/spineps)](https://pypi.org/project/spineps/)
[![PyPI version spineps](https://badge.fury.io/py/spineps.svg)](https://pypi.python.org/pypi/spineps/)
[![Stable Version](https://img.shields.io/pypi/v/spineps?label=stable)](https://pypi.python.org/pypi/spineps/)
[![codecov](https://codecov.io/gh/Hendrik-code/spineps/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/Hendrik-code/spineps)
[![tests](https://github.com/Hendrik-code/spineps/actions/workflows/tests.yml/badge.svg)](https://github.com/Hendrik-code/spineps/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# SPINEPS – Automatic Whole Spine Segmentation of T2w MR images using a Two-Phase Approach to Multi-class Semantic and Instance Segmentation.
# and
# VERIDAH: Solving Enumeration Anomaly Aware Vertebra Labeling across Imaging Sequences

This is a segmentation pipeline to automatically, and robustly, segment the whole spine in T2w sagittal images.

![pipeline_process](spineps/example/figures/pipeline_processflow.png?raw=true)

## Citation

If you are using SPINEPS, please cite the following:

```
SPINEPS:

Hendrik Möller, Robert Graf, Joachim Schmitt, Benjamin Keinert, Hanna Schön, Matan Atad,
Anjany Sekuboyina, Felix Streckenbach, Florian Kofler, Thomas Kroencke, Stefanie Bette,
Stefan N. Willich, Thomas Keil, Thoralf Niendorf, Tobias Pischon, Beate Endemann, Bjoern Menze,
Daniel Rueckert, Jan S. Kirschke. SPINEPS—automatic whole spine segmentation of
T2-weighted MR images using a two-phase approach to multi-class semantic and instance segmentation.
Eur Radiol (2024). https://doi.org/10.1007/s00330-024-11155-y

Source of the T2w/T1w Segmentation:

Robert Graf, Joachim Schmitt, Sarah Schlaeger, Hendrik Kristian Möller, Vasiliki
Sideri-Lampretsa, Anjany Sekuboyina, Sandro Manuel Krieg, Benedikt Wiestler, Bjoern
Menze, Daniel Rueckert, Jan Stefan Kirschke. Denoising diffusion-based MRI to CT image
translation enables automated spinal segmentation. Eur Radiol Exp 7, 70 (2023).
https://doi.org/10.1186/s41747-023-00385-2
```
SPINEPS:

Paper link: <a href="https://link.springer.com/article/10.1007/s00330-024-11155-y#citeas">https://link.springer.com/article/10.1007/s00330-024-11155-y#citeas</a>

Source of the T2w/T1w Segmentation:

Open Access link: <a href="https://doi.org/10.1186/s41747-023-00385-2">https://doi.org/10.1186/s41747-023-00385-2</a>

BibTeX citation:
```
@article{moller_spinepsautomatic_2024,
	title = {{SPINEPS}—automatic whole spine segmentation of T2-weighted {MR} images using a two-phase approach to multi-class semantic and instance segmentation},
	issn = {1432-1084},
	url = {https://doi.org/10.1007/s00330-024-11155-y},
	doi = {10.1007/s00330-024-11155-y},
	abstract = {Introducing {SPINEPS}, a deep learning method for semantic and instance segmentation of 14 spinal structures (ten vertebra substructures, intervertebral discs, spinal cord, spinal canal, and sacrum) in whole-body sagittal T2-weighted turbo spin echo images.},
	journaltitle = {European Radiology},
	shortjournal = {Eur Radiol},
	author = {Möller, Hendrik and Graf, Robert and Schmitt, Joachim and Keinert, Benjamin and Schön, Hanna and Atad, Matan and Sekuboyina, Anjany and Streckenbach, Felix and Kofler, Florian and Kroencke, Thomas and Bette, Stefanie and Willich, Stefan N. and Keil, Thomas and Niendorf, Thoralf and Pischon, Tobias and Endemann, Beate and Menze, Bjoern and Rueckert, Daniel and Kirschke, Jan S.},
	urldate = {2024-11-14},
	date = {2024-10-29},
	langid = {english},
	keywords = {Deep learning, Intervertebral disc, Magnetic resonance imaging, Spine, Vertebral body},
}


@article{graf2023denoising,
  title={Denoising diffusion-based MRI to CT image translation enables automated spinal segmentation},
  author={Graf, Robert and Schmitt, Joachim and Schlaeger, Sarah and M{\"o}ller, Hendrik Kristian and Sideri-Lampretsa, Vasiliki and Sekuboyina, Anjany and Krieg, Sandro Manuel and Wiestler, Benedikt and Menze, Bjoern and Rueckert, Daniel and others},
  journal={European Radiology Experimental},
  volume={7},
  number={1},
  pages={70},
  year={2023},
  publisher={Springer}
}
```

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


### Setup this package

You have to install the package to use it, even if you just want to locally use the code.
1. `cd` into the `spineps` folder and install it by running `pip install -e .` or using the `pyproject.toml` inside of the project folder.
2. If you want to use manual modelweights, download them from the corresponding release page.
3. Extract the downloaded modelweights folders into a folder of your choice (the "spineps/spineps/models" folders will be used as default), from now on referred to as your models folder.
This specified folder should have the following structure:
4. You don't need this, SPINEPS will automatically download the newest weights for you.
```
<models_folder>
├── <model_name 1>
    ├── inference_config.json
    ├── <other model-specific files and folders>
├── <model_name 2>
    ├── inference_config.json
    ├── <other model-specific files and folders>
...
```

3. You need to specify this models folder as argument when running. If you want to set it permanently, set the according environment variable in your `.bashrc` or `.zshrc` (whatever you are using).
```bash
export SPINEPS_SEGMENTOR_MODELS=<PATH-to-your-folder>
```
You can also execute the above line whenever you run this segmentation pipeline.

To check that you set the environment variable correctly, call:
```bash
echo ${SPINEPS_SEGMENTOR_MODELS}
```

For Windows, this might help: https://phoenixnap.com/kb/windows-set-environment-variable

If you **don't** set the environment variable, the pipeline will look into `spineps/spineps/models/` by default.


## Usage

### Installed as package:

1. Activate your venv
2. Run `spineps -h` to see the arguments

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

There are a lot more arguments, run `spineps sample -h` to see them.

#### Example
```bash
#T2w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility -i /path/sub-testsample_T2w.nii.gz -model_semantic t2w -model_instance instance
#T1w sagittal
spineps sample -ignore_bids_filter -ignore_inference_compatibility -i ~/path/sub-testsample_T1w.nii.gz -model_semantic t1w -model_instance instance
```


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

To that end, we are using TPTBox (see https://github.com/Hendrik-code/TPTBox)

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

![example_semantic](spineps/example/figures/example_semantic.png?raw=true)

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

## VERIDAH:

To run the vertebra labeling after segmentation, specify a -model_labeling model (similar to -model_semantic and -model_instance).

If you use VERIDAH (labeling model) in addition to the segmentation models from SPINEPS, then a labeling model will run and give each vertebrae detected by SPINEPS a vertebra label. These are

| Label | Structure |
| :---: | --------- |
| 1  | C1 |
| 2 - 7  | C2 - C7 |
| 8 - 19  | T1 - T12 |
| 28  | T13 |
| 20  | L1 |
| 21 - 25  | L2 - L6 |
| 26  | Sacrum |

The labels 100+X still correspond to the vertebra's IVD and 200+X the respective endplate. For example, the label 119 is the IVD below the T12 vertebra.

## Using the Code

If you want to call the code snippets yourself, start by initializing your models using `seg_model.get_segmentation_model()` giving it the absolute path to your model folder.

Depending on whether you want to process a single sample or a whole dataset, go into `seg_run.py` and run either `process_img_nii()` or `process_dataset()`.

If you want to perform even more detailed changes or code injections, see `process_img_nii()` as inspiration on how the underlaying functions work and behave. Treat with care!


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
