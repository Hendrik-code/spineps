"""Command-line interface for SPINEPS, wiring CLI arguments to single-image and whole-dataset processing."""

from __future__ import annotations

import argparse
import cProfile
import os
from argparse import Namespace
from pathlib import Path
from time import perf_counter

from TPTBox import BIDS_FILE, Log_Type, No_Logger

from spineps.get_models import (
    get_actual_model,
    get_instance_model,
    get_labeling_model,
    get_semantic_model,
    modelid2folder_instance,
    modelid2folder_labeling,
    modelid2folder_semantic,
)
from spineps.seg_run import process_dataset, segment_image
from spineps.utils.citation_reminder import citation_reminder

logger = No_Logger(prefix="Init")


# TODO replace with Class_to_ArgParse and then load only from config files!
def parser_arguments(parser: argparse.ArgumentParser):
    """Add the shared SPINEPS processing options to an argument parser.

    Registers flags common to both the ``sample`` and ``dataset`` subcommands, such as derivatives naming,
    override toggles, debug/saving options, cropping, n4 bias correction and device selection.

    Args:
        parser (argparse.ArgumentParser): The parser (or subparser) to add the arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with the shared arguments registered.
    """
    parser.add_argument("--derivative-name", "-dn", type=str, default="derivatives_seg", help="Name of the derivatives folder")
    parser.add_argument("--save-debug", "-sd", action="store_true", help="Saves a lot of debug data and intermediate results")
    parser.add_argument(
        "--save-softmax-logits", "-ssl", action="store_true", help="Saves an .npz containing the softmax logit outputs of the semantic mask"
    )
    parser.add_argument(
        "--save-modelres-mask",
        "-smrm",
        action="store_true",
        help="If true, will additionally save the semantic mask in the resolution used by the model",
    )
    parser.add_argument("--override-semantic", "-os", action="store_true", help="Will override existing seg-spine files")
    parser.add_argument(
        "--override-instance", "-oi", action="store_true", help="Will override existing seg-vert files (True if semantic mask changed)"
    )
    parser.add_argument(
        "--override-postpair",
        "-opp",
        action="store_true",
        help="Will override existing cleaned files (True if either semantic or instance mask changed)",
    )
    parser.add_argument(
        "--override-ctd", "-oc", action="store_true", help="Will override existing centroid files (True if the instance mask changed)"
    )
    parser.add_argument(
        "--ignore-inference-compatibility",
        "-iic",
        action="store_true",
        help="Does not skip input masks that do not match the models modalities",
    )
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="crop_input",
        help="Crop the input to the spine before semantic segmentation. Use --no-crop to disable (can slightly improve the "
        "segmentation but costs more computation time).",
    )
    parser.add_argument(
        "--n4",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="n4",
        help="Apply N4 bias field correction before semantic segmentation (MRI only). Use --no-n4 to disable (faster).",
    )
    parser.add_argument(
        "--enforce-12-thoracic",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enforce_12_thoracic",
        help="Force the labeling model to predict exactly 12 thoracic vertebrae (assume no thoracolumbar transition anomaly).",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=4,
        help="Number of vertebra cutouts run through the instance model per batched forward pass. Higher is faster but uses "
        "more GPU memory; falls back to one-by-one on out-of-memory.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run the instance model's forward pass under CUDA autocast (faster, may slightly change the output).",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Sliding-window tile step size for the semantic model (e.g. 0.7). Larger is faster but less accurate; "
        "by default uses the model's configured value.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force test-time augmentation (mirroring) on/off for the semantic model. Pass --no-tta to disable it for "
        "a speed-up; by default uses the model's configured setting.",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (will take way longer)")
    parser.add_argument("--run-cprofiler", "-rcp", action="store_true", help="Runs a cprofiler over the entire action")
    parser.add_argument("--verbose", "-v", action="store_true", help="Prints much more stuff, may fully clutter your terminal")
    return parser


@citation_reminder
def entry_point():
    """Parse command-line arguments and dispatch to the ``sample`` or ``dataset`` workflow.

    Builds the top-level parser with the ``sample`` and ``dataset`` subcommands, parses ``sys.argv`` and
    calls :func:`run_sample` or :func:`run_dataset` accordingly.

    Raises:
        NotImplementedError: If an unrecognized subcommand is supplied.
    """
    ###########################
    ###########################
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparsers = main_parser.add_subparsers(title="cmd", help="Possible subcommands", dest="cmd", required=True)
    parser_sample = cmdparsers.add_parser(
        "sample", help="Process a single NIfTI image", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_dataset = cmdparsers.add_parser(
        "dataset", help="Process a whole dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ###########################
    ###########################
    parser_sample.add_argument("--input", "-i", required=True, type=str, help="path to the input NIfTI file")
    parser_sample.add_argument(
        "--model-semantic",
        "-ms",
        default=None,
        required=True,
        help="The model used for the semantic segmentation. You can also pass an absolute path to the model folder",
    )
    parser_sample.add_argument(
        "--model-instance",
        "-mv",
        "-mi",
        default="instance",
        help="The model used for the vertebra instance segmentation. You can also pass an absolute path to the model folder",
    )
    parser_sample.add_argument(
        "--model-labeling",
        "-ml",
        default="t2w_labeling",
        help="The model used for the vertebra labeling classification. You can also pass an absolute path to the model folder",
    )
    parser_sample = parser_arguments(parser_sample)

    ###########################
    #
    #
    parser_dataset.add_argument(
        "--directory", "-i", "-d", required=True, type=str, help="path to the input directory, preferably a BIDS dataset"
    )
    parser_dataset.add_argument("--rawdata-name", "-rn", type=str, default="rawdata", help="Name of the rawdata folder")
    parser_dataset.add_argument(
        "--model-semantic",
        "-ms",
        default="t2w",
        help="The model used for the subregion segmentation. Pass 'auto' to auto-select a model by modality, or an absolute path to the model folder",
    )
    parser_dataset.add_argument(
        "--model-instance",
        "-mv",
        "-mi",
        default="instance",
        help="The model used for the vertebra segmentation. You can also pass an absolute path to the model folder",
    )
    parser_dataset.add_argument(
        "--model-labeling",
        "-ml",
        default="t2w_labeling",
        help="The model used for the vertebra labeling classification. You can also pass an absolute path to the model folder",
    )
    parser_dataset.add_argument(
        "--ignore-bids-filter",
        "-ibf",
        action="store_true",
        help="If true, will search the BIDS dataset without the strict filters. Use with care!",
    )
    parser_dataset.add_argument(
        "--ignore-model-compatibility",
        "-imc",
        action="store_true",
        help="If true, will not stop the pipeline to use the given models on unfitting input modalities",
    )
    parser_dataset.add_argument(
        "--save-log", "-sl", action="store_true", help="If true, saves the log into a separate folder in the dataset directory"
    )
    parser_dataset.add_argument(
        "--save-snaps-folder",
        "-ssf",
        action="store_true",
        help="If true, saves the snapshots also in a separate folder in the dataset directory",
    )
    parser_dataset = parser_arguments(parser_dataset)
    #
    ###########################
    opt = main_parser.parse_args()
    if opt.verbose:
        logger.print("Parsed arguments:", opt)
    if opt.cmd == "sample":
        run_sample(opt)
    elif opt.cmd == "dataset":
        run_dataset(opt)
    else:
        raise NotImplementedError("cmd", opt.cmd)


@citation_reminder
def run_sample(opt: Namespace):
    """Run the full segmentation pipeline on a single input NIfTI file.

    Loads the requested semantic, instance and (optional) labeling models, wraps the input as a
    ``BIDS_FILE`` and calls :func:`segment_image`, optionally under a cProfiler.

    Args:
        opt (Namespace): Parsed CLI arguments from the ``sample`` subcommand (input path, model ids/paths,
            override and saving flags, device and verbosity options).

    Returns:
        int: ``1`` on completion.

    Raises:
        ValueError: If only a filename was given instead of a path to the file.
        FileNotFoundError: If the input path's parent directory is missing, or the input file does not exist.
    """
    input_path = Path(opt.input).absolute()
    dataset = str(input_path.parent)
    if dataset == "":
        raise ValueError(f"-input you only gave a filename, not a path to the file, got {input_path}")
    if not os.path.exists(dataset):  # noqa: PTH110
        raise FileNotFoundError(f"-input parent directory does not exist, got {dataset}")
    input_path = str(input_path)
    if not input_path.endswith(".nii.gz"):
        input_path += ".nii.gz"
    if not os.path.isfile(input_path):  # noqa: PTH113
        raise FileNotFoundError(f"-input does not exist or is not a file, got {input_path}")
    # model semantic
    if "/" in str(opt.model_semantic):
        model_semantic = get_actual_model(opt.model_semantic, use_cpu=opt.cpu).load()
    else:
        model_semantic = get_semantic_model(opt.model_semantic, use_cpu=opt.cpu).load()
    # model instance
    if "/" in str(opt.model_instance):
        model_instance = get_actual_model(opt.model_instance, use_cpu=opt.cpu).load()
    else:
        model_instance = get_instance_model(opt.model_instance, use_cpu=opt.cpu).load()
    # model labeling
    if opt.model_labeling == "none":
        model_labeling = None
    elif "/" in str(opt.model_labeling):
        model_labeling = get_actual_model(opt.model_labeling, use_cpu=opt.cpu).load()
    else:
        model_labeling = get_labeling_model(opt.model_labeling, use_cpu=opt.cpu).load()

    if opt.tta is not None:
        model_semantic.set_test_time_augmentation(opt.tta)

    bids_sample = BIDS_FILE(input_path, dataset=dataset, verbose=True)

    kwargs = {
        "img_ref": bids_sample,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "model_labeling": model_labeling,
        "derivative_name": opt.derivative_name,
        #
        # "save_uncertainty_image": opt.save_unc_img,
        "save_softmax_logits": opt.save_softmax_logits,
        "save_debug_data": opt.save_debug,
        "save_modelres_mask": opt.save_modelres_mask,
        "override_semantic": opt.override_semantic,
        "override_instance": opt.override_instance,
        "override_postpair": opt.override_postpair,
        "override_ctd": opt.override_ctd,
        "proc_sem_crop_input": opt.crop_input,
        "proc_sem_n4_bias_correction": opt.n4,
        "proc_lab_force_no_tl_anomaly": opt.enforce_12_thoracic,
        "proc_inst_batch_size": opt.batch_size,
        "proc_inst_amp": opt.amp,
        "proc_sem_step_size": opt.step_size,
        "ignore_compatibility_issues": opt.ignore_inference_compatibility,
        "verbose": opt.verbose,
    }

    start_time = perf_counter()
    if opt.run_cprofiler:
        from TPTBox.logger.log_file import format_time_short, get_time

        timestamp = format_time_short(get_time())
        cprofile_out = bids_sample.get_changed_path(
            bids_format="log",
            parent=opt.derivative_name,
            file_type="log",
            info={"desc": "cprofile", "mod": bids_sample.format, "ses": timestamp},
        )
        with cProfile.Profile() as pr:
            segment_image(**kwargs)
        pr.dump_stats(cprofile_out)
        logger.print(f"Saved cprofile log into {cprofile_out}", Log_Type.SAVE)
    else:
        segment_image(**kwargs)

    logger.print(f"Sample took: {perf_counter() - start_time} seconds")
    return 1


@citation_reminder
def run_dataset(opt: Namespace):
    """Run the segmentation pipeline over a whole (preferably BIDS) dataset directory.

    Resolves the semantic, instance and (optional) labeling models (``"auto"`` defers model selection to the
    pipeline), then calls :func:`process_dataset`, optionally under a cProfiler.

    Args:
        opt (Namespace): Parsed CLI arguments from the ``dataset`` subcommand (dataset directory, rawdata and
            derivatives names, model ids/paths, override/compatibility/saving flags, device and verbosity options).

    Returns:
        int: ``1`` on completion.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the given path is not a directory.
        ValueError: If no instance model could be resolved.
    """
    input_dir = Path(opt.directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"-directory does not exist, got {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"-directory is not a directory, got {input_dir}")

    # Model semantic
    if opt.model_semantic == "auto":
        model_semantic = None
    elif "/" in str(opt.model_semantic):
        model_semantic = get_actual_model(opt.model_semantic, use_cpu=opt.cpu).load()
    else:
        model_semantic = get_semantic_model(opt.model_semantic, use_cpu=opt.cpu).load()

    # Model Instance
    if "/" in str(opt.model_instance):
        model_instance = get_actual_model(opt.model_instance, use_cpu=opt.cpu).load()
    else:
        model_instance = get_instance_model(opt.model_instance, use_cpu=opt.cpu).load()

    # Model Labeling
    if opt.model_labeling == "none":
        model_labeling = None
    elif "/" in str(opt.model_labeling):
        model_labeling = get_actual_model(opt.model_labeling, use_cpu=opt.cpu).load()
    else:
        model_labeling = get_labeling_model(opt.model_labeling, use_cpu=opt.cpu).load()

    if model_instance is None:
        raise ValueError("-model_instance/-mv resolved to None; pass a valid instance model id or path")

    kwargs = {
        "dataset_path": input_dir,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "model_labeling": model_labeling,
        "rawdata_name": opt.rawdata_name,
        "derivative_name": opt.derivative_name,
        #
        # "save_uncertainty_image": opt.save_unc_img,
        "save_modelres_mask": opt.save_modelres_mask,
        "save_softmax_logits": opt.save_softmax_logits,
        "save_debug_data": opt.save_debug,
        "save_log_data": opt.save_log,
        "override_semantic": opt.override_semantic,
        "override_instance": opt.override_instance,
        "override_postpair": opt.override_postpair,
        "override_ctd": opt.override_ctd,
        "ignore_model_compatibility": opt.ignore_model_compatibility,
        "ignore_inference_compatibility": opt.ignore_inference_compatibility,
        "ignore_bids_filter": opt.ignore_bids_filter,
        "proc_sem_crop_input": opt.crop_input,
        "proc_sem_n4_bias_correction": opt.n4,
        "proc_lab_force_no_tl_anomaly": opt.enforce_12_thoracic,
        "proc_inst_batch_size": opt.batch_size,
        "proc_inst_amp": opt.amp,
        "proc_sem_step_size": opt.step_size,
        "snapshot_copy_folder": opt.save_snaps_folder,
        "tta": opt.tta,
        "verbose": opt.verbose,
    }

    if opt.run_cprofiler:
        from TPTBox.logger.log_file import format_time_short, get_time

        start_time = get_time()
        start_time_short = format_time_short(start_time)
        cprofile_out = input_dir.joinpath("logs")
        cprofile_out.mkdir(parents=True, exist_ok=True)
        cprofile_out = cprofile_out.joinpath(f"spineps_dataset_{start_time_short}_cprofiler_log.log")
        with cProfile.Profile() as pr:
            process_dataset(**kwargs)
        pr.dump_stats(cprofile_out)
        logger.print(f"Saved cprofile log into {cprofile_out}", Log_Type.SAVE)
    else:
        process_dataset(**kwargs)
    return 1


if __name__ == "__main__":
    entry_point()
