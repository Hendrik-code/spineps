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
from spineps.seg_run import process_dataset, process_img_nii
from spineps.utils.citation_reminder import citation_reminder

logger = No_Logger(prefix="Init")


# TODO replace with Class_to_ArgParse and then load only from config files!
def parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-der_name", "-dn", type=str, default="derivatives_seg", metavar="", help="Name of the derivatives folder")
    parser.add_argument("-save_debug", "-sd", action="store_true", help="Saves a lot of debug data and intermediate results")
    # parser.add_argument("-save_unc_img", "-sui", action="store_true", help="Saves a uncertainty image from the subreg prediction")
    parser.add_argument(
        "-save_softmax_logits", "-ssl", action="store_true", help="Saves an .npz containing the softmax logit outputs of the semantic mask"
    )
    parser.add_argument(
        "-save_modelres_mask",
        "-smrm",
        action="store_true",
        help="If true, will additionally save the semantic mask in the resolution used by the model",
    )
    parser.add_argument("-override_semantic", "-os", action="store_true", help="Will override existing seg-spine files")
    parser.add_argument(
        "-override_instance", "-oi", action="store_true", help="Will override existing seg-vert files (True if semantic mask changed)"
    )
    parser.add_argument(
        "-override_postpair",
        "-opp",
        action="store_true",
        help="Will override existing cleaned files (True if either semantic or instance mask changed)",
    )
    parser.add_argument(
        "-override_ctd", "-oc", action="store_true", help="Will override existing centroid files (True if the instance mask changed)"
    )
    parser.add_argument(
        "-ignore_inference_compatibility",
        "-iic",
        action="store_true",
        help="Does not skip input masks that do not match the models modalities",
    )
    parser.add_argument(
        "-nocrop",
        "-nc",
        action="store_true",
        help="Does not crop input before semantically segmenting. Can improve the segmentation a little but depending on size costs more computation time",
    )
    parser.add_argument(
        "-no_tltv_labeling",
        "-ntl",
        action="store_true",
        help="Enforces the labeling model to predict exactly 12 thoracic vertebrae",
    )
    # proc_lab_force_no_tl_anomaly
    parser.add_argument(
        "-non4",
        action="store_true",
        help="Does not apply n4 bias field correction",
    )
    parser.add_argument("-cpu", action="store_true", help="Use CPU instead of GPU (will take way longer)")
    parser.add_argument("-run_cprofiler", "-rcp", action="store_true", help="Runs a cprofiler over the entire action")
    parser.add_argument("-verbose", "-v", action="store_true", help="Prints much more stuff, may fully clutter your terminal")
    return parser


@citation_reminder
def entry_point():
    ###########################
    ###########################
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparsers = main_parser.add_subparsers(title="cmd", help="Possible subcommands", dest="cmd", required=True)
    parser_sample = cmdparsers.add_parser(
        "sample", help="Process a single image nifty", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_dataset = cmdparsers.add_parser(
        "dataset", help="Process a whole dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ###########################
    ###########################
    parser_sample.add_argument("-input", "-i", required=True, type=str, help="path to the input nifty file")
    parser_sample.add_argument(
        "-model_semantic",
        "-ms",
        # type=str.lower,
        default=None,
        required=True,
        # choices=modelids_semantic,
        metavar="",
        help="The model used for the semantic segmentation. You can also pass an absolute path the model folder",
    )
    parser_sample.add_argument(
        "-model_instance",
        "-mv",
        # type=str.lower,
        default="instance",
        # required=True,
        # choices=modelids_instance,
        metavar="",
        help="The model used for the vertebra instance segmentation. You can also pass an absolute path the model folder",
    )
    parser_sample.add_argument(
        "-model_labeling",
        "-ml",
        # type=str.lower,
        default="t2w_labeling",
        # required=True,
        # choices=modelids_instance,
        metavar="",
        help="The model used for the vertebra labeling classification. You can also pass an absolute path the model folder",
    )
    parser_sample = parser_arguments(parser_sample)

    ###########################
    #
    #
    parser_dataset.add_argument(
        "-directory", "-i", "-d", required=True, type=str, help="path to the input directory, preferably a BIDS dataset"
    )
    parser_dataset.add_argument("-raw_name", "-rn", type=str, default="rawdata", metavar="", help="Name of the rawdata folder")
    parser_dataset.add_argument(
        "-model_semantic",
        "-ms",
        # type=str.lower,
        default="auto",
        # choices=model_subreg_choices,
        metavar="",
        help="The model used for the subregion segmentation. You can also pass an absolute path the model folder",
    )
    parser_dataset.add_argument(
        "-model_instance",
        "-mv",
        # type=str.lower,
        default="auto",
        # choices=model_vert_choices,
        metavar="",
        help="The model used for the vertebra segmentation. You can also pass an absolute path the model folder",
    )
    parser_dataset.add_argument(
        "-model_labeling",
        "-ml",
        # type=str.lower,
        default="t2w_labeling",
        # required=True,
        # choices=modelids_instance,
        metavar="",
        help="The model used for the vertebra labeling classification. You can also pass an absolute path the model folder",
    )
    parser_dataset.add_argument(
        "-ignore_bids_filter",
        "-ibf",
        action="store_true",
        help="If true, will search the BIDS dataset without the strict filters. Use with care!",
    )
    parser_dataset.add_argument(
        "-ignore_model_compatibility",
        "-imc",
        action="store_true",
        help="If true, will not stop the pipeline to use the given models on unfitting input modalities",
    )
    parser_dataset.add_argument(
        "-save_log", "-sl", action="store_true", help="If true, saves the log into a separate folder in the dataset directory"
    )
    parser_dataset.add_argument(
        "-save_snaps_folder",
        "-ssf",
        action="store_true",
        help="If true, saves the snapshots also in a separate folder in the dataset directory",
    )
    parser_dataset = parser_arguments(parser_dataset)
    #
    ###########################
    opt = main_parser.parse_args()
    print(opt)
    print()
    # print(opt)
    if opt.cmd == "sample":
        run_sample(opt)
    elif opt.cmd == "dataset":
        run_dataset(opt)
    else:
        raise NotImplementedError("cmd", opt.cmd)


@citation_reminder
def run_sample(opt: Namespace):
    input_path = Path(opt.input).absolute()
    dataset = str(input_path.parent)
    assert os.path.exists(dataset), f"-input parent does not exist, got {dataset}"  # noqa: PTH110
    assert dataset != "", f"-input you only gave a filename, not a direction to the file, got {input_path}"
    input_path = str(input_path)
    if not input_path.endswith(".nii.gz"):
        input_path += ".nii.gz"
    assert os.path.isfile(input_path), f"-input does not exist or is not a file, got {input_path}"  # noqa: PTH113
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

    bids_sample = BIDS_FILE(input_path, dataset=dataset, verbose=True)

    kwargs = {
        "img_ref": bids_sample,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "model_labeling": model_labeling,
        "derivative_name": opt.der_name,
        #
        # "save_uncertainty_image": opt.save_unc_img,
        "save_softmax_logits": opt.save_softmax_logits,
        "save_debug_data": opt.save_debug,
        "save_modelres_mask": opt.save_modelres_mask,
        "override_semantic": opt.override_semantic,
        "override_instance": opt.override_instance,
        "override_postpair": opt.override_postpair,
        "override_ctd": opt.override_ctd,
        "proc_sem_crop_input": not opt.nocrop,
        "proc_sem_n4_bias_correction": not opt.non4,
        "proc_lab_force_no_tl_anomaly": opt.no_tltv_labeling,
        "ignore_compatibility_issues": opt.ignore_inference_compatibility,
        "verbose": opt.verbose,
    }

    start_time = perf_counter()
    if opt.run_cprofiler:
        from TPTBox.logger.log_file import format_time_short, get_time

        timestamp = format_time_short(get_time())
        cprofile_out = bids_sample.get_changed_path(
            bids_format="log",
            parent=opt.der_name,
            file_type="log",
            info={"desc": "cprofile", "mod": bids_sample.format, "ses": timestamp},
        )
        with cProfile.Profile() as pr:
            process_img_nii(**kwargs)
        pr.dump_stats(cprofile_out)
        logger.print(f"Saved cprofile log into {cprofile_out}", Log_Type.SAVE)
    else:
        process_img_nii(**kwargs)

    logger.print(f"Sample took: {perf_counter() - start_time} seconds")
    return 1


@citation_reminder
def run_dataset(opt: Namespace):
    input_dir = Path(opt.directory)
    assert input_dir.exists(), f"-input does not exist, {input_dir}"
    assert input_dir.is_dir(), f"-input is not a directory, got {input_dir}"

    # Model semantic
    if opt.model_semantic == "auto":
        model_semantic = None
    elif "/" in str(opt.model_semantic):
        model_semantic = get_actual_model(opt.model_semantic, use_cpu=opt.cpu).load()
    else:
        model_semantic = get_semantic_model(opt.model_semantic, use_cpu=opt.cpu).load()

    # Model Instance
    if opt.model_instance == "auto":
        model_instance = None
    elif "/" in str(opt.model_instance):
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

    assert model_instance is not None, "-model_vert was None"

    kwargs = {
        "dataset_path": input_dir,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "model_labeling": model_labeling,
        "rawdata_name": opt.raw_name,
        "derivative_name": opt.der_name,
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
        "proc_sem_crop_input": not opt.nocrop,
        "proc_sem_n4_bias_correction": not opt.non4,
        "proc_lab_force_no_tl_anomaly": opt.no_tltv_labeling,
        "snapshot_copy_folder": opt.save_snaps_folder,
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
