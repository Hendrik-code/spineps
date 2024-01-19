from pathlib import Path

from BIDS import BIDS_FILE, No_Logger, Log_Type
import os
from spineps.seg_run import process_dataset, process_img_nii
from spineps.models import modelid2folder_semantic, modelid2folder_instance, get_semantic_model, get_instance_model
import argparse
from argparse import Namespace
import cProfile
from time import perf_counter


logger = No_Logger()
logger.override_prefix = "Init"


def parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-der_name", "-dn", type=str, default="derivatives_seg", metavar="", help="Name of the derivatives folder")
    #
    parser.add_argument("-save_debug", "-sd", action="store_true", help="Saves a lot of debug data and intermediate results")
    parser.add_argument("-save_unc_img", "-sui", action="store_true", help="Saves a uncertainty image from the subreg prediction")
    parser.add_argument(
        "-save_softmax_logits", "-ssl", action="store_true", help="Saves an .npz containing the softmax logit outputs of the semantic mask"
    )
    parser.add_argument(
        "-save_modelres_mask",
        "-smrm",
        action="store_true",
        help="If true, will additionally save the semantic mask in the resolution used by the model",
    )
    #
    parser.add_argument("-override_semantic", "-os", action="store_true", help="Will override existing seg-spine files")
    parser.add_argument(
        "-override_instance", "-oi", action="store_true", help="Will override existing seg-vert files (True if semantic mask changed)"
    )
    parser.add_argument(
        "-override_ctd", "-oc", action="store_true", help="Will override existing centroid files (True if the instance mask changed)"
    )
    #
    parser.add_argument(
        "-ignore_inference_compatibility",
        "-iic",
        action="store_true",
        help="Does not skip input masks that do not match the models modalities",
    )
    #
    parser.add_argument(
        "-nocrop",
        "-nc",
        action="store_true",
        help="Does not crop input before semantically segmenting",
    )
    parser.add_argument(
        "-non4",
        action="store_true",
        help="Does not apply n4 bias field correction",
    )
    #
    parser.add_argument("-run_cprofiler", "-rcp", action="store_true", help="Runs a cprofiler over the entire action")
    parser.add_argument("-verbose", "-v", action="store_true", help="Prints much more stuff, may fully clutter your terminal")
    return parser


def entry_point():
    modelids_semantic = list(modelid2folder_semantic().keys())
    modelids_instance = list(modelid2folder_instance().keys())
    ###########################
    ###########################
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # main_parser.add_argument(
    #    "-model_dir",
    #    type=str,
    #    default=None,
    #    metavar="",
    #    help="Set this if you want to use the models from this folder instead of the ones specified in the environment path",
    # )
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
        type=str.lower,
        default=None,
        # required=True,
        choices=modelids_semantic,
        metavar="",
        help=f"The model used for the semantic segmentation. Choices are {modelids_semantic}",
    )
    parser_sample.add_argument(
        "-model_instance",
        "-mv",
        type=str.lower,
        default=None,
        # required=True,
        choices=modelids_instance,
        metavar="",
        help=f"The model used for the vertebra instance segmentation. Choices are {modelids_instance}",
    )
    parser_sample = parser_arguments(parser_sample)
    #

    ###########################
    #
    model_subreg_choices = ["auto"] + modelids_semantic
    model_vert_choices = ["auto"] + modelids_instance
    #
    parser_dataset.add_argument(
        "-directory", "-i", "-d", required=True, type=str, help="path to the input directory, preferably a BIDS dataset"
    )
    parser_dataset.add_argument("-raw_name", "-rn", type=str, default="rawdata", metavar="", help="Name of the rawdata folder")
    parser_dataset.add_argument(
        "-model_semantic",
        "-ms",
        type=str.lower,
        default="auto",
        choices=model_subreg_choices,
        metavar="",
        help=f"The model used for the subregion segmentation. Choices are {model_subreg_choices}",
    )
    parser_dataset.add_argument(
        "-model_instance",
        "-mv",
        type=str.lower,
        default="auto",
        choices=model_vert_choices,
        metavar="",
        help=f"The model used for the vertebra segmentation. Choices are {model_vert_choices}",
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

    # print(opt)
    if opt.cmd == "sample":
        run_sample(opt)
    elif opt.cmd == "dataset":
        run_dataset(opt)
    else:
        raise NotImplementedError("cmd", opt.cmd)


def run_sample(opt: Namespace):
    input = Path(opt.input)
    dataset = str(input.parent)
    assert os.path.exists(dataset), f"-input parent does not exist, got {dataset}"
    assert dataset != "" and dataset != ".", f"-input you only gave a filename, not a direction to the file, got {input}"
    input = str(input)
    if not input.endswith(".nii.gz"):
        input += ".nii.gz"
    assert os.path.isfile(input), f"-input does not exist or is not a file, got {input}"
    model_semantic = get_semantic_model(opt.model_semantic).load()
    model_instance = get_instance_model(opt.model_instance).load()

    bids_sample = BIDS_FILE(input, dataset=dataset, verbose=True)

    kwargs = {
        "img_ref": bids_sample,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "derivative_name": opt.der_name,
        #
        "save_uncertainty_image": opt.save_unc_img,
        "save_softmax_logits": opt.save_softmax_logits,
        "save_debug_data": opt.save_debug,
        "save_modelres_mask": opt.save_modelres_mask,
        #
        "override_semantic": opt.override_semantic,
        "override_instance": opt.override_instance,
        "override_ctd": opt.override_ctd,
        #
        "do_crop_semantic": opt.nocrop,
        "proc_n4correction": opt.non4,
        "ignore_compatibility_issues": opt.ignore_inference_compatibility,
        "verbose": opt.verbose,
    }

    start_time = perf_counter()
    if opt.run_cprofiler:
        from BIDS.logger.log_file import get_time, format_time_short

        timestamp = format_time_short(get_time())
        cprofile_out = bids_sample.get_changed_path(
            format="log",
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

    logger.print(f"mriseg sample took: {perf_counter() - start_time} seconds")
    return 1


def run_dataset(opt: Namespace):
    input = Path(opt.directory)
    assert os.path.exists(input), f"-input does not exist, {input}"
    assert os.path.isdir(input), f"-input is not a directory, got {input}"

    if opt.model_semantic == "auto":
        model_semantic = None
    else:
        model_semantic = get_semantic_model(opt.model_semantic).load()
    if opt.model_instance == "auto":
        model_instance = None
    else:
        model_instance = get_instance_model(opt.model_instance).load()

    assert model_instance is not None, "-model_vert was None"

    kwargs = {
        "dataset_path": input,
        "model_semantic": model_semantic,
        "model_instance": model_instance,
        "rawdata_name": opt.rawdata_name,
        "derivative_name": opt.der_name,
        #
        "save_uncertainty_image": opt.save_unc_img,
        "save_modelres_mask": opt.save_modelres_mask,
        "save_softmax_logits": opt.save_softmax_logits,
        "save_debug_data": opt.save_debug,
        "save_log_data": opt.save_log,
        #
        "override_semantic": opt.override_semantic,
        "override_instance": opt.override_instance,
        "override_ctd": opt.override_ctd,
        #
        "ignore_model_compatibility": opt.ignore_model_compatibility,
        "ignore_inference_compatibility": opt.ignore_inference_compatibility,
        "ignore_bids_filter": opt.ignore_bids_filter,
        #
        "do_crop_semantic": opt.nocrop,
        "proc_n4correction": opt.non4,
        "snapshot_copy_folder": opt.save_snaps_folder,
        "verbose": opt.verbose,
    }

    if opt.run_cprofiler:
        from BIDS.logger.log_file import get_time, format_time_short

        start_time = get_time()
        start_time_short = format_time_short(start_time)
        cprofile_out = input.joinpath("logs")
        cprofile_out.mkdir(parents=True, exist_ok=True)
        cprofile_out = cprofile_out.joinpath(f"mriseg_dataset_{start_time_short}_cprofiler_log.log")
        with cProfile.Profile() as pr:
            process_dataset(**kwargs)
        pr.dump_stats(cprofile_out)
        logger.print(f"Saved cprofile log into {cprofile_out}", Log_Type.SAVE)
    else:
        process_dataset(**kwargs)
    return 1


if __name__ == "__main__":
    entry_point()
