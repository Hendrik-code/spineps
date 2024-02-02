import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import (
    BIDS_FILE,
)
from spineps.models import get_segmentation_model
from spineps.seg_run import process_img_nii
from spineps.utils.filepaths import filepath_model
import argparse


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_parser.add_argument("-i", type=str)
    main_parser.add_argument("-ds", type=str)
    main_parser.add_argument("-der", type=str)
    main_parser.add_argument("-ms", type=str)
    main_parser.add_argument("-mv", type=str)

    opt = main_parser.parse_args()

    input = BIDS_FILE(file=opt.i, dataset=opt.ds)

    model_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/"
    ms = get_segmentation_model(in_config=filepath_model(opt.ms, model_dir=model_dir))
    mv = get_segmentation_model(in_config=filepath_model(opt.mv, model_dir=model_dir))

    process_img_nii(
        img_ref=input,
        derivative_name=opt.der,
        model_semantic=ms,
        model_instance=mv,
        override_semantic=False,
        override_instance=False,
        save_debug_data=False,
        verbose=False,
    )
