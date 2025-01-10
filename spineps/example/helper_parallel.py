import sys  # noqa: INP001
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import argparse  # noqa: E402

from TPTBox import BIDS_FILE  # noqa: E402

from spineps.get_models import get_instance_model, get_semantic_model  # noqa: E402
from spineps.seg_run import process_img_nii  # noqa: E402

# Example
# python /spineps/example/helper_parallel.py -i PATH/TO/IMG.nii.gz -ds DATASET-PATH -der derivatives -ms [t1w,t2w,vibe] -mv instance

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_parser.add_argument("-i", type=str)
    main_parser.add_argument("-ds", type=str)
    main_parser.add_argument("-der", default="derivatives", type=str)
    main_parser.add_argument("-ms", default="t2w", type=str)
    main_parser.add_argument("-mv", default="instance", type=str)
    main_parser.add_argument("-snap", default=None, type=str)

    opt = main_parser.parse_args()

    input_bids_file = BIDS_FILE(file=opt.i, dataset=opt.ds)

    ms = get_semantic_model(opt.ms)
    mv = get_instance_model(opt.mv)
    if opt.snap is not None:
        Path(opt.snap).mkdir(exist_ok=True, parents=True)
    process_img_nii(
        img_ref=input_bids_file,
        derivative_name=opt.der,
        model_semantic=ms,
        model_instance=mv,
        override_semantic=False,
        override_instance=False,
        save_debug_data=False,
        verbose=False,
        ignore_compatibility_issues=False,  # If true, we do not check if the file ending match like _T2w.nii.gz for T2w images
        ignore_bids_filter=False,  # If true, we do not check if BIDS compliant
        save_raw=False,  # Save output as they are produced by the model
        snapshot_copy_folder=opt.snap,
    )
