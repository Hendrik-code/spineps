import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import (
    BIDS_Global_info,
    BIDS_FILE,
    No_Logger,
    NII,
    POI,
)
import time
from spineps.seg_run import process_img_nii, ErrCode
from spineps.models import get_semantic_model, get_instance_model
import numpy as np

# INPUT
in_ds = Path("DATASET_PATH")
raw = "rawdata"  # TODO
der = "derivatives"  # TODO

head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)


block = ""  # put i.e. 101 in here for block
parent_raw = str(Path(raw).joinpath(str(block)))
parent_der = str(Path(der).joinpath(str(block)))

model_semantic = get_semantic_model("Model_name")
model_instance = get_instance_model("Model_name")

bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

execution_times = []


def injection_function(seg_nii: NII):
    # do something with semantic mask
    return seg_nii


for name, subject in bids_ds.enumerate_subjects(sort=True):
    logger = head_logger.add_sub_logger(name=name)
    q = subject.new_query()
    q.flatten()
    q.filter("part", "inphase", required=False)
    q.filter("chunk", "LWS")
    q.unflatten()
    q.filter_format("T2w")
    q.filter_filetype("nii.gz")
    families = q.loop_dict(sort=True, key_addendum=["part"])
    for f in families:
        fid = f.family_id

        # TODO if you have a list of ids, check here if this family fits

        # TODO adapt this to your needs
        if "T2w_part-inphase" not in f:
            logger.print(f"{fid}: T2w_part-inphase not found, skip")
            if "T2w" not in f:
                logger.print(f"{fid}: T2w without part- not found, skip")
                continue

        start_time = time.perf_counter()
        ref: BIDS_FILE = f["T2w_part-inphase"][0] if "T2w_part-inphase" in f else f["T2w"][0]
        # Call to the pipeline
        output_paths, errcode = process_img_nii(
            img_ref=ref,
            derivative_name=der,
            model_semantic=model_semantic,
            model_instance=model_instance,
            override_semantic=False,
            override_instance=False,
            lambda_semantic=injection_function,
            save_debug_data=False,
            verbose=False,
        )
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.print(f"Inference time is: {execution_time}")
        execution_times.append(execution_time)

        if errcode not in [ErrCode.OK, ErrCode.ALL_DONE]:
            logger.print(f"{fid}: Pipeline threw errorcode {errcode}")
            # TODO continue? assert?

        # Load Outputs
        img_nii = ref.open_nii()
        seg_nii = NII.load(output_paths["out_spine"], seg=True)  # semantic mask
        vert_nii = NII.load(output_paths["out_vert"], seg=True)  # instance mask
        ctd = POI.load(output_paths["out_ctd"])  # centroid file

        # TODO do something with outputs, potentially saving them again to the output paths

if len(execution_times) > 0:
    head_logger.print(
        f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
    )
