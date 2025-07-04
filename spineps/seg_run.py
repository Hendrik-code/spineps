from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from TPTBox import BIDS_FILE, NII, POI, BIDS_Global_info, Location, Log_Type, Logger
from TPTBox.spine.snapshot2D.snapshot_templates import mri_snapshot

from spineps.phase_instance import predict_instance_mask
from spineps.phase_labeling import VertLabelingClassifier
from spineps.phase_post import phase_postprocess_combined
from spineps.phase_pre import preprocess_input
from spineps.phase_semantic import predict_semantic_mask
from spineps.seg_enums import Acquisition, ErrCode, Modality
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import logger, predict_centroids_from_both
from spineps.seg_utils import Modality_Pair, check_input_model_compatibility, check_model_modality_acquisition, find_best_matching_model
from spineps.utils.citation_reminder import citation_reminder


@citation_reminder
def process_dataset(
    dataset_path: Path,
    model_instance: Segmentation_Model,
    model_semantic: list[Segmentation_Model] | Segmentation_Model | None = None,
    model_labeling: VertLabelingClassifier | None = None,
    #
    rawdata_name: str = "rawdata",
    derivative_name: str = "derivatives_seg",
    modalities: list[Modality_Pair] | Modality_Pair = [(Modality.T2w, Acquisition.sag)],  # noqa: B006
    save_debug_data: bool = False,
    # save_uncertainty_image: bool = False,
    save_modelres_mask: bool = False,
    save_softmax_logits: bool = False,
    save_log_data: bool = True,
    override_semantic: bool = False,
    override_instance: bool = False,
    override_postpair: bool = False,
    override_ctd: bool = False,
    snapshot_copy_folder: Path | None | bool = None,
    pad_size: int = 4,
    # Processings
    # Semantic
    proc_sem_crop_input: bool = True,
    proc_sem_n4_bias_correction: bool = True,
    proc_sem_remove_inferior_beyond_canal: bool = False,
    proc_sem_clean_beyond_largest_bounding_box: bool = True,
    proc_sem_clean_small_cc_artifacts: bool = True,
    # Instance
    proc_inst_corpus_clean: bool = True,
    proc_inst_clean_small_cc_artifacts: bool = True,
    proc_inst_largest_k_cc: int = 0,
    proc_inst_detect_and_solve_merged_corpi: bool = True,
    # Labeling
    proc_lab_force_no_tl_anomaly: bool = False,
    # Both
    proc_fill_3d_holes: bool = True,
    proc_assign_missing_cc: bool = True,
    proc_clean_inst_by_sem: bool = True,
    proc_vertebra_inconsistency: bool = True,
    # Misc
    ignore_model_compatibility: bool = False,
    ignore_inference_compatibility: bool = False,
    ignore_bids_filter: bool = False,
    log_inference_time: bool = True,
    verbose: bool = False,
):
    """Runs the SPINEPS framework over a whole BIDS-conform dataset

    Args:
        dataset_path (Path): Path to the dataset
        model_instance (Segmentation_Model): Model for the vertebra segmentation
        model_semantic (list[Segmentation_Model] | Segmentation_Model | None, optional): Models for the subregion segmentation. If none, will attempt to find the correct one. Defaults to None.
        rawdata_name (str, optional): Name of the rawdata folder. Defaults to "rawdata".
        derivative_name (str, optional): Name of the derivatives folder. Defaults to "derivatives_seg".
        modalities (list[Modality_Pair] | Modality_Pair, optional): List of modalities you want to segment in the dataset. Defaults to [(Modality.T2w, Acquisition.sag)].

        save_debug_data (bool, optional): If true, saves debug data. Increases space usage! Defaults to False.
        #save_uncertainty_image (bool, optional): If true, saves a uncertainty image for the semantic segmentation. Defaults to False.
        save_modelres_mask (bool, optional): If true, will additionally save the semantic mask in the resolution of the model. Defaults to False.
        save_softmax_logits (bool, optional): If true, additionally saves the softmax logits (averaged over folds) as an npz. Defaults to False.
        save_log_data (bool, optional): If true, will save the log to a file. Defaults to True.

        override_subreg (bool, optional): If true, will redo existing subregion segmentations. Defaults to False.
        override_vert (bool, optional): If true, will redo existing vertebra segmentations. Defaults to False.
        override_ctd (bool, optional): If true, will redo existing cetnroid files. Defaults to False.

        snapshot_copy_folder (Path | None | bool, optional): If given a path, will copy all created snapshots in here. Defaults to None.
        do_crop_semantic (bool, optional): _description_. Defaults to True.

        proc_n4correction (bool, optional): _description_. Defaults to True.
        proc_fillholes (bool, optional): If true, will use fill holes in postprocessing step. Defaults to True.
        proc_clean (bool, optional): If true, will use CC cleaning in postprocessing step. Defaults to True.
        proc_corpus_clean (bool, optional): _description_. Defaults to True.
        proc_cleanvert (bool, optional): If true, will use CC cleaning in vertebra postprocessing. Defaults to True.
        proc_assign_missing_cc (bool, optional): _description_. Defaults to True.
        proc_largest_cc (int, optional): _description_. Defaults to 0.

        ignore_model_compatibility (bool, optional): If true, will ignore initialization compatibility issues. Defaults to False.
        ignore_inference_compatibility (bool, optional): If true, will ignore compatibility issues between models and individual inputs. Defaults to False.
        ignore_bids_filter (bool, optional): _description_. Defaults to False.
        log_inference_time (bool, optional): If true, will log the inference time for each subject. Defaults to True.
        verbose (bool, optional): If true, will spam your terminal with info. Defaults to False.
    """
    global logger  # noqa: PLW0603
    logger.print(f"Initialize setup for dataset in {dataset_path}", Log_Type.BOLD)
    # INITIALIZATION
    if not isinstance(modalities, list):
        modalities = [modalities]
    assert len(modalities) > 0, "you must specifiy the modalities to be segmented!"

    if snapshot_copy_folder is True:
        snapshot_copy_folder = dataset_path.joinpath("snaps_seg")
    elif snapshot_copy_folder is False:
        snapshot_copy_folder = None

    if model_semantic is None:
        model_semantic = [find_best_matching_model(m, expected_resolution=None) for m in modalities]
        logger.print("Found matching models:")
        for idx, m in enumerate(model_semantic):
            logger.print("-", str(modalities[idx]), ":", str(m.modelid()))
        del idx, m
    if not isinstance(model_semantic, list):
        model_semantic = [model_semantic]

    # check models and mod, acq tuples
    compatible = True
    for idx, mp in enumerate(modalities):
        compatible = False if not check_model_modality_acquisition(model_semantic[idx], mp) else compatible
        compatible = False if model_labeling is not None and not check_model_modality_acquisition(model_labeling, mp) else compatible
    del idx, mp

    if not compatible and not ignore_model_compatibility:
        logger.print("Compatibility issues (see above), stop program", Log_Type.FAIL)

    # Activate logger
    args = locals()
    if save_log_data:
        logger = Logger(dataset_path, log_filename="segmentation_pipeline", default_verbose=True, log_arguments=args, prefix="SegPipeline")
    logger.print(f"Processing dataset in {dataset_path}", Log_Type.BOLD)

    # RUN
    bids_ds = BIDS_Global_info(datasets=[dataset_path], parents=[rawdata_name, derivative_name], verbose=False)
    n_subjects = len(bids_ds)
    logger.print(f"Found {n_subjects} Subjects in {dataset_path}, parents={bids_ds.parents}")

    processed_seen_counter = 0
    processed_alldone_counter = 0
    processed_counter = 0
    not_properly_processed: list[str] = []

    for s_idx, (name, subject) in enumerate(bids_ds.enumerate_subjects(sort=True)):
        logger.print()
        logger.print(f"Processing {s_idx + 1} / {n_subjects} subject: {name}", Log_Type.ITALICS)
        subject_scan_processed = 0
        if name == "unsorted" and not ignore_bids_filter:
            logger.print("Unsorted, will skip")
            continue
        for idx, mod_pair in enumerate(modalities):
            model = model_semantic[idx]
            allowed_format = Modality.format_keys(mod_pair[0])
            allowed_acq = Acquisition.format_keys(mod_pair[1])
            q = subject.new_query(flatten=True)
            # optional give subject list
            q.filter_filetype("nii.gz")
            q.filter_non_existence("seg", required=True)
            if not ignore_bids_filter:
                q.filter_format(allowed_format)
                q.filter_dixon_only_inphase()
                q.filter_non_existence("lesions", required=True)
                q.filter_non_existence("label", required=True)
                q.filter("acq", lambda x: x in allowed_acq, required=False)  # noqa: B023
            scans = q.loop_list(sort=True)  # TODO make it family to allow for multi-inputs
            for s in scans:
                output_paths, errcode = process_img_nii(
                    img_ref=s,
                    model_semantic=model,
                    model_instance=model_instance,
                    model_labeling=model_labeling,
                    #
                    derivative_name=derivative_name,
                    #
                    # save_uncertainty_image=save_uncertainty_image,
                    save_modelres_mask=save_modelres_mask,
                    save_softmax_logits=save_softmax_logits,
                    save_debug_data=save_debug_data,
                    override_semantic=override_semantic,
                    override_instance=override_instance,
                    override_postpair=override_postpair,
                    override_ctd=override_ctd,
                    proc_pad_size=pad_size,
                    proc_sem_crop_input=proc_sem_crop_input,
                    proc_sem_n4_bias_correction=proc_sem_n4_bias_correction,
                    proc_fill_3d_holes=proc_fill_3d_holes,
                    proc_sem_remove_inferior_beyond_canal=proc_sem_remove_inferior_beyond_canal,
                    proc_sem_clean_beyond_largest_bounding_box=proc_sem_clean_beyond_largest_bounding_box,
                    proc_sem_clean_small_cc_artifacts=proc_sem_clean_small_cc_artifacts,
                    proc_inst_detect_and_solve_merged_corpi=proc_inst_detect_and_solve_merged_corpi,
                    proc_inst_corpus_clean=proc_inst_corpus_clean,
                    proc_inst_clean_small_cc_artifacts=proc_inst_clean_small_cc_artifacts,
                    proc_assign_missing_cc=proc_assign_missing_cc,
                    proc_inst_largest_k_cc=proc_inst_largest_k_cc,
                    proc_clean_inst_by_sem=proc_clean_inst_by_sem,
                    proc_lab_force_no_tl_anomaly=proc_lab_force_no_tl_anomaly,
                    proc_vertebra_inconsistency=proc_vertebra_inconsistency,
                    snapshot_copy_folder=snapshot_copy_folder,
                    ignore_bids_filter=ignore_bids_filter,
                    return_output_instead_of_save=False,
                    ignore_compatibility_issues=ignore_inference_compatibility,
                    log_inference_time=log_inference_time,
                    verbose=verbose,
                )
                subject_scan_processed += 1
                processed_seen_counter += 1
                if errcode == ErrCode.OK:
                    processed_counter += 1
                elif errcode == ErrCode.ALL_DONE:
                    processed_alldone_counter += 1
                else:
                    not_properly_processed.append((errcode, str(s.file["nii.gz"])))
        if subject_scan_processed == 0:
            logger.print(f"Subject {s_idx + 1}: {name} had no scans to be processed")

    logger.print()
    logger.print(f"Processed {processed_seen_counter} scans with {modalities}", Log_Type.BOLD)
    (
        logger.print(f"Scans that were skipped because all derivatives were present: {processed_alldone_counter}")
        if processed_alldone_counter > 0
        else None
    )
    not_processed_ok = processed_seen_counter - processed_alldone_counter - processed_counter
    if not_processed_ok > 0:
        logger.print(f"Scans that were not properly processed: {not_processed_ok}")
        (
            logger.print("Consult the log file for more info!")
            if save_log_data
            else logger.print("Set save_log_data=True to get a detailed log. Here are the scans in question:")
        )
        logger.print(not_properly_processed)


@citation_reminder
def process_img_nii(  # noqa: C901
    img_ref: BIDS_FILE,
    model_semantic: Segmentation_Model,
    model_instance: Segmentation_Model,
    model_labeling: VertLabelingClassifier | None = None,
    derivative_name: str = "derivatives_seg",
    #
    # save_uncertainty_image: bool = False,
    save_modelres_mask: bool = False,
    save_softmax_logits: bool = False,
    save_debug_data: bool = False,
    save_raw: bool = True,
    override_semantic: bool = False,
    override_instance: bool = False,
    override_postpair: bool = False,
    override_ctd: bool = False,
    proc_pad_size: int = 4,
    proc_normalize_input: bool = True,
    # Processings
    # Semantic
    proc_sem_crop_input: bool = True,
    proc_sem_n4_bias_correction: bool = True,
    proc_sem_remove_inferior_beyond_canal: bool = False,
    proc_sem_clean_beyond_largest_bounding_box: bool = True,
    proc_sem_clean_small_cc_artifacts: bool = True,
    # Instance
    proc_inst_corpus_clean: bool = True,
    proc_inst_clean_small_cc_artifacts: bool = True,
    proc_inst_largest_k_cc: int = 0,
    proc_inst_detect_and_solve_merged_corpi: bool = True,
    vertebra_instance_labeling_offset=2,
    # Labeling
    proc_lab_force_no_tl_anomaly: bool = False,
    # Both
    proc_fill_3d_holes: bool = True,
    proc_assign_missing_cc: bool = True,
    proc_clean_inst_by_sem: bool = True,
    proc_vertebra_inconsistency: bool = True,
    # Misc
    lambda_semantic: Callable[[NII], NII] | None = None,
    snapshot_copy_folder: Path | None = None,
    ignore_bids_filter: bool = False,
    ignore_compatibility_issues: bool = False,
    log_inference_time: bool = True,
    return_output_instead_of_save: bool = False,
    crop=None,
    verbose: bool = False,
) -> tuple[dict[str, Path], ErrCode]:
    """Runs the SPINEPS framework over one nifty

    Args:
        img_ref (BIDS_FILE): input BIDS_FILE
        model_instance (Segmentation_Model): Model for the vertebra segmentation
        model_semantic (list[Segmentation_Model] | Segmentation_Model | None, optional): Models for the subregion segmentation. If none, will attempt to find the correct one. Defaults to None.
        rawdata_name (str, optional): Name of the rawdata folder. Defaults to "rawdata".
        derivative_name (str, optional): Name of the derivatives folder. Defaults to "derivatives_seg".
        modalities (list[Modality_Pair] | Modality_Pair, optional): List of modalities you want to segment in the dataset. Defaults to [(Modality.T2w, Acquisition.sag)].

        save_debug_data (bool, optional): If true, saves debug data. Increases space usage! Defaults to False.
        #save_uncertainty_image (bool, optional): If true, saves a uncertainty image for the semantic segmentation. Defaults to False.
        save_modelres_mask (bool, optional): If true, will additionally save the semantic mask in the resolution of the model. Defaults to False.
        save_softmax_logits (bool, optional): If true, additionally saves the softmax logits (averaged over folds) as an npz. Defaults to False.
        save_log_data (bool, optional): If true, will save the log to a file. Defaults to True.

        override_semantic (bool, optional): If true, will redo existing semantic segmentations. Defaults to False.
        override_instance (bool, optional): If true, will redo existing instance segmentations. Defaults to False.
        override_ctd (bool, optional): If true, will redo existing cetnroid files. Defaults to False.

        snapshot_copy_folder (Path | None | bool, optional): If given a path, will copy all created snapshots in here. Defaults to None.
        do_crop_semantic (bool, optional): _description_. Defaults to True.

        proc_n4correction (bool, optional): _description_. Defaults to True.
        proc_fillholes (bool, optional): If true, will use fill holes in postprocessing step. Defaults to True.
        proc_clean (bool, optional): If true, will use CC cleaning in postprocessing step. Defaults to True.
        proc_corpus_clean (bool, optional): _description_. Defaults to True.
        proc_cleanvert (bool, optional): If true, will use CC cleaning in vertebra postprocessing. Defaults to True.
        proc_assign_missing_cc (bool, optional): _description_. Defaults to True.
        proc_largest_cc (int, optional): _description_. Defaults to 0.

        ignore_model_compatibility (bool, optional): If true, will ignore initialization compatibility issues. Defaults to False.
        ignore_inference_compatibility (bool, optional): If true, will ignore compatibility issues between models and individual inputs. Defaults to False.
        ignore_bids_filter (bool, optional): _description_. Defaults to False.
        log_inference_time (bool, optional): If true, will log the inference time for each subject. Defaults to True.
        verbose (bool, optional): If true, will spam your terminal with info. Defaults to False.

    Returns:
        ErrCode: Error code depicting whether the operation was successful or not
    """
    arguments = locals()
    input_format = img_ref.format

    output_paths = output_paths_from_input(
        img_ref, derivative_name, snapshot_copy_folder, input_format=input_format, non_strict_mode=ignore_bids_filter
    )
    out_spine = output_paths["out_spine"]
    out_spine_raw = output_paths["out_spine_raw"]
    out_vert = output_paths["out_vert"]
    out_vert_raw = output_paths["out_vert_raw"]
    out_logits = output_paths["out_logits"]
    out_snap = output_paths["out_snap"]
    out_ctd = output_paths["out_ctd"]
    out_snap2 = output_paths["out_snap2"]
    out_raw = output_paths["out_raw"]
    out_debug = output_paths["out_debug"]

    if isinstance(snapshot_copy_folder, Path):
        snapshot_copy_folder.mkdir(parents=True, exist_ok=True)

    if (
        out_spine.exists()
        and out_vert.exists()
        and out_snap.exists()
        and out_ctd.exists()
        and not override_semantic
        and not override_instance
        and not override_postpair
        and not override_ctd
        and (snapshot_copy_folder is None or out_snap2.exists())
    ):
        logger.print(f"{out_spine.name}: Outputs are all already created and no override set, will skip")
        return output_paths, ErrCode.ALL_DONE

    done_something = False
    debug_data_run: dict[str, NII] = {}

    if Modality.CT in model_semantic.modalities():
        proc_normalize_input = False  # Never normalize input if it is an CT

    compatible = check_input_model_compatibility(img_ref, model=model_semantic)
    compatible_labeling = check_input_model_compatibility(img_ref, model=model_labeling) if model_labeling is not None else True
    if not (compatible and compatible_labeling):
        if not ignore_compatibility_issues:
            return output_paths, ErrCode.COMPATIBILITY
        else:
            logger.print("Issues are ignored, might not have expected outcome", Log_Type.WARNING)

    start_time = perf_counter()
    file_dir = img_ref.file["nii.gz"]

    logger.print("Processing", file_dir.name)
    with logger:
        if verbose:
            model_semantic.logger.default_verbose = True
        input_nii = img_ref.open_nii()
        input_nii.seg = False
        input_nii_ = input_nii.copy()
        if crop is not None:
            try:
                input_nii = input_nii.apply_crop(crop)
            except Exception:
                pass
        logger.print("Input image", input_nii.zoom, input_nii.orientation, input_nii.shape)

        # First stage
        if not out_spine_raw.exists() or override_semantic:
            input_preprocessed, errcode = preprocess_input(
                input_nii,
                pad_size=proc_pad_size,
                debug_data=debug_data_run,
                proc_crop_input=proc_sem_crop_input,
                proc_normalize_input=proc_normalize_input,
                proc_do_n4_bias_correction=proc_sem_n4_bias_correction,
                verbose=verbose,
            )
            if errcode != ErrCode.OK:
                logger.print("Got Error from preprocessing", Log_Type.FAIL)
                return output_paths, errcode
            # make subreg mask
            assert input_preprocessed is not None
            seg_nii_modelres, softmax_logits, errcode = predict_semantic_mask(
                input_preprocessed,
                model_semantic,
                debug_data=debug_data_run,
                verbose=verbose,
                proc_fill_3d_holes=proc_fill_3d_holes,
                proc_clean_small_cc_artifacts=proc_sem_clean_small_cc_artifacts,
                proc_clean_beyond_largest_bounding_box=proc_sem_clean_beyond_largest_bounding_box,
                proc_remove_inferior_beyond_canal=proc_sem_remove_inferior_beyond_canal,
            )
            if errcode != ErrCode.OK:
                return output_paths, errcode

            assert isinstance(seg_nii_modelres, NII), "subregion segmentation is not a NII!"
            logger.print("seg_nii out", seg_nii_modelres.zoom, seg_nii_modelres.orientation, seg_nii_modelres.shape, verbose=verbose)
            if seg_nii_modelres.is_empty:
                logger.print("Subregion mask is empty, skip this", Log_Type.FAIL)
                return output_paths, ErrCode.EMPTY
            logger.print("Output seg_nii", seg_nii_modelres.zoom, seg_nii_modelres.orientation, seg_nii_modelres.shape, verbose=verbose)

            # Lambda Injection
            if lambda_semantic is not None:
                seg_nii_modelres = lambda_semantic(seg_nii_modelres)
            if not return_output_instead_of_save:
                if save_raw:
                    seg_nii_modelres.save(out_spine_raw, verbose=logger)
                if save_softmax_logits and isinstance(softmax_logits, np.ndarray):
                    save_nparray(softmax_logits, out_logits)
            done_something = True
        else:
            logger.print("Subreg Mask already exists. Set -override_subreg to create it anew")
            seg_nii_modelres = NII.load(out_spine_raw, seg=True)
            print("seg_nii", seg_nii_modelres.zoom, seg_nii_modelres.orientation, seg_nii_modelres.shape)

        # Second stage
        if not out_vert_raw.exists() or override_instance:
            whole_vert_nii, errcode = predict_instance_mask(
                seg_nii_modelres.copy(),
                model_instance,
                debug_data=debug_data_run,
                verbose=verbose,
                proc_inst_fill_3d_holes=proc_fill_3d_holes,
                proc_detect_and_solve_merged_corpi=proc_inst_detect_and_solve_merged_corpi,
                proc_corpus_clean=proc_inst_corpus_clean,
                proc_inst_clean_small_cc_artifacts=proc_inst_clean_small_cc_artifacts,
                proc_inst_largest_k_cc=proc_inst_largest_k_cc,
            )
            if errcode != ErrCode.OK:
                logger.print(f"Vert Mask creation failed with errcode {errcode}", Log_Type.FAIL)
                return output_paths, errcode
            assert whole_vert_nii is not None, "whole_vert_nii is None"
            whole_vert_nii = whole_vert_nii.copy()  # .reorient(orientation, verbose=True).rescale(zms, verbose=True)
            logger.print("vert_out", whole_vert_nii.zoom, whole_vert_nii.orientation, whole_vert_nii.shape, verbose=verbose)
            if save_raw and not return_output_instead_of_save:
                whole_vert_nii.save(out_vert_raw, verbose=logger)
            done_something = True
        else:
            logger.print("Vert Mask already exists. Set -override_vert to create it anew")
            whole_vert_nii = NII.load(out_vert_raw, seg=True)

        # Cleanup Step
        if not out_spine.exists() or not out_vert.exists() or done_something or override_postpair:
            # back to input space
            #
            if not save_modelres_mask:
                seg_nii_back = seg_nii_modelres.resample_from_to(input_nii_)
                whole_vert_nii = whole_vert_nii.resample_from_to(input_nii_)
            else:
                seg_nii_back = seg_nii_modelres

            seg_nii_back.assert_affine(other=input_nii_)
            # use both seg_raw and vert_raw to clean each other, add ivd_ep ...
            seg_nii_clean, vert_nii_clean = phase_postprocess_combined(
                img_nii=input_nii_,
                seg_nii=seg_nii_back,
                vert_nii=whole_vert_nii,
                model_labeling=model_labeling,
                debug_data=debug_data_run,
                proc_lab_force_no_tl_anomaly=proc_lab_force_no_tl_anomaly,
                labeling_offset=vertebra_instance_labeling_offset - 1,
                proc_clean_inst_by_sem=proc_clean_inst_by_sem,
                proc_assign_missing_cc=proc_assign_missing_cc,
                proc_vertebra_inconsistency=proc_vertebra_inconsistency,
                verbose=verbose,
            )

            seg_nii_clean.assert_affine(shape=vert_nii_clean.shape, zoom=vert_nii_clean.zoom, orientation=vert_nii_clean.orientation)
            vert_nii_clean.assert_affine(other=input_nii_)
            # input_package.make_nii_from_this(seg_nii_clean)
            # input_package.make_nii_from_this(vert_nii_clean)
            if not return_output_instead_of_save:
                seg_nii_clean.save(out_spine, verbose=logger)
                vert_nii_clean.save(out_vert, verbose=logger)
            done_something = True
        else:
            seg_nii_clean = NII.load(out_spine, seg=True)
            vert_nii_clean = NII.load(out_vert, seg=True)

        # Centroid
        if not out_ctd.exists() or done_something or override_ctd:
            ctd = predict_centroids_from_both(
                vert_nii_clean,
                seg_nii_clean,
                models=[model_semantic, model_instance, model_labeling],  # TODO add labeling info and parameters
                parameter={l: v for l, v in arguments.items() if "proc_" in l},
            )
            ctd.resample_from_to(input_nii_).save(out_ctd, verbose=logger)
            done_something = True
        else:
            logger.print("Centroids already exists, will load instead. Set -override_ctd = True to create it anew")
            ctd = POI.load(out_ctd)

        # return_output_instead_of_save:
        if return_output_instead_of_save:
            return seg_nii_clean, vert_nii_clean, ctd, ErrCode.OK

        # save debug
        if save_debug_data:
            if debug_data_run is None:
                logger.print("Save_debug_data: no debug data found", Log_Type.WARNING)
            else:
                out_debug.parent.mkdir(parents=True, exist_ok=True)
                for k, v in debug_data_run.items():
                    v.reorient_(input_nii_.orientation).save(
                        out_debug.joinpath(k + f"_{input_format}.nii.gz"), make_parents=True, verbose=False
                    )
                logger.print(f"Saved debug data into {out_debug}/*", Log_Type.OK)

        # Snapshot
        if not out_snap.exists() or done_something:
            # make only snapshot
            if snapshot_copy_folder is not None:
                out_snap = [out_snap, out_snap2]
            ctd = ctd.extract_subregion(Location.Vertebra_Corpus)
            try:
                mri_snapshot(  # TODO update snapshot
                    img_ref,
                    vert_nii_clean,
                    ctd,
                    subreg_msk=seg_nii_clean,
                    out_path=out_snap,
                    mode="MRI" if img_ref.bids_format.lower() != "ct" else "CT",
                )
            except Exception:
                # Fall back for older TPTBox versions TODO remove later
                mri_snapshot(img_ref, vert_nii_clean, ctd, subreg_msk=seg_nii_clean, out_path=out_snap)
            logger.print(f"Snapshot saved into {out_snap}", Log_Type.SAVE)
        elif not out_snap2.exists():
            logger.print(f"Copying snapshot into {snapshot_copy_folder!s}")
            out_snap2.parent.mkdir(exist_ok=True)
            shutil.copy(out_snap, out_snap2)

    logger.print(f"Pipeline took: {perf_counter() - start_time}", Log_Type.OK, verbose=log_inference_time)
    return output_paths, ErrCode.OK


def output_paths_from_input(
    img_ref: BIDS_FILE,
    derivative_name: str,
    snapshot_copy_folder: Path | str | None,
    input_format: str,
    non_strict_mode: bool = False,
):
    out_spine = img_ref.get_changed_path(
        bids_format="msk",
        parent=derivative_name,
        info={"seg": "spine", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_vert = img_ref.get_changed_path(
        bids_format="msk",
        parent=derivative_name,
        info={"seg": "vert", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_snap = img_ref.get_changed_path(
        bids_format="snp",
        file_type="png",
        parent=derivative_name,
        info={"seg": "spine", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_ctd = img_ref.get_changed_path(
        bids_format="ctd",
        file_type="json",
        parent=derivative_name,
        info={"seg": "spine", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_snap2 = Path(snapshot_copy_folder).joinpath(out_snap.name) if snapshot_copy_folder is not None else out_snap
    out_debug = out_vert.parent.joinpath(f"debug_{input_format}")
    out_raw = out_vert.parent.joinpath(f"output_raw_{input_format}")
    out_spine_raw = img_ref.get_changed_path(
        bids_format="msk",
        parent=derivative_name,
        info={"seg": "spine-raw", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_spine_raw = out_raw.joinpath(out_spine_raw.name)
    out_vert_raw = img_ref.get_changed_path(
        bids_format="msk",
        parent=derivative_name,
        info={"seg": "vert-raw", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_vert_raw = out_raw.joinpath(out_vert_raw.name)
    out_unc = img_ref.get_changed_path(
        bids_format="uncertainty",
        parent=derivative_name,
        info={"seg": "spine", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_unc = out_raw.joinpath(out_unc.name)
    out_logits = img_ref.get_changed_path(
        file_type="npz",
        bids_format="logit",
        parent=derivative_name,
        info={"seg": "spine", "mod": img_ref.format},
        non_strict_mode=non_strict_mode,
        make_parent=False,
    )
    out_logits = out_raw.joinpath(out_logits.name)
    return {
        "out_spine": out_spine,
        "out_spine_raw": out_spine_raw,
        "out_vert": out_vert,
        "out_vert_raw": out_vert_raw,
        "out_unc": out_unc,
        "out_logits": out_logits,
        "out_snap": out_snap,
        "out_ctd": out_ctd,
        "out_snap2": out_snap2,
        "out_debug": out_debug,
        "out_raw": out_raw,
    }


def save_nparray(arr: np.ndarray, out_path: Path):
    """Saves an numpy array to the disk

    Args:
        arr (np.ndarray): numpy array to be saved
        out_path (Path): output path
    """
    np.savez_compressed(out_path, arr)
    logger.print(f"Array of shape {arr.shape} saved: {out_path}", Log_Type.SAVE)
