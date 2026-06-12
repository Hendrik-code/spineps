"""Grouped configuration objects for the SPINEPS pipeline.

These dataclasses bundle the many flat ``proc_*`` keyword arguments of :func:`spineps.seg_run.segment_image` into a few
themed groups so the high-level :func:`spineps.segment` API stays readable. Each config maps back to the flat keyword
arguments via :meth:`to_kwargs`, so nothing about the underlying pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SemanticConfig:
    """Options for the semantic (subregion) segmentation phase."""

    crop_input: bool = True
    n4_bias_correction: bool = True
    remove_inferior_beyond_canal: bool = False
    clean_beyond_largest_bounding_box: bool = True
    clean_small_cc_artifacts: bool = True
    step_size: float | None = None

    def to_kwargs(self) -> dict:
        """Returns the corresponding flat ``segment_image`` keyword arguments."""
        return {
            "proc_sem_crop_input": self.crop_input,
            "proc_sem_n4_bias_correction": self.n4_bias_correction,
            "proc_sem_remove_inferior_beyond_canal": self.remove_inferior_beyond_canal,
            "proc_sem_clean_beyond_largest_bounding_box": self.clean_beyond_largest_bounding_box,
            "proc_sem_clean_small_cc_artifacts": self.clean_small_cc_artifacts,
            "proc_sem_step_size": self.step_size,
        }


@dataclass
class InstanceConfig:
    """Options for the vertebra (instance) segmentation phase."""

    corpus_clean: bool = True
    clean_small_cc_artifacts: bool = True
    largest_k_cc: int = 0
    detect_and_solve_merged_corpi: bool = True
    batch_size: int = 4
    amp: bool = False
    labeling_offset: int = 2

    def to_kwargs(self) -> dict:
        """Returns the corresponding flat ``segment_image`` keyword arguments."""
        return {
            "proc_inst_corpus_clean": self.corpus_clean,
            "proc_inst_clean_small_cc_artifacts": self.clean_small_cc_artifacts,
            "proc_inst_largest_k_cc": self.largest_k_cc,
            "proc_inst_detect_and_solve_merged_corpi": self.detect_and_solve_merged_corpi,
            "proc_inst_batch_size": self.batch_size,
            "proc_inst_amp": self.amp,
            "vertebra_instance_labeling_offset": self.labeling_offset,
        }


@dataclass
class LabelingConfig:
    """Options for the vertebra-labeling phase."""

    force_no_tl_anomaly: bool = False

    def to_kwargs(self) -> dict:
        """Returns the corresponding flat ``segment_image`` keyword arguments."""
        return {"proc_lab_force_no_tl_anomaly": self.force_no_tl_anomaly}


@dataclass
class PostConfig:
    """Options for the combined post-processing phase (semantic + instance reconciliation)."""

    fill_3d_holes: bool = True
    assign_missing_cc: bool = True
    assign_missing_cc_fast: bool = False
    clean_inst_by_sem: bool = True
    vertebra_inconsistency: bool = True

    def to_kwargs(self) -> dict:
        """Returns the corresponding flat ``segment_image`` keyword arguments."""
        return {
            "proc_fill_3d_holes": self.fill_3d_holes,
            "proc_assign_missing_cc": self.assign_missing_cc,
            "proc_assign_missing_cc_fast": self.assign_missing_cc_fast,
            "proc_clean_inst_by_sem": self.clean_inst_by_sem,
            "proc_vertebra_inconsistency": self.vertebra_inconsistency,
        }
