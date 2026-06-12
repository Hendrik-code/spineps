"""SPINEPS: spine MRI segmentation package exposing the high-level API, model loaders and pipeline phases."""

from spineps.api import SpinepsPipeline, SpinepsResult, segment
from spineps.config import InstanceConfig, LabelingConfig, PostConfig, SemanticConfig
from spineps.entrypoint import entry_point
from spineps.get_models import get_instance_model, get_labeling_model, get_semantic_model
from spineps.phase_instance import predict_instance_mask
from spineps.phase_labeling import perform_labeling_step
from spineps.phase_post import phase_postprocess_combined
from spineps.phase_semantic import predict_semantic_mask
from spineps.seg_model import SegmentationModel
from spineps.seg_run import process_dataset, segment_image

__all__ = [
    "InstanceConfig",
    "LabelingConfig",
    "PostConfig",
    "SegmentationModel",
    "SemanticConfig",
    "SpinepsPipeline",
    "SpinepsResult",
    "entry_point",
    "get_instance_model",
    "get_labeling_model",
    "get_semantic_model",
    "perform_labeling_step",
    "phase_postprocess_combined",
    "predict_instance_mask",
    "predict_semantic_mask",
    "process_dataset",
    "segment",
    "segment_image",
]
