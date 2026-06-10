"""High-level, one-call API for SPINEPS.

The pipeline functions in :mod:`spineps.seg_run` are powerful but verbose: you must load three models yourself and pass
many flat keyword arguments. This module wraps that into a single :func:`segment` call (and a reusable
:class:`SpinepsPipeline` for batches), returning a small :class:`SpinepsResult`.

Example::

    import spineps

    result = spineps.segment("sub-01_T2w.nii.gz")  # saves a BIDS derivatives folder next to the input
    result = spineps.segment(nii, output_in_memory=True)  # returns the masks in memory
    if result.success:
        seg, vert = result.semantic, result.vertebra
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from TPTBox import BIDS_FILE, NII, POI

from spineps.config import InstanceConfig, LabelingConfig, PostConfig, SemanticConfig
from spineps.get_models import get_actual_model, get_instance_model, get_labeling_model, get_semantic_model
from spineps.phase_labeling import VertLabelingClassifier
from spineps.seg_enums import ErrCode
from spineps.seg_model import SegmentationModel
from spineps.seg_run import segment_image

# Accepted forms for an input image and a model argument.
ImageInput = Union[str, Path, NII, BIDS_FILE]
ModelInput = Union[str, Path, SegmentationModel, VertLabelingClassifier]


@dataclass
class SpinepsResult:
    """Result of a single :func:`segment` call.

    Attributes:
        errcode (ErrCode): Outcome of the run (``ErrCode.OK`` / ``ErrCode.ALL_DONE`` mean success).
        semantic (NII | None): Subregion (semantic) mask, when run with ``output_in_memory=True``.
        vertebra (NII | None): Vertebra (instance) mask, when run with ``output_in_memory=True``.
        centroids (POI | None): Computed centroids, when run with ``output_in_memory=True``.
        output_paths (dict[str, Path] | None): Written output file paths, when run with ``output_in_memory=False``.
    """

    errcode: ErrCode
    semantic: NII | None = None
    vertebra: NII | None = None
    centroids: POI | None = None
    output_paths: dict[str, Path] | None = None

    @property
    def success(self) -> bool:
        """True if the run completed (freshly processed or already done)."""
        return self.errcode in (ErrCode.OK, ErrCode.ALL_DONE)


def _resolve_model(model: ModelInput | None, getter, use_cpu: bool):
    """Turns a model id / path / already-loaded model into a loaded model object (or None)."""
    if model is None or model == "none":
        return None
    if isinstance(model, (SegmentationModel, VertLabelingClassifier)):
        return model
    if "/" in str(model):  # treat anything path-like as an explicit model folder
        return get_actual_model(model, use_cpu=use_cpu).load()
    return getter(str(model), use_cpu=use_cpu).load()


@contextmanager
def _as_bids_file(image: ImageInput):
    """Yields a ``BIDS_FILE`` for the given image, writing an in-memory NII to a temporary file if needed."""
    if isinstance(image, BIDS_FILE):
        yield image
    elif isinstance(image, NII):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub-spineps_T2w.nii.gz"
            image.save(path, verbose=False)
            yield BIDS_FILE(str(path), dataset=tmp, verbose=False)
    else:
        path = Path(image).absolute()
        if not str(path).endswith(".nii.gz"):
            raise ValueError(f"image must be a .nii.gz file, got {path}")
        if not path.is_file():
            raise FileNotFoundError(f"image does not exist or is not a file, got {path}")
        yield BIDS_FILE(str(path), dataset=str(path.parent), verbose=False)


def _to_result(raw: tuple) -> SpinepsResult:
    """Normalizes the two return shapes of ``segment_image`` into a SpinepsResult."""
    if len(raw) == 4:  # in-memory mode, fully processed: (seg, vert, centroids, errcode)
        seg_nii, vert_nii, centroids, errcode = raw
        return SpinepsResult(errcode=errcode, semantic=seg_nii, vertebra=vert_nii, centroids=centroids)
    output_paths, errcode = raw  # save mode or any early exit: (output_paths, errcode)
    return SpinepsResult(errcode=errcode, output_paths=output_paths)


class SpinepsPipeline:
    """Holds loaded SPINEPS models so repeated :meth:`segment` calls don't reload them.

    Args:
        model_semantic: Semantic model id, path, or an already-loaded model. Defaults to ``"t2w"``.
        model_instance: Instance model id, path, or an already-loaded model. Defaults to ``"instance"``.
        model_labeling: Labeling model id/path/model, or ``None``/``"none"`` to skip labeling. Defaults to ``"t2w_labeling"``.
        use_cpu: If true, runs on CPU instead of GPU. Defaults to False.
    """

    def __init__(
        self,
        model_semantic: ModelInput = "t2w",
        model_instance: ModelInput = "instance",
        model_labeling: ModelInput | None = "t2w_labeling",
        *,
        use_cpu: bool = False,
    ) -> None:
        self.model_semantic = _resolve_model(model_semantic, get_semantic_model, use_cpu)
        self.model_instance = _resolve_model(model_instance, get_instance_model, use_cpu)
        self.model_labeling = _resolve_model(model_labeling, get_labeling_model, use_cpu)
        if self.model_semantic is None:
            raise ValueError("model_semantic could not be resolved")
        if self.model_instance is None:
            raise ValueError("model_instance could not be resolved")

    def segment(
        self,
        image: ImageInput,
        *,
        output_in_memory: bool = False,
        derivative_name: str = "derivatives_seg",
        override: bool = False,
        semantic: SemanticConfig | None = None,
        instance: InstanceConfig | None = None,
        labeling: LabelingConfig | None = None,
        post: PostConfig | None = None,
        verbose: bool = False,
    ) -> SpinepsResult:
        """Segments a single image with the already-loaded models.

        Args:
            image: A path to a ``.nii.gz`` file, an in-memory ``NII``, or a ``BIDS_FILE``. An in-memory ``NII`` always
                returns its results in memory.
            output_in_memory: If true, returns the masks in memory instead of writing a derivatives folder. Defaults to False.
            derivative_name: Name of the derivatives output folder (save mode only). Defaults to ``"derivatives_seg"``.
            override: If true, recomputes and overwrites existing outputs. Defaults to False.
            semantic, instance, labeling, post: Optional grouped config objects; unset groups use the pipeline defaults.
            verbose: If true, prints verbose information. Defaults to False.

        Returns:
            SpinepsResult: The outcome, carrying either in-memory masks or the written output paths.
        """
        in_memory = output_in_memory or isinstance(image, NII)
        kwargs: dict = {}
        for cfg in (semantic, instance, labeling, post):
            if cfg is not None:
                kwargs.update(cfg.to_kwargs())
        with _as_bids_file(image) as img_ref:
            raw = segment_image(
                img_ref=img_ref,
                model_semantic=self.model_semantic,
                model_instance=self.model_instance,
                model_labeling=self.model_labeling,
                derivative_name=derivative_name,
                override_semantic=override,
                override_instance=override,
                override_postpair=override,
                override_ctd=override,
                return_output_instead_of_save=in_memory,
                verbose=verbose,
                **kwargs,
            )
        return _to_result(raw)


def segment(
    image: ImageInput,
    *,
    model_semantic: ModelInput = "t2w",
    model_instance: ModelInput = "instance",
    model_labeling: ModelInput | None = "t2w_labeling",
    use_cpu: bool = False,
    output_in_memory: bool = False,
    derivative_name: str = "derivatives_seg",
    override: bool = False,
    semantic: SemanticConfig | None = None,
    instance: InstanceConfig | None = None,
    labeling: LabelingConfig | None = None,
    post: PostConfig | None = None,
    verbose: bool = False,
) -> SpinepsResult:
    """Segments a single image end to end in one call (loads the models, runs the pipeline).

    This is the easiest entry point: it resolves and loads the three models, wraps the input as needed and runs the
    full pipeline. To segment many images, build a :class:`SpinepsPipeline` once and call its :meth:`~SpinepsPipeline.segment`
    repeatedly so the models are loaded only once.

    Args:
        image: A path to a ``.nii.gz`` file, an in-memory ``NII``, or a ``BIDS_FILE``.
        model_semantic: Semantic model id, path, or loaded model. Defaults to ``"t2w"``.
        model_instance: Instance model id, path, or loaded model. Defaults to ``"instance"``.
        model_labeling: Labeling model id/path/model, or ``None``/``"none"`` to skip labeling. Defaults to ``"t2w_labeling"``.
        use_cpu: If true, runs on CPU instead of GPU. Defaults to False.
        output_in_memory: If true, returns the masks in memory instead of writing a derivatives folder. Defaults to False.
        derivative_name: Name of the derivatives output folder (save mode only). Defaults to ``"derivatives_seg"``.
        override: If true, recomputes and overwrites existing outputs. Defaults to False.
        semantic, instance, labeling, post: Optional grouped config objects (see :mod:`spineps.config`).
        verbose: If true, prints verbose information. Defaults to False.

    Returns:
        SpinepsResult: The outcome, carrying either in-memory masks or the written output paths.
    """
    pipeline = SpinepsPipeline(model_semantic, model_instance, model_labeling, use_cpu=use_cpu)
    return pipeline.segment(
        image,
        output_in_memory=output_in_memory,
        derivative_name=derivative_name,
        override=override,
        semantic=semantic,
        instance=instance,
        labeling=labeling,
        post=post,
        verbose=verbose,
    )
