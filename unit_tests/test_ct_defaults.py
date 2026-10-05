# Call 'python -m unittest' on this folder
"""`--model-semantic ct` must pick the CT instance and labeling models, not the T2w ones.

The instance and labeling networks are modality specific, but their argparse defaults were the hard-coded
T2w ids. So `spineps sample -i sub-01_ct.nii.gz --model-semantic ct` loaded the sagittal T2w instance
model and the T2w labeling classifier, the labeling model failed `check_input_model_compatibility`
against a CT input, and `segment_image` returned `ErrCode.COMPATIBILITY` without segmenting anything --
for the CT support the README advertises on its first screen.
"""

from __future__ import annotations

import argparse
import unittest
from unittest import mock

from TPTBox import No_Logger

from spineps import entrypoint
from spineps.seg_enums import Modality
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel

logger = No_Logger()


class _Model(SegmentationModel):
    """A loaded-looking semantic model with a given modality."""

    def __init__(self, modality: str, acquisition: str = "sag") -> None:
        self.logger = No_Logger()
        config = Segmentation_Inference_Config(
            logger=No_Logger(),
            modality=modality,
            acquisition=acquisition,
            log_name=f"{modality}_semantic",
            modeltype="nnunet",
            model_expected_orientation=("P", "I", "R"),
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[1.0, 1.0, 1.0],
            default_step_size=0.5,
            labels={1: 1},
        )
        super().__init__(__file__, config, default_verbose=False, default_allow_tqdm=False)

    def load(self, folds=None):  # noqa: ARG002
        self.predictor = object()
        return self

    def run(self, input_nii, verbose=False):  # noqa: ARG002
        raise AssertionError("not used")


class Test_Companion_Models(unittest.TestCase):
    def test_ct_semantic_model_pairs_with_ct_models(self):
        companions = entrypoint.companion_models(_Model("CT", "iso"))
        self.assertEqual(companions, {"instance": "ct_instance", "labeling": "ct_labeling"})

    def test_mr_semantic_models_keep_the_t2w_defaults(self):
        for modality in ("T2w", "T1w", "Vibe"):
            with self.subTest(modality=modality):
                companions = entrypoint.companion_models(_Model(modality))
                self.assertEqual(companions, {"instance": "instance", "labeling": "t2w_labeling"})

    def test_explicit_argument_always_wins(self):
        ct_model = _Model("CT", "iso")
        self.assertEqual(entrypoint.resolve_companion("instance", "instance", ct_model), "instance")
        self.assertEqual(entrypoint.resolve_companion("none", "labeling", ct_model), "none")
        self.assertEqual(entrypoint.resolve_companion("/weights/mine", "instance", ct_model), "/weights/mine")

    def test_unset_argument_resolves_from_the_model(self):
        self.assertEqual(entrypoint.resolve_companion(None, "instance", _Model("CT", "iso")), "ct_instance")
        self.assertEqual(entrypoint.resolve_companion(None, "labeling", _Model("CT", "iso")), "ct_labeling")
        self.assertEqual(entrypoint.resolve_companion(None, "instance", _Model("T2w")), "instance")
        self.assertEqual(entrypoint.resolve_companion(None, "labeling", _Model("T2w")), "t2w_labeling")


class Test_Parser_Defaults(unittest.TestCase):
    def test_companion_models_default_to_unset(self):
        # argparse must not pin a modality: the default is resolved after the semantic model is loaded.
        for sub in ("sample", "dataset"):
            with self.subTest(sub=sub):
                opt = self._subparser_defaults(sub)
                self.assertIsNone(opt.model_instance)
                self.assertIsNone(opt.model_labeling)
                self.assertEqual(opt.model_semantic, "ct")

    @staticmethod
    def _subparser_defaults(sub: str) -> argparse.Namespace:
        import contextlib
        import io
        import sys

        # Build the real CLI and parse the minimal valid invocation for the subcommand.
        argv = {
            "sample": ["spineps", "sample", "-i", "scan.nii.gz", "-ms", "ct"],
            "dataset": ["spineps", "dataset", "-i", "/data", "-ms", "ct"],
        }[sub]
        captured: dict = {}

        def capture(opt):
            captured["opt"] = opt

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(entrypoint, "run_sample", side_effect=capture),
            mock.patch.object(entrypoint, "run_dataset", side_effect=capture),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            entrypoint.entry_point()
        return captured["opt"]


class Test_Ct_Run_Loads_Ct_Companions(unittest.TestCase):
    def test_run_sample_requests_the_ct_models(self):
        ct_model = _Model("CT", "iso").load()
        with (
            mock.patch.object(entrypoint, "get_semantic_model", return_value=ct_model),
            mock.patch.object(entrypoint, "get_instance_model") as get_instance_model,
            mock.patch.object(entrypoint, "get_labeling_model") as get_labeling_model,
            mock.patch.object(entrypoint, "segment_image", return_value=({"out_spine": mock.MagicMock()}, None)),
            mock.patch.object(entrypoint, "BIDS_FILE"),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("os.path.isfile", return_value=True),
        ):
            parser = argparse.ArgumentParser()
            entrypoint.parser_arguments(parser)
            opt = parser.parse_args([])
            opt.input = "/tmp/sub-01_ct.nii.gz"
            opt.model_semantic, opt.model_instance, opt.model_labeling = "ct", None, None
            opt.tta = None
            entrypoint.run_sample(opt)

        self.assertEqual(get_instance_model.call_args[0][0], "ct_instance")
        self.assertEqual(get_labeling_model.call_args[0][0], "ct_labeling")


class Test_Snapshot_Mode(unittest.TestCase):
    def test_ct_is_detected_from_the_model_not_only_the_file_name(self):
        # The snapshot used MR windowing whenever the file was not named "*_ct.nii.gz".
        ct_model = _Model("CT", "iso")
        self.assertIn(Modality.CT, ct_model.modalities())
        mr_model = _Model("T2w")
        self.assertNotIn(Modality.CT, mr_model.modalities())


if __name__ == "__main__":
    unittest.main()
