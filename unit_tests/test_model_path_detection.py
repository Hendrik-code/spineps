# Call 'python -m unittest' on this folder
"""A model folder handed to --model-* must be loaded as a folder, not looked up as a built-in id.

The test used to be ``"/" in str(model)``, so on Windows every absolute model path
(``C:\\weights\\T2w_semantic``) and, everywhere, every relative ``Path("weights")`` was classified as a
built-in model id and the run died with ``KeyError: Model 'c:\\weights\\t2w_semantic' does not exist.
Available semantic models: [...]`` -- while the folder sat right there.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from spineps.get_models import looks_like_model_path


class Test_Looks_Like_Model_Path(unittest.TestCase):
    def test_built_in_ids_are_ids(self):
        for model_id in ("t2w", "t1w", "vibe", "ct", "instance", "ct_instance", "t2w_labeling", "ct_labeling"):
            self.assertFalse(looks_like_model_path(model_id), model_id)

    def test_posix_paths_are_paths(self):
        for path in ("/opt/weights/T2w_semantic", "./weights", "../weights/t2w", "weights/t2w"):
            self.assertTrue(looks_like_model_path(path), path)

    def test_windows_paths_are_paths(self):
        # The actual regression: no forward slash anywhere in the string.
        for path in (r"C:\weights\T2w_semantic", r"\\server\share\weights", r"weights\t2w"):
            self.assertTrue(looks_like_model_path(path), path)

    def test_path_objects_are_paths(self):
        self.assertTrue(looks_like_model_path(Path("/opt/weights/T2w_semantic")))
        # A relative Path has no separator in its string form, but it is still a path.
        self.assertTrue(looks_like_model_path(Path("weights")))


class Test_Api_Model_Resolution(unittest.TestCase):
    def test_relative_path_object_loads_the_folder(self):
        from spineps.api import _resolve_model

        with (
            mock.patch("spineps.api.get_actual_model") as get_actual_model,
            mock.patch("spineps.api.get_semantic_model") as get_semantic_model,
        ):
            _resolve_model(Path("weights"), get_semantic_model, use_cpu=True)
        get_actual_model.assert_called_once()
        get_semantic_model.assert_not_called()

    def test_windows_path_loads_the_folder(self):
        from spineps.api import _resolve_model

        with (
            mock.patch("spineps.api.get_actual_model") as get_actual_model,
            mock.patch("spineps.api.get_semantic_model") as get_semantic_model,
        ):
            _resolve_model(r"C:\weights\T2w_semantic", get_semantic_model, use_cpu=True)
        get_actual_model.assert_called_once()
        get_semantic_model.assert_not_called()

    def test_model_id_still_goes_through_the_getter(self):
        from spineps.api import _resolve_model

        with (
            mock.patch("spineps.api.get_actual_model") as get_actual_model,
            mock.patch("spineps.api.get_semantic_model") as get_semantic_model,
        ):
            _resolve_model("t2w", get_semantic_model, use_cpu=True)
        get_semantic_model.assert_called_once_with("t2w", use_cpu=True)
        get_actual_model.assert_not_called()


class Test_Entrypoint_Model_Resolution(unittest.TestCase):
    def _run_sample_models(self, semantic, instance, labeling):
        """Returns which loader each --model-* argument was routed to."""
        import argparse

        from spineps import entrypoint

        with (
            mock.patch.object(entrypoint, "get_actual_model") as get_actual_model,
            mock.patch.object(entrypoint, "get_semantic_model") as get_semantic_model,
            mock.patch.object(entrypoint, "get_instance_model") as get_instance_model,
            mock.patch.object(entrypoint, "get_labeling_model") as get_labeling_model,
            mock.patch.object(entrypoint, "segment_image", return_value=({"out_spine": Path("/tmp/x")}, None)),
            mock.patch.object(entrypoint, "BIDS_FILE"),
            mock.patch("pathlib.Path.absolute", return_value=Path("/tmp/sub-01_T2w.nii.gz")),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("os.path.isfile", return_value=True),
        ):
            parser = argparse.ArgumentParser()
            entrypoint.parser_arguments(parser)
            opt = parser.parse_args([])
            opt.input = "/tmp/sub-01_T2w.nii.gz"
            opt.model_semantic, opt.model_instance, opt.model_labeling = semantic, instance, labeling
            opt.tta = None
            try:
                entrypoint.run_sample(opt)
            except (TypeError, ValueError, AttributeError):
                pass  # the mocked return value is not a real result; only the loader routing matters
            return {
                "actual": get_actual_model.call_count,
                "semantic": get_semantic_model.call_count,
                "instance": get_instance_model.call_count,
                "labeling": get_labeling_model.call_count,
            }

    def test_windows_paths_route_to_get_actual_model(self):
        counts = self._run_sample_models(r"C:\w\sem", r"C:\w\inst", r"C:\w\lab")
        self.assertEqual(counts["actual"], 3)
        self.assertEqual(counts["semantic"] + counts["instance"] + counts["labeling"], 0)

    def test_ids_route_to_the_id_getters(self):
        counts = self._run_sample_models("t2w", "instance", "t2w_labeling")
        self.assertEqual(counts["actual"], 0)
        self.assertEqual(counts["semantic"], 1)
        self.assertEqual(counts["instance"], 1)
        self.assertEqual(counts["labeling"], 1)


if __name__ == "__main__":
    unittest.main()
