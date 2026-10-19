# Call 'python -m unittest' on this folder
"""First-use weight download: resolve every advertised model id, and never leave a broken folder behind.

Three defects made the automatic download unreliable:

* the ``*_semantic`` ids (``t2w_semantic``, ``ct_semantic``, ...) are offered in the model registry and
  even listed in the "Available semantic models" error message, but had no ``download_names`` entry, so
  asking for one raised ``KeyError: 't2w_semantic'`` on a machine without the weights,
* ``download_weights`` logged "Download attempt failed" and **returned normally** when the URL could not
  be opened, so the caller happily returned a path that does not exist and the failure surfaced much
  later as a confusing config-not-found from the model loader, and
* the archive was extracted straight into the final folder, which the caller reads as "the model is
  installed". A download or extraction that broke halfway therefore left a folder that was never retried
  and never worked -- the user had to know to delete it by hand.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from typing_extensions import Self

from spineps.seg_enums import SpinepsPhase
from spineps.utils import auto_download
from spineps.utils.auto_download import (
    canonical_model_key,
    download_if_missing,
    download_names,
    download_weights,
    instances,
    labeling,
    semantic,
)


class _FakeResponse:
    """Minimal stand-in for ``urllib.request.urlopen``'s context manager (only ``info()`` is used)."""

    def __init__(self, size: int = 1024) -> None:
        self._size = size

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def info(self) -> dict:
        return {"Content-Length": str(self._size)}


def _model_zip(path, nested: bool = False, with_config: bool = True) -> None:
    """Writes a minimal model archive (an inference_config.json plus one weights file) to a path or buffer."""
    prefix = "wrapper/" if nested else ""
    with zipfile.ZipFile(path, "w") as zf:
        if with_config:
            zf.writestr(prefix + "inference_config.json", json.dumps({"log_name": "test"}))
        zf.writestr(prefix + "weights.ckpt", "not really a checkpoint")


class Test_Every_Advertised_Model_Id_Resolves(unittest.TestCase):
    def test_all_registry_ids_have_a_download_name(self):
        for key in (*semantic, *instances, *labeling):
            with self.subTest(model=key):
                self.assertIn(canonical_model_key(key), download_names)

    def test_semantic_aliases_map_to_the_canonical_id(self):
        self.assertEqual(canonical_model_key("t2w_semantic"), "t2w")
        self.assertEqual(canonical_model_key("ct_semantic"), "ct")
        # the canonical ids and unrelated ids are untouched
        self.assertEqual(canonical_model_key("t2w"), "t2w")
        self.assertEqual(canonical_model_key("t2w_labeling"), "t2w_labeling")
        self.assertEqual(canonical_model_key("something_semantic"), "something_semantic")

    def test_alias_and_canonical_id_resolve_to_the_same_folder(self):
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.object(auto_download, "get_mri_segmentor_models_dir", return_value=Path(td)),
                mock.patch.object(auto_download, "download_weights") as download,
            ):
                canonical = download_if_missing("ct", semantic["ct"], SpinepsPhase.SEMANTIC)
                alias = download_if_missing("ct_semantic", semantic["ct_semantic"], SpinepsPhase.SEMANTIC)
            self.assertEqual(canonical, alias)
            # ... and at the CT release version, not the generic semantic one
            self.assertTrue(canonical.name.endswith(auto_download.current_highest_ct_version), canonical.name)
            self.assertEqual(download.call_count, 2)

    def test_unknown_id_raises_a_readable_error(self):
        with tempfile.TemporaryDirectory() as td:
            patched_dir = mock.patch.object(auto_download, "get_mri_segmentor_models_dir", return_value=Path(td))
            with patched_dir, self.assertRaises(KeyError) as cm:
                download_if_missing("not_a_model", "http://example.invalid/x.zip", SpinepsPhase.SEMANTIC)
        self.assertIn("not_a_model", str(cm.exception))


class Test_Download_Failure_Is_Loud(unittest.TestCase):
    def test_unreachable_url_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "T2w_semantic_v1.0.9"
            with self.assertRaises(RuntimeError) as cm:
                download_weights("http://localhost:1/does-not-exist.zip", out)
            self.assertIn("download the weights manually", str(cm.exception))
            self.assertFalse(out.exists(), "a failed download must not leave a model folder behind")
            self.assertEqual([p.name for p in Path(td).iterdir()], [], "staging files were left behind")

    def test_truncated_archive_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "T2w_semantic_v1.0.9"

            def _retrieve(_url, filename, reporthook=None):  # noqa: ARG001
                # a download that stopped halfway: the file exists but is not a readable zip
                buffer = io.BytesIO()
                _model_zip(buffer)
                Path(filename).write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 2])
                return filename, None

            with (
                mock.patch("urllib.request.urlopen", return_value=_FakeResponse()),
                mock.patch("urllib.request.urlretrieve", side_effect=_retrieve),
                self.assertRaises(RuntimeError) as cm,
            ):
                download_weights("http://example.invalid/t2w.zip", out)
            self.assertIn("corrupt", str(cm.exception))
            self.assertFalse(out.exists())
            self.assertEqual([p.name for p in Path(td).iterdir()], [])

    def test_archive_without_a_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "T2w_semantic_v1.0.9"

            def _retrieve(_url, filename, reporthook=None):  # noqa: ARG001
                _model_zip(Path(filename), with_config=False)
                return filename, None

            with (
                mock.patch("urllib.request.urlopen", return_value=_FakeResponse()),
                mock.patch("urllib.request.urlretrieve", side_effect=_retrieve),
                self.assertRaises(RuntimeError) as cm,
            ):
                download_weights("http://example.invalid/t2w.zip", out)
            self.assertIn("inference_config.json", str(cm.exception))
            self.assertFalse(out.exists())


class Test_Successful_Download_Layout(unittest.TestCase):
    def _download(self, nested: bool) -> Path:
        td = Path(tempfile.mkdtemp())
        out = td / "T2w_semantic_v1.0.9"

        def _retrieve(_url, filename, reporthook=None):  # noqa: ARG001
            _model_zip(Path(filename), nested=nested)
            return filename, None

        with (
            mock.patch("urllib.request.urlopen", return_value=_FakeResponse()),
            mock.patch("urllib.request.urlretrieve", side_effect=_retrieve),
        ):
            download_weights("http://example.invalid/t2w.zip", out)
        return out

    def test_flat_archive(self):
        out = self._download(nested=False)
        self.assertTrue(out.joinpath("inference_config.json").is_file())
        self.assertTrue(out.joinpath("weights.ckpt").is_file())
        self.assertEqual([p.name for p in out.parent.iterdir()], [out.name], "staging files were left behind")

    def test_nested_archive_is_unwrapped(self):
        # The release archives wrap the model in one extra folder; it must not end up as out/wrapper/...
        out = self._download(nested=True)
        self.assertTrue(out.joinpath("inference_config.json").is_file())
        self.assertTrue(out.joinpath("weights.ckpt").is_file())

    def test_existing_folder_is_not_touched(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "T2w_semantic_v1.0.9"
            out.mkdir()
            out.joinpath("inference_config.json").write_text("{}")
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("must not download")):
                download_weights("http://example.invalid/t2w.zip", out)
            self.assertEqual(out.joinpath("inference_config.json").read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
