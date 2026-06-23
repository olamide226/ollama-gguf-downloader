import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import requests

from download_gguf import (cleanup, download_chunk, download_file,
                           download_large_file, fetch_manifest, get_blob_url,
                           get_file_info, merge_parts)


class TestFetchManifest(unittest.TestCase):
    """Test cases for the fetch_manifest function."""

    @patch("download_gguf.requests.get")
    def test_fetch_manifest_library_success(self, mock_get):
        """Test successful manifest fetch from library."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "manifest"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result, model_type = fetch_manifest("phi3", "3.8b")

        self.assertEqual(result, {"test": "manifest"})
        self.assertEqual(model_type, "library")
        mock_get.assert_called_once_with(
            "https://registry.ollama.ai/v2/library/phi3/manifests/3.8b", timeout=30
        )

    @patch("download_gguf.requests.get")
    @patch("builtins.print")
    def test_fetch_manifest_fallback_to_user_format(self, mock_print, mock_get):
        """Test fallback to user-specific format when library fails with 404."""
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_404.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404"
        )

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"user": "manifest"}
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_404, mock_response_success]

        result, model_name = fetch_manifest("wavecut/vikhr", "latest")

        self.assertEqual(result, {"user": "manifest"})
        self.assertEqual(model_name, "vikhr")
        self.assertEqual(mock_get.call_count, 2)

        calls = mock_get.call_args_list
        self.assertIn("library/wavecut/vikhr", calls[0][0][0])
        self.assertIn("wavecut/vikhr", calls[1][0][0])

    @patch("download_gguf.requests.get")
    @patch("builtins.print")
    @patch("download_gguf.sys.exit")
    def test_fetch_manifest_no_slash_404_error(self, mock_exit, mock_print, mock_get):
        """Test error handling when model has no slash and library returns 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404"
        )
        mock_get.return_value = mock_response

        fetch_manifest("nonexistent", "latest")

        mock_exit.assert_called_once_with(1)
        mock_print.assert_any_call(unittest.mock.ANY)

    @patch("download_gguf.requests.get")
    @patch("builtins.print")
    @patch("download_gguf.sys.exit")
    def test_fetch_manifest_network_error(self, mock_exit, mock_print, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        fetch_manifest("phi3", "3.8b")

        mock_exit.assert_called_once_with(1)

    @patch("download_gguf.requests.get")
    @patch("builtins.print")
    @patch("download_gguf.sys.exit")
    def test_fetch_manifest_invalid_json(self, mock_exit, mock_print, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetch_manifest("phi3", "3.8b")

        mock_exit.assert_called_once_with(1)


class TestDownloadFile(unittest.TestCase):
    """Test cases for the download_file function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_url = "https://example.com/test.gguf"
        self.test_filename = "test_model.gguf"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("download_gguf.requests.get")
    @patch("download_gguf.tqdm")
    def test_download_file_success(self, mock_tqdm, mock_get):
        """Test successful file download."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [
            b"test_data_chunk1",
            b"test_data_chunk2",
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        mock_progress = MagicMock()
        mock_progress.n = 100
        mock_tqdm.return_value = mock_progress

        with patch("builtins.open", mock_open()) as mock_file:
            result = download_file(self.test_url, self.test_filename, self.test_dir)

            expected_path = os.path.join(self.test_dir, self.test_filename)
            self.assertEqual(result, expected_path)
            mock_file.assert_called_once_with(expected_path, "wb")
            handle = mock_file.return_value
            self.assertEqual(handle.write.call_count, 2)

    @patch("download_gguf.requests.get")
    @patch("builtins.print")
    @patch("download_gguf.sys.exit")
    def test_download_file_network_error(self, mock_exit, mock_print, mock_get):
        """Test handling of network errors during download."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        download_file(self.test_url, self.test_filename, self.test_dir)

        mock_exit.assert_called_once_with(1)

    @patch("download_gguf.requests.get")
    @patch("download_gguf.tqdm")
    @patch("builtins.print")
    @patch("download_gguf.sys.exit")
    def test_download_file_incomplete_download(
        self, mock_exit, mock_print, mock_tqdm, mock_get
    ):
        """Test handling of incomplete downloads."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"incomplete_data"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        mock_progress = MagicMock()
        mock_progress.n = 50
        mock_tqdm.return_value = mock_progress

        with patch("builtins.open", mock_open()):
            download_file(self.test_url, self.test_filename, self.test_dir)

            mock_exit.assert_called_once_with(1)

    def test_download_file_directory_creation(self):
        """Test that the function creates directories if they don't exist."""
        nested_dir = os.path.join(self.test_dir, "nested", "directory")

        with patch("download_gguf.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": "100"}
            mock_response.iter_content.return_value = [b"test_data"]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value.__enter__.return_value = mock_response

            with patch("download_gguf.tqdm") as mock_tqdm:
                mock_progress = MagicMock()
                mock_progress.n = 100
                mock_tqdm.return_value = mock_progress

                with patch("builtins.open", mock_open()):
                    download_file(self.test_url, self.test_filename, nested_dir)
                    self.assertTrue(os.path.exists(nested_dir))


class TestFilenameGeneration(unittest.TestCase):
    """Test cases for filename generation logic."""

    def test_safe_filename_generation(self):
        """Test that slashes in model names are properly handled in filenames."""
        model_name = "wavecut/vikhr"
        model_parameters = "latest"

        safe_model_name = model_name.replace("/", "_")
        output_filename = f"{safe_model_name}_{model_parameters}.gguf"

        expected_filename = "wavecut_vikhr_latest.gguf"
        self.assertEqual(output_filename, expected_filename)
        self.assertNotIn("/", output_filename)
        self.assertNotIn("\\", output_filename)

    def test_library_model_filename(self):
        """Test filename generation for library models."""
        model_name = "phi3"
        model_parameters = "3.8b"

        safe_model_name = model_name.replace("/", "_")
        output_filename = f"{safe_model_name}_{model_parameters}.gguf"

        expected_filename = "phi3_3.8b.gguf"
        self.assertEqual(output_filename, expected_filename)


class TestURLGeneration(unittest.TestCase):
    """Test cases for URL generation logic."""

    def test_library_url_generation(self):
        """Test URL generation for library models."""
        model_name = "phi3"
        model_parameters = "3.8b"

        manifest_url = f"https://registry.ollama.ai/v2/library/{model_name}/manifests/{model_parameters}"
        expected_url = "https://registry.ollama.ai/v2/library/phi3/manifests/3.8b"

        self.assertEqual(manifest_url, expected_url)

    def test_user_specific_url_generation(self):
        """Test URL generation for user-specific models."""
        model_name = "wavecut/vikhr"
        model_parameters = "latest"

        if "/" in model_name:
            user, model = model_name.split("/", 1)
            manifest_url = f"https://registry.ollama.ai/v2/{user}/{model}/manifests/{model_parameters}"
            blob_url = (
                f"https://registry.ollama.ai/v2/{user}/{model}/blobs/sha256:abcd1234"
            )

            expected_manifest_url = (
                "https://registry.ollama.ai/v2/wavecut/vikhr/manifests/latest"
            )
            expected_blob_url = (
                "https://registry.ollama.ai/v2/wavecut/vikhr/blobs/sha256:abcd1234"
            )

            self.assertEqual(manifest_url, expected_manifest_url)
            self.assertEqual(blob_url, expected_blob_url)


class TestGetBlobUrl(unittest.TestCase):
    """Test cases for the get_blob_url function."""

    def test_library_blob_url(self):
        url = get_blob_url("phi3", "sha256:abc123", "library")
        self.assertEqual(
            url,
            "https://registry.ollama.ai/v2/library/phi3/blobs/sha256:abc123",
        )

    def test_user_blob_url(self):
        url = get_blob_url("wavecut/vikhr", "sha256:abc123", "vikhr")
        self.assertEqual(
            url,
            "https://registry.ollama.ai/v2/wavecut/vikhr/blobs/sha256:abc123",
        )


class TestGetFileInfo(unittest.TestCase):
    """Test cases for the get_file_info function."""

    @patch("download_gguf.requests.get")
    @patch("download_gguf.requests.head")
    def test_supports_ranges(self, mock_head, mock_get):
        """Test range support detection when server returns 206."""
        mock_head_response = MagicMock()
        mock_head_response.headers = {"content-length": "1024"}
        mock_head_response.raise_for_status.return_value = None
        mock_head.return_value = mock_head_response

        mock_get_response = MagicMock()
        mock_get_response.status_code = 206
        mock_get.return_value = mock_get_response

        size, supports = get_file_info("https://example.com/file")

        self.assertEqual(size, 1024)
        self.assertTrue(supports)

    @patch("download_gguf.requests.get")
    @patch("download_gguf.requests.head")
    def test_supports_ranges_via_header(self, mock_head, mock_get):
        """Test range support via Content-Range header."""
        mock_head_response = MagicMock()
        mock_head_response.headers = {"content-length": "2048"}
        mock_head_response.raise_for_status.return_value = None
        mock_head.return_value = mock_head_response

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.headers = {"content-range": "bytes 0-0/2048"}
        mock_get.return_value = mock_get_response

        size, supports = get_file_info("https://example.com/file")

        self.assertEqual(size, 2048)
        self.assertTrue(supports)

    @patch("download_gguf.requests.get")
    @patch("download_gguf.requests.head")
    def test_no_range_support(self, mock_head, mock_get):
        """Test range support detection when server does not support ranges."""
        mock_head_response = MagicMock()
        mock_head_response.headers = {"content-length": "4096"}
        mock_head_response.raise_for_status.return_value = None
        mock_head.return_value = mock_head_response

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.headers = {}
        mock_get.return_value = mock_get_response

        size, supports = get_file_info("https://example.com/file")

        self.assertEqual(size, 4096)
        self.assertFalse(supports)


class TestDownloadChunk(unittest.TestCase):
    """Test cases for the download_chunk function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.part_file = os.path.join(self.test_dir, "part_00000")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("download_gguf.requests.get")
    def test_download_chunk_full(self, mock_get):
        """Test downloading a full chunk from scratch."""
        mock_response = MagicMock()
        mock_response.status_code = 206
        mock_response.iter_content.return_value = [b"a" * 100]
        mock_response.__enter__.return_value = mock_response
        mock_get.return_value = mock_response

        progress = MagicMock()
        download_chunk("https://example.com/file", self.part_file, 0, 99, progress)

        self.assertTrue(os.path.exists(self.part_file))
        self.assertEqual(os.path.getsize(self.part_file), 100)
        progress.update.assert_called()

    @patch("download_gguf.requests.get")
    def test_download_chunk_resume_partial(self, mock_get):
        """Test resuming a partially downloaded chunk."""
        # Write 50 bytes as existing partial chunk
        os.makedirs(os.path.dirname(self.part_file), exist_ok=True)
        with open(self.part_file, "wb") as f:
            f.write(b"x" * 50)

        mock_response = MagicMock()
        mock_response.status_code = 206
        mock_response.iter_content.return_value = [b"y" * 50]
        mock_response.__enter__.return_value = mock_response
        mock_get.return_value = mock_response

        progress = MagicMock()
        download_chunk("https://example.com/file", self.part_file, 0, 99, progress)

        # Check that the Range header was set to resume from byte 50
        call_args = mock_get.call_args
        self.assertEqual(call_args[1]["headers"]["Range"], "bytes=50-99")

        self.assertEqual(os.path.getsize(self.part_file), 100)

    @patch("download_gguf.requests.get")
    def test_download_chunk_already_complete(self, mock_get):
        """Test that an already-complete chunk is skipped."""
        os.makedirs(os.path.dirname(self.part_file), exist_ok=True)
        with open(self.part_file, "wb") as f:
            f.write(b"x" * 100)

        progress = MagicMock()
        download_chunk("https://example.com/file", self.part_file, 0, 99, progress)

        mock_get.assert_not_called()
        progress.update.assert_called_once_with(100)

    @patch("download_gguf.requests.get")
    def test_download_chunk_corrupt_existing(self, mock_get):
        """Test that a corrupt (oversized) existing chunk is replaced."""
        os.makedirs(os.path.dirname(self.part_file), exist_ok=True)
        with open(self.part_file, "wb") as f:
            f.write(b"x" * 200)  # oversized — expected is 100

        mock_response = MagicMock()
        mock_response.status_code = 206
        mock_response.iter_content.return_value = [b"a" * 100]
        mock_response.__enter__.return_value = mock_response
        mock_get.return_value = mock_response

        progress = MagicMock()
        download_chunk("https://example.com/file", self.part_file, 0, 99, progress)

        mock_get.assert_called_once()
        # Should have started from byte 0 (old corrupt file removed)
        self.assertEqual(mock_get.call_args[1]["headers"]["Range"], "bytes=0-99")
        self.assertEqual(os.path.getsize(self.part_file), 100)

    def test_download_chunk_range_rejected(self):
        """Test that a non-206 response raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.__enter__.return_value = mock_response

        with patch("download_gguf.requests.get", return_value=mock_response):
            progress = MagicMock()
            with self.assertRaises(RuntimeError) as ctx:
                download_chunk(
                    "https://example.com/file", self.part_file, 0, 99, progress
                )
            self.assertIn("ignored range request", str(ctx.exception))


class TestMergeParts(unittest.TestCase):
    """Test cases for the merge_parts function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.parts_dir = os.path.join(self.test_dir, "output.gguf.parts")
        os.makedirs(self.parts_dir)
        self.output_file = os.path.join(self.test_dir, "output.gguf")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_merge_parts_basic(self):
        """Test merging multiple chunk files into one output."""
        # Create 3 chunks
        for idx, data in enumerate([b"AAA", b"BBB", b"CCC"]):
            part_path = os.path.join(self.parts_dir, f"part_{idx:05d}")
            with open(part_path, "wb") as f:
                f.write(data)

        merge_parts(self.output_file, self.parts_dir, 3)

        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"AAABBBCCC")
        # The merging temp file should be gone
        self.assertFalse(os.path.exists(self.output_file + ".merging"))


class TestCleanup(unittest.TestCase):
    """Test cases for the cleanup function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.parts_dir = os.path.join(self.test_dir, "model.gguf.parts")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cleanup_removes_directory(self):
        os.makedirs(self.parts_dir)
        with open(os.path.join(self.parts_dir, "part_00000"), "w") as f:
            f.write("data")

        cleanup(self.parts_dir)

        self.assertFalse(os.path.exists(self.parts_dir))

    def test_cleanup_noop_on_missing(self):
        """Test that cleanup does not raise on a non-existent directory."""
        nonexistent = os.path.join(self.test_dir, "nonexistent.parts")
        cleanup(nonexistent)  # should not raise


class TestDownloadLargeFile(unittest.TestCase):
    """Test cases for the download_large_file function."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, "model.gguf")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("download_gguf.download_file")
    @patch("download_gguf.get_file_info")
    def test_falls_back_to_single_threaded_when_no_ranges(
        self, mock_get_file_info, mock_download_file
    ):
        """Test that range-unsupporting servers trigger single-threaded fallback."""
        mock_get_file_info.return_value = (1024 * 1024 * 10, False)

        download_large_file("https://example.com/model", self.output_file, workers=4)

        mock_download_file.assert_called_once()
        mock_get_file_info.assert_called_once()

    @patch("download_gguf.download_file")
    @patch("download_gguf.get_file_info")
    def test_falls_back_when_workers_is_one(
        self, mock_get_file_info, mock_download_file
    ):
        """Test that workers=1 triggers single-threaded download."""
        mock_get_file_info.return_value = (1024 * 1024 * 10, True)

        download_large_file("https://example.com/model", self.output_file, workers=1)

        mock_download_file.assert_called_once()

    @patch("download_gguf.get_file_info")
    def test_skips_when_file_already_exists(self, mock_get_file_info):
        """Test that a fully downloaded file is not re-downloaded."""
        mock_get_file_info.return_value = (100, True)

        with open(self.output_file, "wb") as f:
            f.write(b"x" * 100)

        download_large_file("https://example.com/model", self.output_file, workers=4)

        # get_file_info was called but no download occurred
        mock_get_file_info.assert_called_once()
        self.assertEqual(os.path.getsize(self.output_file), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
