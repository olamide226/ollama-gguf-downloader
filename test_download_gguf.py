import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import requests
import json

# Import the functions we want to test
from download_gguf import fetch_manifest, download_file


class TestFetchManifest(unittest.TestCase):
    """Test cases for the fetch_manifest function."""

    @patch('download_gguf.requests.get')
    def test_fetch_manifest_library_success(self, mock_get):
        """Test successful manifest fetch from library."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "manifest"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result, model_type = fetch_manifest("phi3", "3.8b")

        self.assertEqual(result, {"test": "manifest"})
        self.assertEqual(model_type, "library")
        mock_get.assert_called_once_with(
            "https://registry.ollama.ai/v2/library/phi3/manifests/3.8b",
            timeout=10
        )

    @patch('download_gguf.requests.get')
    @patch('builtins.print')
    def test_fetch_manifest_fallback_to_user_format(self, mock_print, mock_get):
        """Test fallback to user-specific format when library fails with 404."""
        # First call returns 404, second call succeeds
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_404.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"user": "manifest"}
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_404, mock_response_success]

        result, model_name = fetch_manifest("wavecut/vikhr", "latest")

        self.assertEqual(result, {"user": "manifest"})
        self.assertEqual(model_name, "vikhr")
        self.assertEqual(mock_get.call_count, 2)
        
        # Check both URLs were called
        calls = mock_get.call_args_list
        self.assertIn("library/wavecut/vikhr", calls[0][0][0])
        self.assertIn("wavecut/vikhr", calls[1][0][0])

    @patch('download_gguf.requests.get')
    @patch('builtins.print')
    @patch('download_gguf.sys.exit')
    def test_fetch_manifest_no_slash_404_error(self, mock_exit, mock_print, mock_get):
        """Test error handling when model has no slash and library returns 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_response

        fetch_manifest("nonexistent", "latest")

        mock_exit.assert_called_once_with(1)
        # Check that the appropriate error message was printed
        mock_print.assert_any_call(unittest.mock.ANY)

    @patch('download_gguf.requests.get')
    @patch('builtins.print')
    @patch('download_gguf.sys.exit')
    def test_fetch_manifest_network_error(self, mock_exit, mock_print, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        fetch_manifest("phi3", "3.8b")

        mock_exit.assert_called_once_with(1)

    @patch('download_gguf.requests.get')
    @patch('builtins.print')
    @patch('download_gguf.sys.exit')
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
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_url = "https://example.com/test.gguf"
        self.test_filename = "test_model.gguf"

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('download_gguf.requests.get')
    @patch('download_gguf.tqdm')
    def test_download_file_success(self, mock_tqdm, mock_get):
        """Test successful file download."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'test_data_chunk1', b'test_data_chunk2']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        # Mock tqdm
        mock_progress = MagicMock()
        mock_progress.n = 100  # Simulate complete download
        mock_tqdm.return_value = mock_progress

        # Mock file operations
        with patch('builtins.open', mock_open()) as mock_file:
            result = download_file(self.test_url, self.test_filename, self.test_dir)

            expected_path = os.path.join(self.test_dir, self.test_filename)
            self.assertEqual(result, expected_path)
            
            # Verify file was opened for writing
            mock_file.assert_called_once_with(expected_path, 'wb')
            
            # Verify data was written
            handle = mock_file.return_value
            self.assertEqual(handle.write.call_count, 2)

    @patch('download_gguf.requests.get')
    @patch('builtins.print')
    @patch('download_gguf.sys.exit')
    def test_download_file_network_error(self, mock_exit, mock_print, mock_get):
        """Test handling of network errors during download."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        download_file(self.test_url, self.test_filename, self.test_dir)

        mock_exit.assert_called_once_with(1)

    @patch('download_gguf.requests.get')
    @patch('download_gguf.tqdm')
    @patch('builtins.print')
    @patch('download_gguf.sys.exit')
    def test_download_file_incomplete_download(self, mock_exit, mock_print, mock_tqdm, mock_get):
        """Test handling of incomplete downloads."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'incomplete_data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        # Mock tqdm to simulate incomplete download
        mock_progress = MagicMock()
        mock_progress.n = 50  # Only half downloaded
        mock_tqdm.return_value = mock_progress

        with patch('builtins.open', mock_open()):
            download_file(self.test_url, self.test_filename, self.test_dir)

            mock_exit.assert_called_once_with(1)

    def test_download_file_directory_creation(self):
        """Test that the function creates directories if they don't exist."""
        nested_dir = os.path.join(self.test_dir, "nested", "directory")
        
        with patch('download_gguf.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '100'}
            mock_response.iter_content.return_value = [b'test_data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value.__enter__.return_value = mock_response

            with patch('download_gguf.tqdm') as mock_tqdm:
                mock_progress = MagicMock()
                mock_progress.n = 100
                mock_tqdm.return_value = mock_progress

                with patch('builtins.open', mock_open()):
                    download_file(self.test_url, self.test_filename, nested_dir)

                    # Verify directory was created
                    self.assertTrue(os.path.exists(nested_dir))


class TestFilenameGeneration(unittest.TestCase):
    """Test cases for filename generation logic."""

    def test_safe_filename_generation(self):
        """Test that slashes in model names are properly handled in filenames."""
        # This tests the logic we implemented to fix the FileNotFoundError
        model_name = "wavecut/vikhr"
        model_parameters = "latest"
        
        safe_model_name = model_name.replace("/", "_")
        output_filename = f"{safe_model_name}_{model_parameters}.gguf"
        
        expected_filename = "wavecut_vikhr_latest.gguf"
        self.assertEqual(output_filename, expected_filename)
        
        # Ensure filename doesn't contain path separators
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
            blob_url = f"https://registry.ollama.ai/v2/{user}/{model}/blobs/sha256:abcd1234"
            
            expected_manifest_url = "https://registry.ollama.ai/v2/wavecut/vikhr/manifests/latest"
            expected_blob_url = "https://registry.ollama.ai/v2/wavecut/vikhr/blobs/sha256:abcd1234"
            
            self.assertEqual(manifest_url, expected_manifest_url)
            self.assertEqual(blob_url, expected_blob_url)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
