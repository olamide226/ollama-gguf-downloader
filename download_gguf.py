import argparse
import os
import sys
import requests

from tqdm import tqdm
from colorama import Fore, Style


def fetch_manifest(model_name, model_parameters):
    """Fetch the manifest for a model from the Ollama registry."""
    url = f"https://registry.ollama.ai/v2/library/{model_name}/manifests/{model_parameters}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to fetch manifest: {e}")
        sys.exit(1)
    except ValueError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Invalid JSON response from server.")
        sys.exit(1)


def download_file(url, filename, save_dir):
    """Download a file from a URL and save it to a specified directory."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(save_dir, filename)

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True, ncols=75, 
                     bar_format=f"{Fore.GREEN}{{l_bar}}{{bar}}|{{n_fmt}}/{{total_fmt}}{Style.RESET_ALL}")
            with open(filepath, 'wb') as file:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    file.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Download incomplete.")
                sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to download file: {e}")
        sys.exit(1)
    
    return filepath


def main():
    """Main function to process arguments and download a GGUF model."""
    parser = argparse.ArgumentParser(
        description="Download a .gguf model file from Ollama's registry",
        epilog="Example: python download_gguf.py phi3 3.8b --save-dir /path/to/directory"
    )
    parser.add_argument("model_name", help="The name of the model to download (e.g., phi3)")
    parser.add_argument("model_parameters", help="The model parameters to use (e.g., 3.8b)")
    parser.add_argument("--save-dir", default=".", help="Directory to save the downloaded file (default: current directory)")

    args = parser.parse_args()

    model_name = args.model_name
    model_parameters = args.model_parameters
    save_dir = args.save_dir

    manifest = fetch_manifest(model_name, model_parameters)

    layers = manifest.get("layers", [])
    model_digest = None

    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_digest = layer.get("digest")
            break

    if not model_digest:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Model digest not found in manifest.")
        sys.exit(1)

    download_url = f"https://registry.ollama.ai/v2/library/{model_name}/blobs/{model_digest}"
    output_filename = f"{model_name}_{model_parameters}.gguf"

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Downloading {output_filename} to {save_dir}...")
    filepath = download_file(download_url, output_filename, save_dir)
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Download completed: {filepath}")


if __name__ == "__main__":
    main()
