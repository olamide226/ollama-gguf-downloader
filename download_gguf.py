import argparse
import sys
import requests

from tqdm import tqdm
from colorama import Fore, Style


def fetch_manifest(model_name, model_parameters):
    """ Fetch the manifest for a model from the Ollama registry """
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

def download_file(url, filename):
    """ Download a file from a URL and save it to a local file """
    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True, ncols=75, bar_format=f"{Fore.GREEN}{{l_bar}}{{bar}}|{{n_fmt}}/{{total_fmt}}{Style.RESET_ALL}")
            with open(filename, 'wb') as file:
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

def main():
    """ Main entry point for the script """
    parser = argparse.ArgumentParser(
        description="Download a .gguf model file from Ollama's registry",
        epilog="Example: python download_gguf.py phi3 3.8b"
    )
    parser.add_argument("model_name", help="The name of the model to download (e.g., phi3)")
    parser.add_argument("model_parameters", help="The model parameters to use (e.g., 3.8b)")
    args = parser.parse_args()

    model_name = args.model_name
    model_parameters = args.model_parameters

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
    output_filename = f"{model_name}:{model_parameters}.gguf"

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Downloading {output_filename}...")
    download_file(download_url, output_filename)
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Download completed: {output_filename}")

if __name__ == "__main__":
    main()
