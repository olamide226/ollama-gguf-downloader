#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB per chunk for parallel downloads


def fetch_manifest(model_name, model_parameters):
    """Fetch the manifest for a model from the Ollama registry.

    Returns (manifest_dict, source_type) where source_type is "library"
    or the model name component for user-specific models.
    """
    url = f"https://registry.ollama.ai/v2/library/{model_name}/manifests/{model_parameters}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json(), "library"
        if response.status_code == 404:
            if "/" in model_name:
                print(
                    f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} "
                    "Model not found in library, trying user-specific format..."
                )
                user, model = model_name.split("/", 1)
                fallback_url = (
                    f"https://registry.ollama.ai/v2/{user}/{model}"
                    f"/manifests/{model_parameters}"
                )
                fallback_response = requests.get(fallback_url, timeout=30)
                fallback_response.raise_for_status()
                return fallback_response.json(), model
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL} "
                f"Model '{model_name}:{model_parameters}' not found in library. "
                "For user-specific models, use 'username/modelname'."
            )
            sys.exit(1)
            return  # unreachable; placates mocked sys.exit in tests
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to fetch manifest: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to fetch manifest: {e}")
        sys.exit(1)
    except ValueError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Invalid JSON response from server.")
        sys.exit(1)


def get_blob_url(model_name, digest, source_type):
    """Build the blob download URL for a model."""
    if source_type == "library":
        return f"https://registry.ollama.ai/v2/library/{model_name}/blobs/{digest}"
    user, model = model_name.split("/", 1)
    return f"https://registry.ollama.ai/v2/{user}/{model}/blobs/{digest}"


def get_file_info(url):
    """Probe the server for file size and range-request support.

    Returns (size_in_bytes, supports_ranges).
    """
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    size = int(r.headers.get("content-length", 0))

    supports_ranges = False
    try:
        rr = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        supports_ranges = rr.status_code == 206 or "content-range" in rr.headers
        rr.close()
    except Exception:
        pass

    return size, supports_ranges


def download_file(url, filename, save_dir):
    """Download a file from a URL and save it to a specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            t = tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                ncols=75,
                bar_format=f"{Fore.GREEN}{{l_bar}}{{bar}}|{{n_fmt}}/{{total_fmt}}{Style.RESET_ALL}",
            )
            with open(filepath, "wb") as file:
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


def download_chunk(url, part_file, start, end, progress):
    """Download a single byte range and append to part_file. Resumes partial chunks."""
    expected_size = end - start + 1
    existing = 0

    if os.path.exists(part_file):
        existing = os.path.getsize(part_file)
        if existing == expected_size:
            progress.update(existing)
            return
        if existing > expected_size:
            os.remove(part_file)
            existing = 0

    headers = {"Range": f"bytes={start + existing}-{end}"}
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        if r.status_code != 206:
            raise RuntimeError(f"Server ignored range request (HTTP {r.status_code})")
        with open(part_file, "ab") as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))


def merge_parts(output_file, parts_dir, chunk_count):
    """Concatenate chunk files into the final output file atomically."""
    temp_output = output_file + ".merging"
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Merging chunks...")
    try:
        with open(temp_output, "wb") as out:
            for idx in range(chunk_count):
                part_file = os.path.join(parts_dir, f"part_{idx:05d}")
                with open(part_file, "rb") as inp:
                    while True:
                        data = inp.read(8 * 1024 * 1024)
                        if not data:
                            break
                        out.write(data)
        os.replace(temp_output, output_file)
    except Exception:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise


def cleanup(parts_dir):
    """Remove the temporary parts directory."""
    if os.path.exists(parts_dir):
        shutil.rmtree(parts_dir)


def download_large_file(url, output_file, workers=4):
    """Download a large file using parallel chunked range requests with resume support.

    If the server does not support range requests, falls back to single-threaded download.
    """
    total_size, supports_ranges = get_file_info(url)

    if not supports_ranges or workers <= 1:
        print(
            f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} "
            "Server does not support range requests — using single-threaded download."
        )
        save_dir = os.path.dirname(output_file) or "."
        filename = os.path.basename(output_file)
        return download_file(url, filename, save_dir)

    if os.path.exists(output_file) and os.path.getsize(output_file) == total_size:
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} File already exists.")
        return

    parts_dir = output_file + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    chunk_count = math.ceil(total_size / CHUNK_SIZE)

    # Pre-scan: count already-completed bytes for progress bar
    completed_bytes = 0
    for idx in range(chunk_count):
        start = idx * CHUNK_SIZE
        end = min(total_size - 1, start + CHUNK_SIZE - 1)
        part_file = os.path.join(parts_dir, f"part_{idx:05d}")
        if os.path.exists(part_file) and os.path.getsize(part_file) == end - start + 1:
            completed_bytes += end - start + 1

    progress = tqdm(
        total=total_size,
        initial=completed_bytes,
        unit="B",
        unit_scale=True,
        desc="Downloading",
    )

    try:
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx in range(chunk_count):
                start = idx * CHUNK_SIZE
                end = min(total_size - 1, start + CHUNK_SIZE - 1)
                part_file = os.path.join(parts_dir, f"part_{idx:05d}")
                if (
                    os.path.exists(part_file)
                    and os.path.getsize(part_file) == end - start + 1
                ):
                    continue
                futures.append(
                    executor.submit(
                        download_chunk, url, part_file, start, end, progress
                    )
                )
            for future in as_completed(futures):
                future.result()
    finally:
        progress.close()

    # Verify all chunks
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Verifying chunks...")
    for idx in range(chunk_count):
        start = idx * CHUNK_SIZE
        end = min(total_size - 1, start + CHUNK_SIZE - 1)
        part_file = os.path.join(parts_dir, f"part_{idx:05d}")
        if not os.path.exists(part_file):
            raise RuntimeError(f"Missing chunk: {part_file}")
        actual_size = os.path.getsize(part_file)
        expected_size = end - start + 1
        if actual_size != expected_size:
            raise RuntimeError(
                f"Corrupt chunk {part_file}: {actual_size} != {expected_size}"
            )

    merge_parts(output_file, parts_dir, chunk_count)
    cleanup(parts_dir)
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Download complete.")


def main():
    """Main function to process arguments and download a GGUF model."""
    parser = argparse.ArgumentParser(
        description="Download a .gguf model file from Ollama's registry",
        epilog="Example: python download_gguf.py phi3 3.8b --save-dir /path/to/directory",
    )
    parser.add_argument(
        "model_name", help="The name of the model to download (e.g., phi3)"
    )
    parser.add_argument(
        "model_parameters", help="The model parameters to use (e.g., 3.8b)"
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory to save the downloaded file (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )

    args = parser.parse_args()

    manifest, source_type = fetch_manifest(args.model_name, args.model_parameters)

    layers = manifest.get("layers", [])
    model_digest = None
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_digest = layer.get("digest")
            break

    if not model_digest:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Model digest not found in manifest.")
        sys.exit(1)

    download_url = get_blob_url(args.model_name, model_digest, source_type)
    safe_model_name = args.model_name.replace("/", "_")
    output_file = os.path.join(
        args.save_dir, f"{safe_model_name}_{args.model_parameters}.gguf"
    )

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved to: {output_file}")

    if args.workers > 1:
        download_large_file(download_url, output_file, workers=args.workers)
    else:
        filename = os.path.basename(output_file)
        save_dir = os.path.dirname(output_file) or "."
        filepath = download_file(download_url, filename, save_dir)
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Download completed: {filepath}")


if __name__ == "__main__":
    main()
