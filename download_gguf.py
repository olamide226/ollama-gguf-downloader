#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB


def fetch_manifest(model_name, model_parameters):
    url = f"https://registry.ollama.ai/v2/library/{model_name}/manifests/{model_parameters}"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            return response.json(), "library"

        if response.status_code == 404:
            if "/" in model_name:
                user, model = model_name.split("/", 1)

                fallback_url = (
                    f"https://registry.ollama.ai/v2/{user}/{model}/manifests/{model_parameters}"
                )

                fallback_response = requests.get(
                    fallback_url,
                    timeout=30,
                )

                fallback_response.raise_for_status()

                return fallback_response.json(), "user"

            raise RuntimeError(
                f"Model '{model_name}:{model_parameters}' not found."
            )

        response.raise_for_status()

    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {e}")
        sys.exit(1)


def get_blob_url(model_name, digest, source_type):
    if source_type == "library":
        return (
            f"https://registry.ollama.ai/v2/library/{model_name}/blobs/{digest}"
        )

    user, model = model_name.split("/", 1)

    return (
        f"https://registry.ollama.ai/v2/{user}/{model}/blobs/{digest}"
    )


def get_file_info(url):
    r = requests.head(
        url,
        allow_redirects=True,
        timeout=60,
    )

    r.raise_for_status()

    size = int(r.headers.get("content-length", 0))

    supports_ranges = False

    try:
        rr = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=60,
        )

        supports_ranges = (
            rr.status_code == 206
            or "content-range" in rr.headers
        )

        rr.close()

    except Exception:
        pass

    return size, supports_ranges


def save_metadata(meta_file, metadata):
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(meta_file):
    if not os.path.exists(meta_file):
        return None

    with open(meta_file) as f:
        return json.load(f)


def download_chunk(
    url,
    part_file,
    start,
    end,
    progress,
):
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

    headers = {
        "Range": f"bytes={start + existing}-{end}"
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=300,
    ) as r:

        if r.status_code not in (206,):
            raise RuntimeError(
                f"Server ignored Range request "
                f"({r.status_code})"
            )

        with open(part_file, "ab") as f:
            for chunk in r.iter_content(
                1024 * 1024
            ):
                if not chunk:
                    continue

                f.write(chunk)
                progress.update(len(chunk))

def merge_parts(
    output_file,
    parts_dir,
    chunk_count,
):
    temp_output = output_file + ".merging"

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Merging chunks..."
    )

    with open(temp_output, "wb") as out:

        for idx in range(chunk_count):

            part_file = os.path.join(
                parts_dir,
                f"part_{idx:05d}",
            )

            with open(part_file, "rb") as inp:

                while True:
                    data = inp.read(8 * 1024 * 1024)

                    if not data:
                        break

                    out.write(data)

    os.replace(temp_output, output_file)


def cleanup(parts_dir):
    if os.path.exists(parts_dir):
        shutil.rmtree(parts_dir)


def download_large_file(
    url,
    output_file,
    workers=8,
):
    total_size, _ = get_file_info(url)

    if os.path.exists(output_file):
        existing_size = os.path.getsize(output_file)

        if existing_size == total_size:
            print(
                f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} File already exists."
            )
            return

    parts_dir = output_file + ".parts"

    os.makedirs(parts_dir, exist_ok=True)

    meta_file = os.path.join(
        parts_dir,
        "metadata.json",
    )

    chunk_count = math.ceil(
        total_size / CHUNK_SIZE
    )

    metadata = {
        "url": url,
        "size": total_size,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": chunk_count,
    }

    save_metadata(meta_file, metadata)

    completed_bytes = 0

    for idx in range(chunk_count):

        start = idx * CHUNK_SIZE

        end = min(
            total_size - 1,
            start + CHUNK_SIZE - 1,
        )

        expected_size = end - start + 1

        part_file = os.path.join(
            parts_dir,
            f"part_{idx:05d}",
        )

        if (
            os.path.exists(part_file)
            and os.path.getsize(part_file)
            == expected_size
        ):
            completed_bytes += expected_size

    progress = tqdm(
        total=total_size,
        initial=completed_bytes,
        unit="B",
        unit_scale=True,
        desc="Downloading",
    )

    futures = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        for idx in range(chunk_count):

            start = idx * CHUNK_SIZE

            end = min(
                total_size - 1,
                start + CHUNK_SIZE - 1,
            )

            expected_size = end - start + 1

            part_file = os.path.join(
                parts_dir,
                f"part_{idx:05d}",
            )

            if (
                os.path.exists(part_file)
                and os.path.getsize(part_file)
                == expected_size
            ):
                continue

            futures.append(
                executor.submit(
                    download_chunk,
                    url,
                    part_file,
                    start,
                    end,
                    progress,
                )
            )

        for future in as_completed(futures):
            future.result()

    progress.close()

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Verifying chunks..."
    )

    for idx in range(chunk_count):

        start = idx * CHUNK_SIZE

        end = min(
            total_size - 1,
            start + CHUNK_SIZE - 1,
        )

        expected_size = end - start + 1

        part_file = os.path.join(
            parts_dir,
            f"part_{idx:05d}",
        )

        if not os.path.exists(part_file):
            raise RuntimeError(
                f"Missing chunk: {part_file}"
            )

        actual_size = os.path.getsize(part_file)

        if actual_size != expected_size:
            raise RuntimeError(
                f"Corrupt chunk: {part_file}"
            )

    merge_parts(
        output_file,
        parts_dir,
        chunk_count,
    )

    cleanup(parts_dir)

    print(
        f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Download complete."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF files from Ollama with resume support"
    )

    parser.add_argument(
        "model_name",
    )

    parser.add_argument(
        "model_parameters",
    )

    parser.add_argument(
        "--save-dir",
        default=".",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers",
    )

    args = parser.parse_args()

    manifest, source_type = fetch_manifest(
        args.model_name,
        args.model_parameters,
    )

    model_digest = None

    for layer in manifest.get("layers", []):

        if (
            layer.get("mediaType")
            == "application/vnd.ollama.image.model"
        ):
            model_digest = layer["digest"]
            break

    if not model_digest:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL} Model digest not found."
        )
        sys.exit(1)

    download_url = get_blob_url(
        args.model_name,
        model_digest,
        source_type,
    )

    safe_name = args.model_name.replace("/", "_")

    output_file = os.path.join(
        args.save_dir,
        f"{safe_name}_{args.model_parameters}.gguf",
    )

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Download URL resolved."
    )

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saving to: {output_file}"
    )

    download_large_file(
        download_url,
        output_file,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
