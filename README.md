# GGUFDownloader

**GGUFDownloader** is a simple and user-friendly CLI tool to download GGUF model files directly from Ollama's registry. Supports **parallel chunked downloads** with automatic resume and range-detection — perfect for large models whether you're preparing for training or inference with `llama.cpp`.

## Features

- **Parallel Downloads**: Multi-threaded chunked downloading with configurable worker count
- **Auto-Resume**: Incomplete chunks resume from where they left off — never re-download finished work
- **Range Detection**: Automatically detects server range-request support; falls back to single-threaded when unavailable
- **Smart Fallback**: Automatically tries both library and user-specific model formats
- **User Namespace Support**: Download models from specific users (e.g., `username/model`)
- **Progress Tracking**: Stay informed with a colorful progress bar during downloads
- **Integration Ready**: Use downloaded GGUF files with `llama.cpp` and other AI tools

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/olamide226/ollama-gguf-downloader
   cd ollama-gguf-downloader
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python download_gguf.py <MODEL_NAME> <MODEL_PARAMETERS>
```

### Model Formats

1. **Library models** (official Ollama models):
   ```bash
   python download_gguf.py phi3 3.8b
   ```

2. **User-specific models** (models from specific users):
   ```bash
   python download_gguf.py username/model latest
   ```

The script automatically tries library format first, then falls back to user-specific on 404.

### Options

| Flag | Default | Description |
|---|---|---|
| `--save-dir` | `.` | Directory to save the downloaded file |
| `--workers` | `4` | Number of parallel download workers. Set to `1` to disable parallelism. |

```bash
# Parallel download with 8 workers
python download_gguf.py phi3 3.8b --save-dir ./models --workers 8

# Single-threaded (for servers without range support)
python download_gguf.py phi3 3.8b --workers 1
```

## How Parallel Download Works

1. **Probe**: A `HEAD` request checks file size and range-request support
2. **Chunk**: File is split into 64 MB chunks
3. **Resume scan**: Already-complete chunks are skipped (supports interrupted downloads)
4. **Parallel fetch**: Worker threads download chunks via HTTP `Range` requests
5. **Verify**: Each chunk's size is validated after download
6. **Merge**: Chunks are concatenated atomically (`.merging` temp file → rename)
7. **Cleanup**: Temporary `.parts` directory is removed

If the server doesn't support range requests, falls back transparently to single-threaded download.

## Development

### Quick Start

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Full suite with coverage
python -m pytest --cov=download_gguf --cov-report=term-missing -v

# Or use the runner script
python run_tests.py
```

### Code Formatting

This project uses **black** and **isort** (with `--profile black`). Configuration lives in `pyproject.toml`.

```bash
# Format everything
black .
isort .
```

CI enforces formatting — install the pre-commit hooks to catch issues early:

```bash
pip install pre-commit
# Add .pre-commit-config.yaml (see below)
```

## Troubleshooting

- **Model not found**: Try user-specific format: `username/modelname`
- **Network issues**: The script includes timeout handling and clear error messages
- **Interrupted download**: Re-run the same command — completed chunks resume automatically

## Contributing

1. Install dev deps: `pip install -e ".[dev]"`
2. Format before committing: `black . && isort .`
3. Run tests: `python -m pytest --cov=download_gguf -v`
4. Open a PR against `main`

## Acknowledgements

[Ollama](https://ollama.com/) — GGUF model registry
[llama.cpp](https://github.com/ggerganov/llama.cpp) — Accessible AI model inference
