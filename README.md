# GGUFDownloader

**GGUFDownloader** is a simple and user-friendly CLI tool to help you download GGUF model files directly from Ollama's registry. Whether you're preparing for model training or inference with tools like `llama.cpp`, this script simplifies the process, providing a seamless experience with progress tracking and robust error handling.

## Features

- **Effortless Downloads**: Quickly download GGUF model files from Ollama's registry with a simple command.
- **Smart Fallback**: Automatically tries both library and user-specific model formats.
- **User Namespace Support**: Download models from specific users (e.g., `username/model`).
- **Integration Ready**: Use downloaded GGUF files with `llama.cpp` and other AI tools for model training, inference, and more.
- **Progress Tracking**: Stay informed with a colorful progress bar during downloads.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/gguf-downloader.git
   cd gguf-downloader
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To download a GGUF model file, run the following command:
```
python download_gguf.py <MODEL_NAME> <MODEL_PARAMETERS>
```

### Model Formats

The tool supports two model formats:

1. **Library models** (official Ollama models):
   ```bash
   python download_gguf.py phi3 3.8b
   ```

2. **User-specific models** (models from specific users):
   ```bash
   python download_gguf.py username/model latest
   ```

The script will automatically try the library format first, and if the model is not found (404 error), it will attempt the user-specific format if a username is provided.

### Additional Options

- `--save-dir`: Specify a custom directory to save the downloaded file
  ```bash
  python download_gguf.py phi3 3.8b --save-dir /path/to/models
  ```


## Examples

### Library Models
Download the phi3 model with 3.8b parameters:
```bash
python download_gguf.py phi3 3.8b
```

### Custom Save Directory
Save the model to a specific directory:
```bash
python download_gguf.py phi3 3.8b --save-dir ./models
```

All examples will download the file and save it with a descriptive filename (e.g., `phi3_3.8b.gguf` or `username/model_latest.gguf`).

## Help
For more information about using the script, you can use the --help or -h option:
```bash
python download_gguf.py --help
```

## Troubleshooting

- **Model not found**: If you get a "Model not found in library" error, try using the user-specific format: `username/modelname`
- **Network issues**: The script includes timeout handling and will display clear error messages for network-related problems
- **Invalid model format**: Ensure you're using the correct model name and parameters as listed on the Ollama registry

## Contributing
Contributions are welcome! If you have suggestions, ideas, or bug reports, feel free to open an issue or submit a pull request.

## Acknowledgements
[Ollama](https://ollama.com/): For providing the GGUF model registry.

[llama.cpp](https://github.com/ggerganov/llama.cpp): For making AI model training and inference accessible.