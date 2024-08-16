# GGUFDownloader

**GGUFDownloader** is a simple and user-friendly CLI tool to help you download GGUF model files directly from Ollama's registry. Whether you're preparing for model training or inference with tools like `llama.cpp`, this script simplifies the process, providing a seamless experience with progress tracking and robust error handling.

## Features

- **Effortless Downloads**: Quickly download GGUF model files from Ollama's registry with a simple command.
- **Integration Ready**: Use downloaded GGUF files with `llama.cpp` and other AI tools for model training, inference, and more.
- **Progress Tracking**: Stay informed with a colorful progress bar during downloads.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/gguf-downloader.git
   cd gguf-downloader
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To download a GGUF model file, run the following command:
```
python download_gguf.py <MODEL_NAME> <MODEL_PARAMETERS>
```


## Example
For example, to download the phi3 model with 3.8b parameters, use:
```
python download_gguf.py phi3 3.8b
```

This will download the file and save it as phi3:3.8b.gguf.

## Help
For more information about using the script, you can use the --help or -h option:
`python download_gguf.py --help`

## Contributing
Contributions are welcome! If you have suggestions, ideas, or bug reports, feel free to open an issue or submit a pull request.

## Acknowledgements
[Ollama](https://ollama.com/): For providing the GGUF model registry.

[llama.cpp](https://github.com/ggerganov/llama.cpp): For making AI model training and inference accessible.