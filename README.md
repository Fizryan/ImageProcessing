# Image Restoration Pipeline

An end-to-end, PyTorch-based pipeline for training and deploying deep learning models for various image restoration tasks. This project provides a complete workflow from data collection and preparation to model training and inference, all managed through a simple and interactive command-line interface.

## ✨ Features

-   **End-to-End Workflow**: Integrated tools for every step: downloading images, preprocessing, training, and inference.
-   **Multi-Task Restoration**: Capable of handling several common image corruptions:
    -   **Denoising**: Removing random noise from images.
    -   **Demosaicing**: Reconstructing images from mosaic patterns.
    -   **Inpainting**: Filling in missing or corrupted parts of an image.
-   **Advanced U-Net Architecture**: Utilizes a powerful and efficient U-Net model (`AdvancedUNet`) for high-quality image reconstruction.
-   **High-Performance Training**: Optimized for speed and efficiency with:
    -   Mixed-Precision Training (AMP) on CUDA devices.
    -   `torch.compile` for model optimization (for PyTorch 2.0+).
    -   Gradient Accumulation to simulate larger batch sizes.
    -   Checkpointing system to resume training and save the best models.
-   **Interactive CLI**: A user-friendly command-line menu to easily run any part of the pipeline.
-   **Automated Dataset Preparation**: Scripts to automatically download, resize, and generate corrupted image pairs for training.

## 📂 Project Structure

```
ImageProcessing/
├── dataset/                # Default location for image data
│   ├── clean_images/
│   ├── resized_images/
│   ├── noisy_images/
│   ├── mosaic_images/
│   └── inpainting_images/
├── logs/                   # Generated log files
├── program/                # Main source code modules
│   ├── Architecture.py     # Model architecture definition
│   ├── CollectingImage.py  # Image Data collection script
│   ├── Inference.py        # Inference script
│   ├── Mosaic.py           # Mosaic image generation script
│   ├── Noise.py            # Noise image generation script
│   ├── Resizing.py         # Image resizing script
│   ├── Training.py         # Training script
│   ├── Training_Config.py  # Main configuration for training
│   ├── Logging_Config.py   # Log configuration for logging
│   └── ...
├── Results/                # Default output directory for inference
├── Samples/                # Sample images for testing inference
├── Training/               # Training artifacts
│   ├── checkpoints/        # Saved model checkpoints
│   └── previews/           # Image previews generated during training
├── main.py                 # Main script to run the pipeline
└── requirements.txt        # Project dependencies
```

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   NVIDIA GPU with CUDA support (highly recommended for training)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fizryan/ImageProcessing.git
    cd ImageProcessing
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    *On Windows*
    ```bash
    venv\Scripts\activate
    ```
    *On macOS/Linux*
    ```bash
    source venv/bin/activate
    ```

4.  **Install PyTorch:**
    The project requires PyTorch. Please visit the official PyTorch website to get the correct installation command for your system (e.g., CPU or a specific CUDA version).

    *Example for CUDA 12.1:*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

5.  **Install other dependencies:**
    Once PyTorch is installed, install the remaining packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

The entire pipeline is controlled via the main script `main.py`. Run it from the terminal:

```bash
python main.py
```

This will launch an interactive menu where you can select the desired action.

### Recommended Workflow

A typical workflow involves preparing the dataset, training the model, and finally running inference.

#### 1. Prepare the Dataset (Option 8)

This is the recommended first step. This option runs a complete data preparation pipeline that will prompt you for configuration:
1.  **Download Images**: Downloads a specified number of high-quality images to `dataset/clean_images`.
2.  **Resize Images**: Resizes the clean images to the model's input dimensions and saves them to `dataset/resized_images`.
3.  **Generate Corruptions**: Creates noisy and mosaic versions of the resized images, placing them in their respective directories (`dataset/noisy_images`, `dataset/mosaic_images`, etc.).
*Note: The `inpainting` task uses grayscale images as a stand-in for images with missing parts. This step also generates them.*

#### 2. Train the Model (Option 6)

Once the dataset is ready, you can start training.
-   This option uses the configuration defined in `program/Training_Config.py`. You can modify this file to change hyperparameters like learning rate, batch size, and number of epochs.
-   The trainer automatically looks for datasets in the directories specified in the config.
-   Training progress is saved continuously. You can stop (`Ctrl+C`) and resume training at any time.
-   Checkpoints are saved in `Training/checkpoints/`, with `best_model.pth` being the model with the lowest validation loss.
-   Visual previews of the model's performance are saved in `Training/previews/`.

#### 3. Run Inference (Option 7)

After training, use the best model to restore corrupted images.
-   You will be prompted for:
    -   The path to the trained model (e.g., `Training/checkpoints/best_model.pth`).
    -   The path to the input image you want to restore.
    -   The path for the saved, restored output image.
    -   The `task_type` (e.g., `noise`, `mosaic`, `inpainting`) to help the model process the input correctly.

The restored image will be saved to the specified output path.

## 🛠️ Technology Stack

-   **PyTorch**: The core deep learning framework.
-   **Pillow (PIL)**: For image manipulation and processing.
-   **tqdm**: For elegant and informative progress bars.
-   **GPUtil**: For monitoring GPU status during training.
-   **Requests**: For downloading images.

### ⚙️ Configuration

All major training parameters can be adjusted in `program/Training_Config.py`. This includes:
-   Dataset paths
-   Image dimensions (`img_height`, `img_width`)
-   Hyperparameters (`learning_rate`, `num_epochs`)
-   Dataloader settings (`batch_size`, `num_workers`)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to open an issue or submit a pull request.