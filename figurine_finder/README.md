# Figurine Identifier API

## Overview

The Figurine Identifier API is a project that uses a deep learning model to classify whether an uploaded image contains a figurine or not. It exposes a REST API endpoint for predictions.

## Project Structure

The project is organized as follows:

*   `app/`: Contains the FastAPI application code (`main.py`) for serving the API.
*   `model/`: Contains the model training script (`train_model.py`) and will store the trained model.
*   `dataset/`: Contains placeholder files for the image dataset. **Users must replace these with actual images for training.**
    *   `dataset/train/figures/`: Training images of figurines.
    *   `dataset/train/non_figures/`: Training images that are not figurines.
    *   `dataset/validation/figures/`: Validation images of figurines.
    *   `dataset/validation/non_figures/`: Validation images that are not figurines.
*   `tests/`: Contains unit tests for the API (`test_api.py`).
*   `Dockerfile`: For building a Docker container for the application.
*   `requirements.txt`: Lists the Python dependencies.

## Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   Docker (Optional, for containerized deployment)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd figurine_finder
    ```

2.  **Create a Python virtual environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **On Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```

    *   **On Windows (PowerShell or cmd.exe):**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    Once the virtual environment is activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The `dataset/` directory currently contains placeholder text files (e.g., `figure_sample1.txt`). **You must replace these placeholders with actual image files to train the model.**

This project supports two main types of classification:
1.  **Binary Classification (Figurine vs. Non-Figurine):** For this, use the structure described in earlier versions (e.g., `figures/` and `non_figures/` subdirectories).
2.  **Multi-Class Figurine Identification:** To identify specific types or individual figurines, you need to organize your dataset by creating a subdirectory for each unique figurine ID (or class) within both the `dataset/train/` and `dataset/validation/` directories.

### Directory Structure for Multi-Class Identification

If you want the model to identify *specific types* of figurines (e.g., "figurine_id_A", "figurine_id_B"), structure your dataset as follows:

```
dataset/
├── train/
│   ├── figurine_id_A/  # Images for figurine type A
│   │   ├── image_A1.jpg
│   │   └── image_A2.jpg
│   ├── figurine_id_B/  # Images for figurine type B
│   │   ├── image_B1.jpg
│   │   └── image_B2.jpg
│   └── background/     # Optional: Images that are not figurines (or any specific figurine)
│       └── image_bg1.jpg
└── validation/
    ├── figurine_id_A/
    │   └── image_A_val.jpg
    ├── figurine_id_B/
    │   └── image_B_val.jpg
    └── background/
        └── image_bg_val.jpg
```

**Key points for multi-class setup:**

*   **Figurine-Specific Folders:** All images belonging to the same figurine type (e.g., all pictures of "figurine_id_A") should be placed directly into its corresponding folder (e.g., `dataset/train/figurine_id_A/`).
*   **Class Names from Folder Names:** The names of these subdirectories (e.g., `figurine_id_A`, `figurine_id_B`) will be used as the class labels during training. The API, when adapted for multi-class output, would then use these names as the `figure_id` in its predictions.
*   **Background/Non-Figurine Class:** You can include a `background` or `non_figures` class for images that do not belong to any specific figurine category. This helps the model differentiate figurines from other objects or empty scenes.
*   **Placeholder Files:** Remember to replace any placeholder files (like `.gitkeep` or the initial `.txt` samples) with your actual image files.

Common image formats like JPEG and PNG are generally supported. Ensure you have a good number of diverse images in each category (each figurine ID and the background class) for effective training.

### Train/Validation Split

When preparing your dataset, it's crucial to divide your images into training and validation sets. This split serves two main purposes:

1.  **Performance Evaluation:** The validation set provides an objective way to assess how well your model is generalizing to new, unseen data.
2.  **Overfitting Detection:** If the model performs exceptionally well on the training data but poorly on the validation data, it's a sign of overfitting (i.e., the model has memorized the training data but hasn't learned to generalize).

**Recommendation:**
A common practice is to allocate about 70-80% of your images for the `train` directory and the remaining 20-30% for the `validation` directory. For example, if you have 100 images of "figurine_id_A", you might put 70-80 images in `dataset/train/figurine_id_A/` and 20-30 images in `dataset/validation/figurine_id_A/`.

**Consistency Across Classes:**
It is important to maintain this split ratio consistently across all your classes. If you have multiple figurine IDs, each figurine's image set should be split in a similar ratio between the train and validation folders. This ensures that the validation set is a representative sample of all classes the model needs to learn.

## Training the Model

Once you have populated the `dataset/` directory with your images, you need to train the classification model.

1.  **Run the training script:**
    Navigate to the project root directory (`figurine_finder`) and run:
    ```bash
    python model/train_model.py
    ```
    (If you are already inside the `figurine_finder` directory, this command is correct. If running from one level above, use `python figurine_finder/model/train_model.py`)

2.  The script will use the images in `dataset/train` and `dataset/validation` to train the model.
3.  A trained model file will be saved as `model/figurine_classifier_model.h5`.

**Note:** The training script `model/train_model.py` has the actual training call (`build_and_train_model()`) commented out by default to prevent errors when only placeholder files are present. You will need to uncomment this line in the script after you have added your image dataset.

## Running the API

### Locally (with Uvicorn)

After training the model and ensuring `model/figurine_classifier_model.h5` exists:

1.  Navigate to the project root directory (`figurine_finder`).
2.  Run the Uvicorn server:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The `--reload` flag enables auto-reloading for development.

### With Docker

1.  **Build the Docker image:**
    From the project root directory (`figurine_finder`):
    ```bash
    docker build -t figurine_finder_api .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 figurine_finder_api
    ```
    This maps port 8000 of the container to port 8000 on your host machine.

## Using the API

The API provides one main endpoint for predictions:

*   `POST /predict/`

To use this endpoint, send a POST request with a multipart/form-data payload, where the `file` field contains the image you want to classify.

**Example using `curl`:**

```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/predict/
```

Replace `/path/to/your/image.jpg` with the actual path to an image file.

**Expected JSON Response:**

The API will return a JSON object with the following structure:

```json
{
  "figure_id": "predicted_figurine_id",
  "confidence": 0.95,
  "filename": "your_image.jpg"
}
```

*   `figure_id`: This field contains the predicted class for the image. If you trained a multi-class model (e.g., with subdirectories like `figurine_id_A`, `figurine_id_B`, `background`), the `figure_id` will be the name of the subdirectory (class) that the model predicts the image belongs to (e.g., "figurine_id_A"). For a binary model, this might be "figures" or "non_figures" (or similar, based on your class names).
*   `confidence`: This is a float representing the model's confidence score (typically between 0.0 and 1.0) in its prediction for the `figure_id`.
*   `filename`: The name of the uploaded file.

## Running Tests

To run the unit tests for the API:

1.  Ensure you have installed development dependencies (pytest is included in `requirements.txt`).
2.  Navigate to the project root directory (`figurine_finder`).
3.  Run pytest:
    ```bash
    PYTHONPATH=. pytest
    ```
    Setting `PYTHONPATH=.` ensures that Python can find the project's modules correctly from the `tests` directory.

This command will discover and run tests in the `tests/` directory.
