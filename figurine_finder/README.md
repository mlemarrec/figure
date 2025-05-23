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

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The `dataset/` directory currently contains placeholder text files (e.g., `figure_sample1.txt`). **You must replace these placeholders with actual image files to train the model.**

The expected directory structure for your images is:

*   `dataset/train/figures/` (e.g., `figure1.jpg`, `figure2.png`)
*   `dataset/train/non_figures/` (e.g., `not_figure1.jpg`, `background1.png`)
*   `dataset/validation/figures/` (e.g., `val_figure1.jpg`)
*   `dataset/validation/non_figures/` (e.g., `val_not_figure1.jpg`)

Common image formats like JPEG and PNG are generally supported. Ensure you have a good number of diverse images in each category for effective training.

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
  "is_figure": true,  // boolean: true if the image is classified as a figurine, false otherwise
  "confidence": 0.95, // float: the model's confidence score (0.0 to 1.0)
  "filename": "your_image.jpg" // string: the name of the uploaded file
}
```

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
