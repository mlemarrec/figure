import pytest
from fastapi.testclient import TestClient
import io
import os

# Adjust the import path based on how tests are run.
# If tests are run from the project root ('figurine_finder/'), this should work.
from figurine_finder.app.main import app, MODEL_PATH

# Ensure the model is not considered loaded for this test
# by temporarily renaming it if it exists, then renaming back.
# This is a bit of a workaround to ensure the "model not loaded" path is tested.
@pytest.fixture(scope="module", autouse=True)
def ensure_model_not_loaded_for_tests():
    original_model_path = MODEL_PATH
    temp_model_path = original_model_path + ".temp_testing"
    model_existed = False

    if os.path.exists(original_model_path):
        model_existed = True
        os.rename(original_model_path, temp_model_path)
        print(f"Temporarily moved model from {original_model_path} to {temp_model_path} for testing.")

    # Need to reload the app or specifically trigger the startup event
    # for the change to be reflected if the app instance was already configured.
    # For TestClient, it re-runs lifespan events on each client instantiation,
    # or we can explicitly call app.router.startup() if needed,
    # but typically TestClient handles this.

    yield # This is where the tests run

    if model_existed:
        os.rename(temp_model_path, original_model_path)
        print(f"Restored model from {temp_model_path} to {original_model_path}.")
    
    # If the app's model object was loaded, this won't reset it without restarting the app context.
    # However, TestClient(app) should call startup events. If main.model is global,
    # we might need a more sophisticated way to reset it or ensure the app's state.
    # For now, we rely on TestClient's behavior and the fact that `main.model` is global
    # and set during `load_dependencies_on_startup`.

client = TestClient(app)

def test_read_root():
    """Test the root GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    # Check for the expected keys, as content may vary (e.g. available_classes)
    assert "message" in response.json()
    assert "model_loaded" in response.json()
    assert "class_labels_loaded" in response.json()


def test_predict_endpoint_model_not_loaded():
    """
    Test the /predict/ POST endpoint when the model is not loaded.
    It should return a 503 Service Unavailable error.
    """
    # Create a dummy file-like object for the upload
    # The content doesn't matter much as the model loading check happens first.
    dummy_file_content = b"dummy image data"
    dummy_file = ("test_image.jpg", io.BytesIO(dummy_file_content), "image/jpeg")

    response = client.post("/predict/", files={"file": dummy_file})

    # As implemented in main.py, if model is None, it should return 503.
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded. Please ensure the model is trained and available."}

def test_predict_endpoint_invalid_image_file_if_model_were_loaded(monkeypatch):
    """
    Test the /predict/ POST endpoint with an invalid image file,
    assuming the model *was* loaded (we mock this part).
    This tests the preprocess_image function's error handling.
    """
    # Mock the global model object in main.py to simulate it being loaded
    # This is to bypass the "model not loaded" check and test deeper.
    class MockModel:
        def predict(self, data):
            # For binary classification, this was [[0.9]]
            # For multi-class, it would be something like [[0.1, 0.8, 0.1]]
            return [[0.1, 0.8, 0.1]] 

    # Mock class_labels as well for this test to pass the class_labels check
    mock_class_labels = {0: "class_A", 1: "class_B", 2: "class_C"}

    monkeypatch.setattr("figurine_finder.app.main.model", MockModel())
    monkeypatch.setattr("figurine_finder.app.main.class_labels", mock_class_labels)


    # Create an invalid image file (e.g., a text file)
    invalid_file_content = b"this is not an image"
    invalid_file = ("invalid.txt", io.BytesIO(invalid_file_content), "text/plain")

    response = client.post("/predict/", files={"file": invalid_file})

    assert response.status_code == 400 # Bad Request due to image processing error
    # The detail message might vary slightly depending on the PIL error.
    # We check if it contains a known part of our error message.
    assert "Invalid image file or format" in response.json()["detail"]

    # Clean up the monkeypatch by restoring the original model (or None)
    monkeypatch.undo()

# To run these tests, navigate to the 'figurine_finder' directory and run:
# PYTHONPATH=. pytest
# Or if figurine_finder is in PYTHONPATH:
# pytest tests/test_api.py
#
# The `ensure_model_not_loaded_for_tests` fixture attempts to hide the model file
# if it exists, to ensure the "model not loaded" path in the API is tested.
# This is important because the app loads the model on startup.
#
# The `PYTHONPATH=.` part is important if you run pytest from the `figurine_finder`
# directory, so that `from figurine_finder.app.main import app` works correctly.
# Alternatively, structure as a package and install with `pip install -e .`
# then `pytest` should work from anywhere.

# --- Comments for Future Multi-Class API Testing ---

# 1. Mocking a Multi-Class Model and `class_indices.json`:
#    To effectively test the multi-class capabilities of the `/predict/` endpoint,
#    we need to simulate a scenario where a multi-class model and its corresponding
#    `class_indices.json` are loaded.
#
#    Considerations for test setup:
#    -   **Mock Keras Model (`.h5`):**
#        *   A fixture could create a temporary dummy `.h5` file representing a
#          trained Keras model. This model should be configured to output a
#          multi-class prediction array (e.g., a softmax output like `[[0.1, 0.7, 0.2]]`).
#        *   Tools like `h5py` might be useful for creating a valid HDF5 file that
#          Keras `load_model` can parse, or more simply, `monkeypatch` `tf.keras.models.load_model`
#          to return a mock model object directly.
#
#    -   **Mock `class_indices.json`:**
#        *   A corresponding `class_indices.json` file needs to be created in the
#          `model/` directory (or wherever `CLASS_INDICES_PATH` points).
#        *   This JSON file should map class names to integer indices that align
#          with the output of the mock Keras model. For example:
#          `{"class_A": 0, "class_B": 1, "class_C": 2}`
#
#    -   **Fixture Adaptation:**
#        *   The existing `ensure_model_not_loaded_for_tests` fixture is designed
#          to test the "model not found" scenario.
#        *   New fixtures will be needed to:
#            a) Set up both a mock model file AND a mock `class_indices.json` file.
#            b) Set up a mock model file BUT NOT `class_indices.json` (for error handling tests).
#            c) Ensure these files are cleaned up after tests.
#        *   Alternatively, use `monkeypatch` extensively to mock `os.path.exists`,
#          `tf.keras.models.load_model`, and `json.load` within specific test functions.

# 2. Testing the `/predict/` Endpoint with Multi-Class Output:
#    Assuming the mock model and class indices are set up by a fixture or monkeypatching.
#
#    Example test case outline:
#    def test_predict_multi_class_success():
#        # Setup:
#        # - Ensure a mock multi-class model is loaded (e.g., via monkeypatching main.model).
#        # - Ensure mock class_labels are loaded (e.g., via monkeypatching main.class_labels).
#        #   main.class_labels should be like {0: "class_A", 1: "class_B", ...}
#
#        # Create a valid dummy image file
#        valid_image_content = b"dummy image data for multi-class" # Replace with actual image bytes if needed for preprocessing
#        valid_file = ("valid_image.jpg", io.BytesIO(valid_image_content), "image/jpeg")
#
#        # Expected prediction from the mock model (e.g., class_B is highest)
#        # mock_model.predict should return something like [[0.1, 0.8, 0.1]]
#        # expected_figure_id = "class_B"
#        # expected_confidence = 0.8
#
#        response = client.post("/predict/", files={"file": valid_file})
#
#        assert response.status_code == 200
#        response_data = response.json()
#        assert "figure_id" in response_data
#        assert isinstance(response_data["figure_id"], str)
#        # assert response_data["figure_id"] == expected_figure_id
#        assert "confidence" in response_data
#        assert isinstance(response_data["confidence"], float)
#        # assert response_data["confidence"] == expected_confidence
#        assert "filename" in response_data


# 3. Testing Error Handling (if `class_indices.json` is missing):
#    This tests the scenario where the model might be present, but the crucial
#    `class_indices.json` file is missing or fails to load.
#
#    Example test case outline:
#    def test_predict_class_indices_missing():
#        # Setup:
#        # - Ensure a mock model IS considered loaded (e.g., monkeypatch main.model).
#        # - Ensure `class_labels` in `main.py` is `None` (e.g., by ensuring
#        #   `CLASS_INDICES_PATH` does not exist and `load_dependencies_on_startup` is run,
#        #   or by directly monkeypatching `main.class_labels` to `None`).
#
#        dummy_file_content = b"dummy image data"
#        dummy_file = ("test_image.jpg", io.BytesIO(dummy_file_content), "image/jpeg")
#
#        response = client.post("/predict/", files={"file": dummy_file})
#
#        assert response.status_code == 503 # Service Unavailable
#        assert response.json() == {"detail": "Class labels not loaded. Please ensure class_indices.json is available from training."}

# Note: The `test_read_root` has been slightly modified to check for presence of keys
# rather than exact message, as the `available_classes` part can change.
# The `test_predict_endpoint_invalid_image_file_if_model_were_loaded` has been updated
# to also mock `class_labels` for the test to pass the initial checks in `predict_image`.
