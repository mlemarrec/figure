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
    # and set during `load_model_on_startup`.

client = TestClient(app)

def test_read_root():
    """Test the root GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Figurine Classifier API!"}

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
            return [[0.9]] # Dummy prediction

    monkeypatch.setattr("figurine_finder.app.main.model", MockModel())

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
