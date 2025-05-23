import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = '../dataset/train'
VALIDATION_DIR = '../dataset/validation'
MODEL_SAVE_PATH = 'figurine_classifier_model.h5'

def build_and_train_model():
    # Image Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Model Architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the Model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Training (Conceptual)
    # This will not run correctly with placeholder text files.
    # Replace placeholder files with actual images for this to work.
    print("Starting model training (conceptual)...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > 0 else 1,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > 0 else 1
    )
    print("Model training (conceptual) finished.")

    # Save the Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == '__main__':
    # Note: This script expects image files in the dataset directories.
    # The current placeholder .txt files will cause errors during image loading.
    # Replace them with actual images before running.
    print("Figurine Finder Model Training Script")
    print("-----------------------------------")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Using training directory: {TRAIN_DIR}")
    print(f"Using validation directory: {VALIDATION_DIR}")
    print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Check if dataset directories have content (even if placeholders)
    # This helps in giving a more direct feedback if ImageDataGenerator would fail immediately
    import os
    if not os.listdir(os.path.join(TRAIN_DIR, 'figures')):
        print(f"Warning: Training directory '{os.path.join(TRAIN_DIR, 'figures')}' is empty.")
    if not os.listdir(os.path.join(TRAIN_DIR, 'non_figures')):
        print(f"Warning: Training directory '{os.path.join(TRAIN_DIR, 'non_figures')}' is empty.")
    if not os.listdir(os.path.join(VALIDATION_DIR, 'figures')):
        print(f"Warning: Validation directory '{os.path.join(VALIDATION_DIR, 'figures')}' is empty.")
    if not os.listdir(os.path.join(VALIDATION_DIR, 'non_figures')):
        print(f"Warning: Validation directory '{os.path.join(VALIDATION_DIR, 'non_figures')}' is empty.")

    # build_and_train_model()
    print("build_and_train_model() is commented out to prevent errors with placeholder files.")
    print("Uncomment it when you have actual images in the dataset folders.")
