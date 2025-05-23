import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import json

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = '../dataset/train'
VALIDATION_DIR = '../dataset/validation'
MODEL_SAVE_PATH = 'figurine_classifier_model.h5' # For the model file
CLASS_INDICES_SAVE_PATH = 'class_indices.json' # For the class indices

def build_and_train_model():
    # Determine number of classes and class names
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory {TRAIN_DIR} does not exist.")
        return

    class_names = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    num_classes = len(class_names)

    if num_classes == 0:
        print(f"Error: No subdirectories (classes) found in {TRAIN_DIR}. Please create subdirectories for each class.")
        return
    
    print(f"Found {num_classes} classes: {class_names}")

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

    # Determine class_mode based on num_classes
    if num_classes <= 2: # Includes case of 1 class (problematic) or 2 classes (binary)
        class_mode = 'binary'
        if num_classes == 1:
            print("Warning: Only one class found. This may lead to issues in training.")
            # Keras flow_from_directory might handle this, but usually binary is for 2 classes
            # Or, if it's just one class of figurines, it's better to have a 'non-figurine' class too.
    else:
        class_mode = 'categorical'
    
    print(f"Using class_mode: '{class_mode}' for ImageDataGenerator.")

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=class_mode
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=class_mode
    )

    # Save class indices
    class_indices = train_generator.class_indices
    print(f"Class indices: {class_indices}")
    try:
        with open(CLASS_INDICES_SAVE_PATH, 'w') as f:
            json.dump(class_indices, f)
        print(f"Saved class indices to {CLASS_INDICES_SAVE_PATH}")
    except IOError as e:
        print(f"Error saving class indices: {e}")
        return


    # Model Architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)

    if class_mode == 'binary':
        predictions = Dense(1, activation='sigmoid')(x)
        loss_function = 'binary_crossentropy'
    else: # categorical
        predictions = Dense(num_classes, activation='softmax')(x)
        loss_function = 'categorical_crossentropy'

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the Model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss_function,
                  metrics=['accuracy'])
    
    model.summary()

    # Training
    # This will not run correctly with placeholder text files if they are not actual images.
    # Replace placeholder files with actual images for this to work.
    print("Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > 0 else 1,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > 0 else 1
    )
    print("Model training finished.")

    # Save the Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == '__main__':
    print("Figurine Finder Model Training Script (Multi-class capable)")
    print("-----------------------------------------------------------")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Using training directory: {TRAIN_DIR}")
    print(f"Using validation directory: {VALIDATION_DIR}")
    print(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("This script trains a model based on the subdirectories found in the TRAIN_DIR.")
    print("For binary classification (e.g., 'figures', 'non_figures'), ensure two subdirectories.")
    print("For multi-class classification, create a subdirectory for each class (e.g., 'figurine_A', 'figurine_B', 'background').")

    # Check if TRAIN_DIR exists before attempting to list its contents for warnings.
    if os.path.exists(TRAIN_DIR):
        # Basic check for subdirectories to guide the user
        subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        if not subdirs:
            print(f"Warning: No subdirectories found in {TRAIN_DIR}. Training will likely fail.")
            print("Please create subdirectories for each class you want to train.")
        elif len(subdirs) == 1:
            print(f"Warning: Only one subdirectory ('{subdirs[0]}') found in {TRAIN_DIR}. This is unusual for classification.")
            print("Consider adding another class (e.g., a 'background' or 'non-figurine' class).")
        else:
            print(f"Found {len(subdirs)} subdirectories (classes): {subdirs}. Proceeding with training setup.")
    else:
        print(f"Error: Training directory {TRAIN_DIR} not found. Please create it and populate it with class subdirectories.")


    # build_and_train_model()
    print("\nbuild_and_train_model() is commented out to prevent errors with placeholder/missing image files.")
    print("Ensure your dataset is correctly set up with actual images in class subdirectories, then uncomment the line above to run training.")
