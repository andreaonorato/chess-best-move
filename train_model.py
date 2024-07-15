import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model

# Define image data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Rescale pixel values
    width_shift_range=0.15,       # Randomly shift images horizontally
    height_shift_range=0.15,      # Randomly shift images vertically
    fill_mode='nearest'          # Fill in new pixels with the nearest pixel value
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation

# Define data generators for training and validation sets
train_generator = train_datagen.flow_from_directory(
    'dataset/train/',             # Path to the training dataset directory
    target_size=(224, 224),       # Resize images to 224x224 pixels
    batch_size=8,                 # Number of images to process in each batch
    class_mode='categorical',     # Use categorical labels for multi-class classification
    shuffle=True                  # Shuffle the training data
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation/',        # Path to the validation dataset directory
    target_size=(224, 224),       # Resize images to 224x224 pixels
    batch_size=8,                 # Number of images to process in each batch
    class_mode='categorical',     # Use categorical labels for multi-class classification
    shuffle=False                 # Do not shuffle the validation data
)

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)            # Add a global average pooling layer
x = Dense(1024, activation='relu')(x)      # Add a fully connected layer with 1024 units and ReLU activation
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  # Add a final softmax layer with units equal to the number of classes

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers to avoid training them
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with an Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate steps per epoch and validation steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Train the model using the training and validation data generators
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the trained model to a file
model.save('chess_piece_recognizer.h5')

# Load the trained model from the file
model = load_model('chess_piece_recognizer.h5')

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')
