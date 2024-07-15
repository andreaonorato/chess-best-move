import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

# Define your class labels (ensure this matches the labels used in training)
class_labels = {
    0: 'b_empty',  # Empty square on black side
    1: 'bb',       # Black bishop
    2: 'bk',       # Black king
    3: 'bn',       # Black pawn
    4: 'bp',       # Black queen
    5: 'bq',       # Black rook
    6: 'br',       # White knight
    7: 'w_empty',  # Empty square on white side
    8: 'wb',       # White pawn
    9: 'wk',       # White queen
    10: 'wn',      # White king
    11: 'wp',      # White pawn
    12: 'wq',      # White queen
    13: 'wr'       # White rook
}

def predict_piece(model, img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the predicted class index to the class label
    predicted_label = class_labels.get(predicted_class, 'Unknown')
    
    return predicted_label

def evaluate_dataset(model, dataset_path):
    correct_predictions = 0
    total_predictions = 0

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                true_label = folder_name
                predicted_label = predict_piece(model, img_path)

                if predicted_label == true_label:
                    correct_predictions += 1
                    print(f'Image: {img_path}')
                    print(f'Predicted: {predicted_label}')
                    print(f'True: {true_label}')
                    print('---')
                else:
                    print(f'Image: {img_path}')
                    print(f'Predicted: {predicted_label}')
                    print(f'True: {true_label}')
                    print('---')

                total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f'Correct predictions: {correct_predictions}')
    print(f'Total predictions: {total_predictions}')
    print(f'Accuracy: {accuracy:.2%}')

# Load the trained model
model = load_model('chess_piece_recognizer.h5')

# Evaluate the entire dataset
dataset_path = r'C:\Users\USER\workspace\python\personal_projects\AI_Chess_cheater\dataset\validation'
evaluate_dataset(model, dataset_path)
