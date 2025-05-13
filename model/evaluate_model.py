# Model Evaluation Script for Pneumonia Detection

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def load_model_and_config():
    """Load the trained model and its configuration."""
    model_path = 'model/model.h5'
    config_path = 'model/model_config.json'
    
    # Check if model and config files exist
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Model or configuration file not found. Please train the model first.")
    
    # Load model
    model = load_model(model_path)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return model, config

def load_test_data(config, batch_size=32):
    """Load test data for evaluation."""
    # Set up data generator for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        'data/processed/test',
        target_size=tuple(config['input_shape'][:2]),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for maintaining order in prediction
    )
    
    return test_generator

def evaluate_model_metrics(model, test_generator):
    """Evaluate model on test set and calculate metrics."""
    # Evaluate the model
    results = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    
    # Print results
    print("\nTest Results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")
    
    # Calculate F1 score
    precision = results[2]  # Index might vary based on model compilation
    recall = results[3]  # Index might vary based on model compilation
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    print(f"F1 Score: {f1_score:.4f}")
    
    # Save results
    metrics = {
        'loss': float(results[0]),
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'f1_score': float(f1_score)
    }
    
    return metrics

def generate_predictions(model, test_generator):
    """Generate predictions for the test set."""
    # Reset the generator to start from the beginning
    test_generator.reset()
    
    # Generate predictions
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    
    # Get true labels
    true_labels = test_generator.classes
    
    # Convert predictions to binary labels
    pred_labels = (predictions > 0.5).astype(int).flatten()
    
    return true_labels, pred_labels, predictions

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """Plot confusion matrix."""
    # Create plots directory if it doesn't exist
    os.makedirs('model/plots', exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('model/plots/confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved to model/plots/confusion_matrix.png")

def plot_roc_curve(true_labels, predictions):
    """Plot ROC curve."""
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('model/plots/roc_curve.png')
    plt.close()
    print("ROC curve plot saved to model/plots/roc_curve.png")

def generate_classification_report(true_labels, pred_labels, class_names):
    """Generate and save classification report."""
    # Generate classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    
    # Save report as JSON
    with open('model/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))
    
    return report

def visualize_predictions(test_generator, true_labels, pred_labels, predictions, num_samples=5):
    """Visualize some sample predictions."""
    # Create plots directory if it doesn't exist
    os.makedirs('model/plots', exist_ok=True)
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Find indices of correct and incorrect predictions
    correct_indices = np.where(true_labels == pred_labels)[0]
    incorrect_indices = np.where(true_labels != pred_labels)[0]
    
    # Select random samples from correct and incorrect predictions
    if len(correct_indices) > 0:
        correct_samples = np.random.choice(correct_indices, size=min(num_samples, len(correct_indices)), replace=False)
    else:
        correct_samples = []
        
    if len(incorrect_indices) > 0:
        incorrect_samples = np.random.choice(incorrect_indices, size=min(num_samples, len(incorrect_indices)), replace=False)
    else:
        incorrect_samples = []
    
    # Visualize correct predictions
    if len(correct_samples) > 0:
        plt.figure(figsize=(15, 5))
        plt.suptitle('Correct Predictions', fontsize=16)
        
        for i, idx in enumerate(correct_samples):
            plt.subplot(1, len(correct_samples), i + 1)
            
            # Get the image from the generator
            batch_idx = idx // test_generator.batch_size
            in_batch_idx = idx % test_generator.batch_size
            
            # Reset the generator and move to the correct batch
            test_generator.reset()
            for _ in range(batch_idx + 1):
                X_batch, _ = next(test_generator)
            
            # Display the image
            img = X_batch[in_batch_idx]
            plt.imshow(img)
            
            # Add title with prediction and confidence
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            confidence = predictions[idx][0] if pred_labels[idx] == 1 else 1 - predictions[idx][0]
            plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('model/plots/correct_predictions.png')
        plt.close()
        print("Correct predictions plot saved to model/plots/correct_predictions.png")
    
    # Visualize incorrect predictions
    if len(incorrect_samples) > 0:
        plt.figure(figsize=(15, 5))
        plt.suptitle('Incorrect Predictions', fontsize=16)
        
        for i, idx in enumerate(incorrect_samples):
            plt.subplot(1, len(incorrect_samples), i + 1)
            
            # Get the image from the generator
            batch_idx = idx // test_generator.batch_size
            in_batch_idx = idx % test_generator.batch_size
            
            # Reset the generator and move to the correct batch
            test_generator.reset()
            for _ in range(batch_idx + 1):
                X_batch, _ = next(test_generator)
            
            # Display the image
            img = X_batch[in_batch_idx]
            plt.imshow(img)
            
            # Add title with prediction and confidence
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            confidence = predictions[idx][0] if pred_labels[idx] == 1 else 1 - predictions[idx][0]
            plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('model/plots/incorrect_predictions.png')
        plt.close()
        print("Incorrect predictions plot saved to model/plots/incorrect_predictions.png")

def main():
    print("Starting model evaluation for pneumonia detection...")
    
    # Load model and configuration
    print("Loading trained model and configuration...")
    model, config = load_model_and_config()
    
    # Load test data
    print("Loading test data...")
    test_generator = load_test_data(config)
    
    # Evaluate model metrics
    print("Evaluating model metrics...")
    metrics = evaluate_model_metrics(model, test_generator)
    
    # Generate predictions
    print("Generating predictions for test set...")
    true_labels, pred_labels, predictions = generate_predictions(model, test_generator)
    
    # Generate classification report
    print("Generating classification report...")
    report = generate_classification_report(true_labels, pred_labels, config['class_names'])
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(true_labels, pred_labels, config['class_names'])
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(true_labels, predictions)
    
    # Visualize predictions
    print("Visualizing sample predictions...")
    visualize_predictions(test_generator, true_labels, pred_labels, predictions)
    
    print("Model evaluation completed!")

if __name__ == "__main__":
    main()
