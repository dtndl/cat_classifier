import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm

def load_model(model_path, num_classes=2):
    """Load the trained model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_test_loader(batch_size=16):
    """Create test data loader."""
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder('data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    return test_loader, test_dataset.classes

def test_model(model, test_loader, device):
    """Test the model and return predictions and true labels."""
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def plot_sample_predictions(test_loader, model, device, classes, num_samples=8):
    """Plot sample predictions with confidence scores."""
    model = model.to(device)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 4, i+1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        confidence = probs[i][preds[i]].item()
        
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png')
    plt.close()

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model('models/best_model.pth')
    
    # Get test loader
    test_loader, classes = get_test_loader()
    
    # Test model
    preds, labels, probs = test_model(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, preds, classes)
    
    # Plot sample predictions
    plot_sample_predictions(test_loader, model, device, classes)
    
    print("\nResults saved to 'results' directory:")
    print("- confusion_matrix.png: Shows how many predictions were correct/incorrect")
    print("- sample_predictions.png: Shows sample images with predictions and confidence scores")

if __name__ == '__main__':
    main() 