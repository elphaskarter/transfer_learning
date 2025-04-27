import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from google.colab import drive

def main():
    # Mount Google Drive
    drive.mount('/content/drive')
    data_dir = '/content/drive/MyDrive/UCMerced'  # Update this path

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset and split
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load pre-trained ResNet18 and modify
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove classification head
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Feature extraction function
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                features.append(outputs)
                labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(labels)

    # Extract features
    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    # Train SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(train_features, train_labels)

    # Evaluate
    y_pred = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=dataset.classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))

if __name__ == "__main__":
    main()