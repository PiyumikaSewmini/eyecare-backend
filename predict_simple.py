

import sys
import json
import os

print("Starting prediction script...")
print("=" * 70)

# Import libraries
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    import joblib
    import numpy as np
    print("✓ All libraries loaded")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Run: pip install torch torchvision pillow joblib")
    sys.exit(1)

print("=" * 70)

# Define CORRECT model architecture (matches training script)
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # Create ResNet18
        resnet = models.resnet18(weights=None)
        # Replace final layer
        resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )
        # Store the modified resnet directly (NOT as self.model)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Find models
print("\nLooking for trained models...")
model_dir = r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\saved_models'

if not os.path.exists(model_dir):
    model_dir = 'saved_models'

if not os.path.exists(model_dir):
    print("✗ saved_models folder not found!")
    sys.exit(1)

image_model_path = os.path.join(model_dir, 'image_model.pth')
if os.path.exists(image_model_path):
    file_size = os.path.getsize(image_model_path) / (1024 * 1024)
    print(f"✓ Found image_model.pth ({file_size:.1f} MB)")
else:
    print(f"✗ image_model.pth not found")
    sys.exit(1)

print("=" * 70)

# Load the model
print("\nLoading model...")
try:
    device = torch.device('cpu')
    model = ImageModel()
    model.load_state_dict(torch.load(image_model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")
    sys.exit(1)

print("=" * 70)

# Find test image
print("\nLooking for test image...")

image_folders = [
    r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\data\images\Mild',
    r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\data\images\Moderate',
    r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\data\images\No_DR',
    r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\data\images\Severe',
    r'C:\Users\Piyumika Thennakoon\Desktop\eyecare-plus-new\DR_Training\data\images\Proliferate_DR',
    r'data\images\Mild',
    r'data\images\Moderate',
]

test_image = None
for folder in image_folders:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            test_image = os.path.join(folder, files[0])
            print(f"✓ Found: {test_image}")
            break

if test_image is None:
    print("✗ No test image found")
    sys.exit(1)

print("=" * 70)

# Process image
print("\nProcessing image...")
try:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print("✓ Image processed")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)

print("=" * 70)

# Make prediction
print("\nMaking prediction...")
try:
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    print("✓ Prediction complete!")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)

print("=" * 70)

# Show results
DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)
print(f"\nDiagnosis: {DR_CLASSES[predicted_class]}")
print(f"Severity Level: {predicted_class}")
print(f"Confidence: {confidence * 100:.1f}%")

print(f"\nAll Class Probabilities:")
for i, prob in enumerate(probabilities):
    bar = '█' * int(prob * 50)
    print(f"  {DR_CLASSES[i]:20} {prob * 100:5.1f}% {bar}")

# Recommendations
recommendations = {
    0: ("Annual screening", "12 months", "Routine"),
    1: ("Monitor closely", "6-12 months", "Non-urgent"),
    2: ("Ophthalmologist consultation", "3-6 months", "Semi-urgent"),
    3: ("Urgent referral needed", "2-3 months", "Urgent"),
    4: ("IMMEDIATE referral required", "1 month", "Emergency")
}

rec = recommendations[predicted_class]
print(f"\nRecommendations:")
print(f"  Action: {rec[0]}")
print(f"  Follow-up: {rec[1]}")
print(f"  Urgency: {rec[2]}")

# Lesions
lesions = {
    0: [],
    1: ["Microaneurysms"],
    2: ["Microaneurysms", "Hard Exudates"],
    3: ["Microaneurysms", "Hard Exudates", "Cotton Wool Spots"],
    4: ["Neovascularization", "Vitreous Hemorrhage"]
}

if lesions[predicted_class]:
    print(f"\nLesions Detected:")
    for lesion in lesions[predicted_class]:
        print(f"  - {lesion}")

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)

# Create JSON output for backend integration
result = {
    "stage": DR_CLASSES[predicted_class],
    "severity": predicted_class,
    "confidence": round(confidence * 100, 2),
    "recommendations": {
        "action": rec[0],
        "followUp": rec[1],
        "urgency": rec[2]
    },
    "lesionsDetected": lesions[predicted_class],
    "probabilities": {DR_CLASSES[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)}
}

print(f"\nJSON Output (for API):")
print(json.dumps(result, indent=2))

print("\nPress Enter to exit...")
input()