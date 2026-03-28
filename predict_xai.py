import sys
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'saved_models', 'image_model.pth')

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )
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
        
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Register hook for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def generate_gradcam(model, img_tensor, target_class):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Forward pass
    output = model(img_tensor)
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # Get gradients and activations
    gradients = model.gradients.data
    activations = model.activations.data
    
    # Calculate weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Weighted combination
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    return cam[0, 0].cpu().numpy()

def create_heatmap_overlay(original_img_path, heatmap, output_path):
    """Create heatmap overlay on original image"""
    # Load original image
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap (red = high attention, blue = low attention)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return output_path

def get_patient_friendly_explanation(severity, confidence, hba1c, diabetes_duration):
    """Generate clear, patient-friendly explanations"""
    
    explanations = {
        0: {
            'title': 'No Diabetic Retinopathy',
            'simple': 'Your retina appears healthy with no signs of diabetic damage.',
            'what_it_means': 'This is excellent news! The blood vessels in your retina are healthy and show no damage from diabetes.',
            'what_next': [
                'Continue your current diabetes management',
                'Keep your blood sugar and blood pressure under control',
                'Have an eye exam every 12 months',
                'Maintain a healthy lifestyle'
            ],
            'urgency': 'Routine',
            'color': 'green',
            'risk_level': 'Low'
        },
        1: {
            'title': 'Mild Diabetic Retinopathy',
            'simple': 'Early signs of diabetic eye disease detected, but vision is not yet affected.',
            'what_it_means': 'Small areas of swelling (microaneurysms) have appeared in blood vessels of your retina. This is an early warning sign but does NOT mean you will lose vision.',
            'what_next': [
                'Monitor your blood sugar levels closely',
                'Have eye exams every 6-12 months',
                'Control your HbA1c (target below 7%)',
                'Manage blood pressure (target below 140/90)',
                'No immediate treatment needed, but close monitoring is important'
            ],
            'urgency': 'Monitor Closely',
            'color': 'yellow',
            'risk_level': 'Low-Moderate'
        },
        2: {
            'title': 'Moderate Diabetic Retinopathy',
            'simple': 'Noticeable damage to retinal blood vessels, but vision may still be normal.',
            'what_it_means': 'Blood vessels in your retina are becoming blocked. You may have bleeding spots (hemorrhages) and fatty deposits (exudates). Vision might still feel normal, but damage is progressing.',
            'what_next': [
                'See an eye specialist (ophthalmologist) soon',
                'Have detailed retinal examination every 3-6 months',
                'Strict blood sugar control is CRITICAL now',
                'May need laser treatment if it progresses',
                'Don\'t wait - early treatment prevents vision loss'
            ],
            'urgency': 'See Specialist Soon',
            'color': 'orange',
            'risk_level': 'Moderate'
        },
        3: {
            'title': 'Severe Diabetic Retinopathy',
            'simple': 'Significant blood vessel damage requiring urgent specialist care.',
            'what_it_means': 'Many blood vessels in your retina are blocked. Your eye is not getting enough oxygen, which can trigger growth of abnormal blood vessels. This is SERIOUS but treatable.',
            'what_next': [
                'URGENT: See an ophthalmologist within 2-4 weeks',
                'You will likely need laser treatment (photocoagulation)',
                'Very strict diabetes control required',
                'Regular monitoring every 2-3 months',
                'Treatment can prevent progression to blindness'
            ],
            'urgency': 'URGENT - See Specialist This Month',
            'color': 'red',
            'risk_level': 'High'
        },
        4: {
            'title': 'Proliferative Diabetic Retinopathy',
            'simple': 'Advanced stage requiring immediate treatment to prevent vision loss.',
            'what_it_means': 'Abnormal blood vessels are growing in your retina (neovascularization). These fragile vessels can bleed and cause retinal detachment, leading to blindness. This is a MEDICAL EMERGENCY.',
            'what_next': [
                'EMERGENCY: Contact ophthalmologist IMMEDIATELY',
                'You need urgent laser treatment or injections',
                'May require surgery if bleeding occurs',
                'Risk of permanent vision loss if untreated',
                'Treatment is still effective at this stage - DO NOT DELAY'
            ],
            'urgency': 'EMERGENCY - Immediate Action Required',
            'color': 'darkred',
            'risk_level': 'Very High'
        }
    }
    
    info = explanations[severity]
    
    # Add personalized risk factors
    risk_factors = []
    if hba1c > 9:
        risk_factors.append(f'Your HbA1c of {hba1c}% is very high (target: <7%). This significantly increases progression risk.')
    elif hba1c > 7:
        risk_factors.append(f'Your HbA1c of {hba1c}% is above target. Better control could slow progression.')
    
    if diabetes_duration > 15:
        risk_factors.append(f'Having diabetes for {diabetes_duration} years increases risk. Regular screening is essential.')
    
    info['personal_risk_factors'] = risk_factors if risk_factors else ['Continue good diabetes management to maintain low risk']
    
    # Add confidence explanation
    if confidence < 70:
        info['confidence_note'] = 'AI confidence is moderate. Recommend follow-up with specialist for confirmation.'
    else:
        info['confidence_note'] = 'AI has high confidence in this assessment.'
    
    return info

def predict_with_xai(image_path, clinical_data_json):
    """Main prediction with XAI"""
    try:
        clinical_data = json.loads(clinical_data_json)
        
        # Load model
        device = torch.device('cpu')
        model = ImageModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        img_tensor.requires_grad = True
        
        # Prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item() * 100
        
        # Generate Grad-CAM
        img_tensor.requires_grad = True
        gradcam_heatmap = generate_gradcam(model, img_tensor, predicted_class)
        
        # Create heatmap overlay
        heatmap_path = image_path.replace('.png', '_heatmap.png').replace('.jpg', '_heatmap.jpg')
        create_heatmap_overlay(image_path, gradcam_heatmap, heatmap_path)
        
        # Get patient-friendly explanation
        explanation = get_patient_friendly_explanation(
            predicted_class,
            confidence,
            clinical_data.get('hba1c', 7.0),
            clinical_data.get('diabetes_duration', 5)
        )
        
        # Build result
        result = {
            'stage': explanation['title'],
            'severity': predicted_class,
            'confidence': round(confidence, 1),
            'explanation': {
                'simple': explanation['simple'],
                'detailed': explanation['what_it_means'],
                'nextSteps': explanation['what_next'],
                'urgency': explanation['urgency'],
                'riskLevel': explanation['risk_level'],
                'personalizedRisks': explanation['personal_risk_factors'],
                'confidenceNote': explanation['confidence_note']
            },
            'visualization': {
                'heatmapPath': heatmap_path,
                'description': 'Red areas show where the AI detected abnormalities. Brighter red = stronger indication of diabetic changes.'
            },
            'probabilities': {
                'No DR': round(probs[0].item() * 100, 1),
                'Mild NPDR': round(probs[1].item() * 100, 1),
                'Moderate NPDR': round(probs[2].item() * 100, 1),
                'Severe NPDR': round(probs[3].item() * 100, 1),
                'Proliferative DR': round(probs[4].item() * 100, 1)
            },
            'technicalDetails': {
                'modelType': 'ResNet18 with Grad-CAM',
                'inputSize': '224x224',
                'gradientBasedAttention': True
            },
            'timestamp': clinical_data.get('timestamp', '')
        }
        
        return result
        
    except Exception as e:
        return {
            'error': 'PREDICTION_ERROR',
            'message': str(e),
            'stage': 'Error',
            'severity': -1
        }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(json.dumps({'error': 'Usage: python predict_xai.py <image_path> <clinical_json>'}))
        sys.exit(1)
    
    result = predict_with_xai(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))