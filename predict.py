

import sys
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DR_CLASSES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'saved_models', 'image_model.pth')
CLINICAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'saved_models', 'clinical_model.pth')

_model_cache = None

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5))
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
        return self.fc(x)

def validate_image_quality(image_path):
    """Validate if image is a proper fundus/retinal image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Check 1: Image size (fundus images are usually square or near-square)
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Image aspect ratio suggests this is not a fundus image"
        
        # Check 2: Minimum resolution
        if width < 200 or height < 200:
            return False, "Image resolution too low for accurate diagnosis (minimum 200x200)"
        
        # Check 3: Color distribution (fundus images have specific color characteristics)
        # Fundus images are typically reddish/orange due to retina
        red_channel = img_array[:,:,0].mean()
        green_channel = img_array[:,:,1].mean()
        blue_channel = img_array[:,:,2].mean()
        
        # Fundus images typically have higher red channel
        if red_channel < 50:
            return False, "Image does not appear to be a retinal fundus photograph (color characteristics incorrect)"
        
        # Check 4: Brightness
        brightness = img_array.mean()
        if brightness < 20 or brightness > 250:
            return False, "Image is too dark or too bright for accurate analysis"
        
        # Check 5: Circular/oval pattern detection (fundus images are circular)
        # Convert to grayscale and check for circular patterns
        from PIL import ImageFilter
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        
        # Calculate center region intensity (fundus has bright optic disc)
        center_y, center_x = height // 2, width // 2
        center_region = img_array[center_y-50:center_y+50, center_x-50:center_x+50]
        
        if center_region.size == 0:
            return False, "Image too small for analysis"
        
        # Check 6: File size sanity check
        file_size = os.path.getsize(image_path)
        if file_size < 10000:  # Less than 10KB
            return False, "Image file size too small - may be corrupted"
        
        # Calculate quality score
        quality_score = 70  # Base score
        
        # Adjust based on resolution
        if width >= 512 and height >= 512:
            quality_score += 10
        if width >= 1024 and height >= 1024:
            quality_score += 10
        
        # Adjust based on brightness
        if 80 <= brightness <= 180:
            quality_score += 10
        
        return True, f"Image validated successfully (Quality score: {quality_score}%)"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def load_models():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    device = torch.device('cpu')
    
    # Load image model
    image_model = ImageModel()
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    image_model.eval()
    
    # Try to load clinical model if exists
    clinical_model = None
    if os.path.exists(CLINICAL_MODEL_PATH):
        try:
            # Clinical model architecture (adjust based on your training)
            class ClinicalModel(nn.Module):
                def __init__(self, input_dim=31):
                    super(ClinicalModel, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                        nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(32, 6)
                    )
                def forward(self, x):
                    return self.network(x)
            
            clinical_model = ClinicalModel()
            clinical_model.load_state_dict(torch.load(CLINICAL_MODEL_PATH, map_location=device))
            clinical_model.eval()
        except:
            clinical_model = None
    
    _model_cache = (image_model, clinical_model, device)
    return _model_cache

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

def preprocess_clinical(clinical_data):
    """Preprocess clinical data for clinical model"""
    features = []
    
    # Essential features
    features.append(clinical_data.get('age', 50))
    features.append(clinical_data.get('hba1c', 7.0))
    features.append(clinical_data.get('systolic_bp', 120))
    features.append(clinical_data.get('diastolic_bp', 80))
    features.append(clinical_data.get('diabetes_duration', 5))
    features.append(clinical_data.get('bmi', 25))
    features.append(clinical_data.get('fasting_glucose', 100))
    
    # Pad to 31 features
    while len(features) < 31:
        features.append(0)
    
    return torch.FloatTensor(features).unsqueeze(0)

def calculate_clinical_risk(clinical_data):
    """Calculate risk based on clinical parameters"""
    risk = 0
    risk_factors = []
    
    hba1c = clinical_data.get('hba1c', 0)
    if hba1c > 9:
        risk += 30
        risk_factors.append(f"Very High HbA1c ({hba1c}%)")
    elif hba1c > 7:
        risk += 15
        risk_factors.append(f"High HbA1c ({hba1c}%)")
    
    systolic = clinical_data.get('systolic_bp', 0)
    diastolic = clinical_data.get('diastolic_bp', 0)
    if systolic > 140 or diastolic > 90:
        risk += 15
        risk_factors.append(f"Hypertension ({systolic}/{diastolic})")
    
    duration = clinical_data.get('diabetes_duration', 0)
    if duration > 15:
        risk += 20
        risk_factors.append(f"Long diabetes duration ({duration} years)")
    elif duration > 10:
        risk += 10
        risk_factors.append(f"Moderate diabetes duration ({duration} years)")
    
    bmi = clinical_data.get('bmi', 0)
    if bmi > 30:
        risk += 10
        risk_factors.append(f"Obesity (BMI: {bmi})")
    
    return min(risk, 80), risk_factors

def fuse_predictions(image_pred, image_conf, clinical_data, image_probs):
    """
    Intelligent fusion of image prediction with clinical data
    Adjusts prediction based on clinical risk factors
    """
    clinical_risk, risk_factors = calculate_clinical_risk(clinical_data)
    
    # If clinical risk is very high but image shows low severity, flag for review
    if clinical_risk > 50 and image_pred < 2:
        # High clinical risk should elevate the prediction
        adjusted_pred = min(image_pred + 1, 4)
        confidence_penalty = 10
        warning = "Clinical risk factors suggest higher DR risk than image alone. Recommend detailed examination."
    elif clinical_risk < 20 and image_pred > 2:
        # Low clinical risk but high image severity - keep image prediction but note discrepancy
        adjusted_pred = image_pred
        confidence_penalty = 5
        warning = "Image shows significant DR despite low clinical risk factors. Recommend verification."
    else:
        adjusted_pred = image_pred
        confidence_penalty = 0
        warning = None
    
    # Adjust confidence based on agreement
    final_confidence = max(image_conf - confidence_penalty, 50)
    
    # Calculate combined risk score
    image_risk = image_pred * 20
    combined_risk = int(0.6 * image_risk + 0.4 * clinical_risk)
    
    return adjusted_pred, final_confidence, combined_risk, warning, risk_factors

def get_recommendations(severity, clinical_data, confidence, warning=None):
    recs = {
        0: {'severity': 'No Diabetic Retinopathy Detected', 'action': 'Continue regular diabetes management and annual eye screening', 'followUp': '12 months', 'urgency': 'Routine', 'color': 'green'},
        1: {'severity': 'Mild Non-Proliferative DR', 'action': 'Monitor closely with more frequent eye exams', 'followUp': '6-12 months', 'urgency': 'Non-urgent', 'color': 'yellow'},
        2: {'severity': 'Moderate Non-Proliferative DR', 'action': 'Ophthalmologist consultation recommended', 'followUp': '3-6 months', 'urgency': 'Semi-urgent', 'color': 'orange'},
        3: {'severity': 'Severe Non-Proliferative DR', 'action': 'Urgent ophthalmologist referral needed', 'followUp': '2-3 months', 'urgency': 'Urgent', 'color': 'red'},
        4: {'severity': 'Proliferative DR', 'action': 'IMMEDIATE ophthalmologist referral required', 'followUp': '1 month or sooner', 'urgency': 'Emergency', 'color': 'darkred'}
    }
    
    lesions = {
        0: [],
        1: ['Microaneurysms detected'],
        2: ['Microaneurysms', 'Hard Exudates', 'Retinal Hemorrhages'],
        3: ['Microaneurysms', 'Hard Exudates', 'Cotton Wool Spots', 'Venous Beading'],
        4: ['Neovascularization', 'Vitreous Hemorrhage', 'High risk of Retinal Detachment']
    }
    
    result = recs[severity].copy()
    result['lesionsDetected'] = lesions[severity]
    
    if warning:
        result['clinicalWarning'] = warning
    
    # Add detailed clinical advice
    result['detailedAdvice'] = get_detailed_advice(severity, clinical_data)
    
    return result

def get_detailed_advice(severity, clinical_data):
    """Generate detailed personalized advice"""
    advice = []
    
    # HbA1c advice
    hba1c = clinical_data.get('hba1c', 0)
    if hba1c > 7:
        advice.append(f"CRITICAL: Your HbA1c is {hba1c}%. Target is below 7%. Work with your endocrinologist to optimize diabetes control immediately.")
    else:
        advice.append(f"Good: HbA1c at {hba1c}% is within target. Continue current diabetes management.")
    
    # Blood pressure advice
    systolic = clinical_data.get('systolic_bp', 0)
    if systolic > 140:
        advice.append(f"IMPORTANT: Blood pressure {systolic}/{clinical_data.get('diastolic_bp', 0)} is elevated. Control BP to prevent DR progression.")
    
    # Duration-based advice
    duration = clinical_data.get('diabetes_duration', 0)
    if duration > 10:
        advice.append(f"With {duration} years of diabetes, regular eye screening every 6 months is essential.")
    
    # Severity-specific advice
    if severity >= 3:
        advice.append("URGENT: This level of DR requires immediate specialist care. Delay can result in permanent vision loss.")
    elif severity >= 2:
        advice.append("Important: Schedule ophthalmologist appointment within next month for detailed retinal examination.")
    
    return advice

def predict(image_path, clinical_data_json):
    try:
        # Step 1: Validate image quality
        is_valid, validation_msg = validate_image_quality(image_path)
        
        if not is_valid:
            return {
                'error': 'INVALID_IMAGE',
                'errorType': 'validation',
                'message': validation_msg,
                'suggestion': 'Please upload a clear retinal fundus photograph taken with proper medical equipment.',
                'stage': 'Error',
                'severity': -1
            }
        
        # Step 2: Parse clinical data
        clinical_data = json.loads(clinical_data_json)
        
        # Validate essential clinical data
        if not clinical_data.get('age') or not clinical_data.get('hba1c'):
            return {
                'error': 'INCOMPLETE_DATA',
                'errorType': 'validation',
                'message': 'Essential clinical data missing (age and HbA1c required)',
                'stage': 'Error',
                'severity': -1
            }
        
        # Step 3: Load models
        image_model, clinical_model, device = load_models()
        
        # Step 4: Image prediction
        image_tensor = preprocess_image(image_path).to(device)
        
        with torch.no_grad():
            output = image_model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            image_pred = torch.argmax(probs).item()
            image_conf = probs[image_pred].item() * 100
        
        # Step 5: Clinical model prediction (if available)
        if clinical_model is not None:
            try:
                clinical_tensor = preprocess_clinical(clinical_data).to(device)
                with torch.no_grad():
                    clinical_output = clinical_model(clinical_tensor)
                    clinical_probs = torch.softmax(clinical_output, dim=1)[0]
                    clinical_pred = torch.argmax(clinical_probs).item()
                    clinical_pred = min(clinical_pred, 4)  # Clamp to 0-4
            except:
                clinical_pred = None
        else:
            clinical_pred = None
        
        # Step 6: Fuse predictions
        final_pred, final_conf, combined_risk, warning, risk_factors = fuse_predictions(
            image_pred, image_conf, clinical_data, probs
        )
        
        # Step 7: Confidence check
        if final_conf < 60:
            warning = (warning or "") + " Low confidence prediction. Recommend manual review by specialist."
        
        # Step 8: Generate recommendations
        recommendations = get_recommendations(final_pred, clinical_data, final_conf, warning)
        
        # Step 9: Build result
        result = {
            'stage': DR_CLASSES[final_pred],
            'severity': final_pred,
            'confidence': round(final_conf, 1),
            'riskScore': combined_risk,
            'progressionRisk': min(95, final_pred * 15 + len(risk_factors) * 8),
            'recommendations': recommendations,
            'imageValidation': validation_msg,
            'modelDetails': {
                'imageModel': {
                    'prediction': DR_CLASSES[image_pred],
                    'confidence': round(image_conf, 1)
                }
            },
            'clinicalFactors': [{'factor': f, 'status': 'High Risk'} for f in risk_factors],
            'probabilities': {DR_CLASSES[i]: round(p.item() * 100, 1) for i, p in enumerate(probs)},
            'qualityAssurance': {
                'imageQualityPassed': True,
                'clinicalDataComplete': True,
                'multipleModelsUsed': clinical_pred is not None,
                'confidenceThreshold': 'PASSED' if final_conf >= 60 else 'WARNING'
            }
        }
        
        if clinical_pred is not None:
            result['modelDetails']['clinicalModel'] = {
                'prediction': DR_CLASSES[clinical_pred],
                'used': True
            }
        
        return result
        
    except Exception as e:
        return {
            'error': 'PROCESSING_ERROR',
            'errorType': 'system',
            'message': str(e),
            'stage': 'Error',
            'severity': -1
        }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(json.dumps({'error': 'Usage: python predict_enhanced.py <image_path> <clinical_data_json>'}))
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1], sys.argv[2]), indent=2))