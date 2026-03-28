
import sys
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

DR_CLASSES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}

# Plain language labels for non-medical users
PLAIN_LABELS = {
    0: "Your Eyes Look Healthy",
    1: "Very Early Warning Signs",
    2: "Moderate Eye Changes",
    3: "Serious Eye Damage",
    4: "Advanced & Urgent"
}

PLAIN_DESCRIPTIONS = {
    0: "No signs of diabetic eye damage were found. Your retina looks healthy right now.",
    1: "Tiny changes are starting in the blood vessels of your eye. You don't need treatment yet, but regular check-ups are important.",
    2: "The blood vessels in your eye are showing clear changes. This needs attention from an eye doctor soon.",
    3: "Significant damage to the blood vessels in your eye has been detected. Please see an eye specialist urgently.",
    4: "Advanced damage with new abnormal blood vessel growth. This is a medical emergency — please seek immediate specialist care."
}

SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH    = os.path.join(SCRIPT_DIR, 'saved_models', 'image_model.pth')
CLINICAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'saved_models', 'clinical_model.pth')

_model_cache = None


# =============================================================================
# VALIDATION GATE
# =============================================================================

def _validate_is_fundus(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return False, 0, "Cannot read image file."

    h, w  = img.shape[:2]
    score = 50
    issues = []

    if w < 100 or h < 100:
        return False, 5, "Image is too small to be a retinal fundus photograph."

    b_ch, g_ch, r_ch = cv2.split(img)
    avg_r = float(np.mean(r_ch))
    avg_g = float(np.mean(g_ch))
    avg_b = float(np.mean(b_ch))

    if avg_b > avg_r + 35:
        return False, 8, (
            "This does NOT appear to be a retinal fundus photograph. "
            f"The image is predominantly blue (R={avg_r:.0f}, B={avg_b:.0f}). "
            "Please upload a proper fundus photograph."
        )
    if avg_g > avg_r + 30:
        return False, 8, (
            "This does NOT appear to be a retinal fundus photograph. "
            f"The image is predominantly green (R={avg_r:.0f}, G={avg_g:.0f}). "
            "Please upload a proper fundus photograph."
        )

    if avg_b > avg_r + 18:
        issues.append("Blue-leaning colour")
        score -= 20
    elif avg_r > avg_b + 20:
        score += 20
    elif avg_r > avg_b + 5:
        score += 8

    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges      = cv2.Canny(gray, 100, 200)
    edge_ratio = float(np.sum(edges > 0)) / (h * w)

    if edge_ratio > 0.15:
        return False, 12, (
            "This does NOT appear to be a retinal fundus photograph. "
            f"Too many sharp edges ({edge_ratio*100:.1f}% of pixels). "
            "Please upload a proper fundus photograph."
        )
    if edge_ratio > 0.09:
        issues.append(f"High edge density ({edge_ratio*100:.1f}%)")
        score -= 12
    elif edge_ratio < 0.05:
        score += 12

    bright_std = float(np.std(gray))
    if bright_std > 115:
        issues.append(f"Extreme contrast (σ={bright_std:.0f})")
        score -= 10

    cx, cy = w // 2, h // 2
    mask   = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (int(w*0.40), int(h*0.40)), 0, 0, 360, 255, -1)
    c_mean = float(np.mean(gray[mask == 255]))
    b_mean = float(np.mean(gray[mask == 0]))
    if c_mean > b_mean + 10:
        score += 14
    elif c_mean < b_mean - 20:
        issues.append("Brighter at edges than centre")
        score -= 8

    score = max(0, min(100, score))

    if score < 30:
        return False, score, (
            "This does NOT appear to be a retinal fundus photograph.\n"
            "Please upload a retinal fundus photograph taken with ophthalmic equipment.\n"
            "Not accepted: buildings, animals, selfies, X-rays, screenshots."
        )

    return True, score, f"Image accepted (score: {score}/100)."


# =============================================================================
# CLINICAL ALERTS — warn about dangerous values immediately
# =============================================================================

def get_clinical_alerts(clinical_data: dict) -> list:
    """Returns list of urgent alert messages for dangerous clinical values."""
    alerts = []

    hba1c = clinical_data.get('hba1c', 0)
    if hba1c >= 10:
        alerts.append({
            "level": "critical",
            "field": "HbA1c",
            "message": f"⚠️ CRITICAL: Your HbA1c of {hba1c}% is dangerously high. Please see your doctor immediately — this level greatly increases the risk of serious complications.",
            "value": hba1c
        })
    elif hba1c >= 8:
        alerts.append({
            "level": "warning",
            "field": "HbA1c",
            "message": f"⚠️ Your HbA1c of {hba1c}% is too high. The target is below 7%. Please consult your doctor about improving your blood sugar control.",
            "value": hba1c
        })

    systolic  = clinical_data.get('systolic_bp', 0)
    diastolic = clinical_data.get('diastolic_bp', 0)
    if systolic >= 180 or diastolic >= 110:
        alerts.append({
            "level": "critical",
            "field": "Blood Pressure",
            "message": f"⚠️CRITICAL: Your blood pressure of {systolic}/{diastolic} mmHg is dangerously high (hypertensive crisis). Seek medical attention immediately.",
            "value": f"{systolic}/{diastolic}"
        })
    elif systolic >= 140 or diastolic >= 90:
        alerts.append({
            "level": "warning",
            "field": "Blood Pressure",
            "message": f"⚠️ Your blood pressure of {systolic}/{diastolic} mmHg is high. High blood pressure significantly worsens diabetic eye disease. Please see your doctor.",
            "value": f"{systolic}/{diastolic}"
        })

    duration = clinical_data.get('diabetes_duration', 0)
    if duration >= 20:
        alerts.append({
            "level": "warning",
            "field": "Diabetes Duration",
            "message": f"ℹ️ You have had diabetes for {duration} years. Long-term diabetes significantly increases eye complication risk — regular screening is essential.",
            "value": duration
        })

    glucose = clinical_data.get('fasting_glucose', 0)
    if glucose >= 250:
        alerts.append({
            "level": "critical",
            "field": "Fasting Glucose",
            "message": f"⚠️ CRITICAL: Your fasting glucose of {glucose} mg/dL is very high. Please contact your doctor urgently.",
            "value": glucose
        })
    elif glucose >= 180:
        alerts.append({
            "level": "warning",
            "field": "Fasting Glucose",
            "message": f"⚠️ Your fasting glucose of {glucose} mg/dL is above normal. Please discuss this with your doctor.",
            "value": glucose
        })

    cholesterol = clinical_data.get('cholesterol', 0)
    if cholesterol >= 280:
        alerts.append({
            "level": "warning",
            "field": "Cholesterol",
            "message": f"⚠️ Your cholesterol of {cholesterol} mg/dL is high. High cholesterol can worsen diabetic eye disease. Please see your doctor.",
            "value": cholesterol
        })

    bmi = clinical_data.get('bmi', 0)
    if bmi >= 40:
        alerts.append({
            "level": "warning",
            "field": "BMI",
            "message": f"⚠️ Your BMI of {bmi} indicates severe obesity, which increases diabetes complication risk. Please consult your doctor.",
            "value": bmi
        })

    return alerts


# =============================================================================
# RISK SCORE — image + all clinical factors combined
# =============================================================================

def calculate_combined_risk(image_severity: int, image_conf: float, clinical_data: dict) -> dict:
    """
    Returns a risk score 0-100 combining image findings + clinical data.
    Also returns component breakdown and plain-language risk level.
    """
    # Image component (0-60 points — primary driver)
    image_risk = {0: 0, 1: 12, 2: 28, 3: 45, 4: 60}[image_severity]

    # Clinical component (0-40 points)
    clinical_risk = 0
    risk_factors  = []

    hba1c = clinical_data.get('hba1c', 0)
    if hba1c >= 10:
        clinical_risk += 12; risk_factors.append(f"Very high HbA1c ({hba1c}%)")
    elif hba1c >= 8:
        clinical_risk += 8;  risk_factors.append(f"High HbA1c ({hba1c}%)")
    elif hba1c >= 7:
        clinical_risk += 4;  risk_factors.append(f"Slightly elevated HbA1c ({hba1c}%)")

    duration = clinical_data.get('diabetes_duration', 0)
    if duration >= 20:
        clinical_risk += 10; risk_factors.append(f"Long diabetes history ({duration} yrs)")
    elif duration >= 10:
        clinical_risk += 6;  risk_factors.append(f"Moderate diabetes duration ({duration} yrs)")
    elif duration >= 5:
        clinical_risk += 3;  risk_factors.append(f"Diabetes for {duration} yrs")

    systolic  = clinical_data.get('systolic_bp', 0)
    diastolic = clinical_data.get('diastolic_bp', 0)
    if systolic >= 160 or diastolic >= 100:
        clinical_risk += 10; risk_factors.append(f"Severely high BP ({systolic}/{diastolic})")
    elif systolic >= 140 or diastolic >= 90:
        clinical_risk += 6;  risk_factors.append(f"High blood pressure ({systolic}/{diastolic})")

    glucose = clinical_data.get('fasting_glucose', 0)
    if glucose >= 250:
        clinical_risk += 5; risk_factors.append(f"Very high fasting glucose ({glucose} mg/dL)")
    elif glucose >= 180:
        clinical_risk += 3; risk_factors.append(f"High fasting glucose ({glucose} mg/dL)")

    cholesterol = clinical_data.get('cholesterol', 0)
    if cholesterol >= 240:
        clinical_risk += 3; risk_factors.append(f"High cholesterol ({cholesterol} mg/dL)")

    bmi = clinical_data.get('bmi', 0)
    if bmi >= 35:
        clinical_risk += 3; risk_factors.append(f"High BMI ({bmi})")
    elif bmi >= 30:
        clinical_risk += 1; risk_factors.append(f"Overweight (BMI {bmi})")

    clinical_risk = min(clinical_risk, 40)
    total_risk    = min(image_risk + clinical_risk, 100)

    # Plain language risk level
    if total_risk <= 15:
        risk_level = "Low Risk"
        risk_color = "green"
        risk_plain = "Your risk is currently low. Keep up good diabetes management."
    elif total_risk <= 35:
        risk_level = "Moderate Risk"
        risk_color = "yellow"
        risk_plain = "Your risk is moderate. Regular monitoring and better blood sugar control can reduce this."
    elif total_risk <= 60:
        risk_level = "High Risk"
        risk_color = "orange"
        risk_plain = "Your risk is high. Please see an eye specialist and work closely with your diabetes team."
    else:
        risk_level = "Very High Risk"
        risk_color = "red"
        risk_plain = "Your risk is very high. Immediate specialist care is strongly recommended."

    return {
        "score":          total_risk,
        "imageComponent": image_risk,
        "clinicalComponent": clinical_risk,
        "level":          risk_level,
        "color":          risk_color,
        "plainSummary":   risk_plain,
        "riskFactors":    risk_factors,
    }


# =============================================================================
# MODEL
# =============================================================================

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5)
        )
        for attr in ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool','fc']:
            setattr(self, attr, getattr(resnet, attr))

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


def load_models():
    global _model_cache
    if _model_cache: return _model_cache
    device = torch.device('cpu')
    m = ImageModel()
    m.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    m.eval()
    _model_cache = (m, None, device)
    return _model_cache


def preprocess_image(path):
    t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t(Image.open(path).convert('RGB')).unsqueeze(0)


# =============================================================================
# RECOMMENDATIONS — plain language
# =============================================================================

def get_recommendations(severity, clinical_data, risk_info):
    base = {
        0: {
            "urgency":   "Routine",
            "urgencyColor": "green",
            "action":    "Continue annual eye check-ups and maintain your current diabetes management.",
            "followUp":  "In 12 months",
            "plainSteps": [
                "Keep your blood sugar at target levels (HbA1c below 7%)",
                "Have a full eye exam every 12 months",
                "Keep your blood pressure below 130/80 mmHg",
                "Maintain a healthy diet and exercise regularly",
            ]
        },
        1: {
            "urgency":   "Monitor Closely",
            "urgencyColor": "yellow",
            "action":    "Early changes have been found. You do not need treatment yet, but closer monitoring is important.",
            "followUp":  "In 6–12 months",
            "plainSteps": [
                "See an eye doctor within the next 6 months for a closer look",
                "Work on bringing your blood sugar closer to target",
                "Monitor your blood pressure and keep it controlled",
                "Do not smoke — smoking makes eye damage much worse",
                "Report any vision changes to your doctor immediately",
            ]
        },
        2: {
            "urgency":   "See a Specialist Soon",
            "urgencyColor": "orange",
            "action":    "Noticeable damage has been found. Please book an appointment with an eye specialist soon.",
            "followUp":  "Within 3–6 months",
            "plainSteps": [
                "Book an ophthalmologist appointment within the next 2 months",
                "Strictly control your blood sugar — every point counts now",
                "Control blood pressure — this is critical to slow progression",
                "You may need laser treatment; your doctor will advise",
                "Tell your eye doctor immediately if you notice any blurring",
            ]
        },
        3: {
            "urgency":   "Urgent — Act Now",
            "urgencyColor": "red",
            "action":    "Serious damage has been detected. Please see an eye specialist as soon as possible — within weeks.",
            "followUp":  "Within 2–4 weeks",
            "plainSteps": [
                "Call an ophthalmologist NOW and request an urgent appointment",
                "Laser treatment (photocoagulation) will likely be recommended",
                "Achieve the tightest possible blood sugar control immediately",
                "Strict blood pressure management is essential",
                "Do NOT delay — untreated severe DR can lead to vision loss",
            ]
        },
        4: {
            "urgency":   "EMERGENCY",
            "urgencyColor": "red",
            "action":    "Advanced damage with abnormal new blood vessel growth has been detected. This is an emergency. Seek specialist care immediately.",
            "followUp":  "This week",
            "plainSteps": [
                "Go to an eye hospital or ophthalmology emergency TODAY",
                "Urgent laser treatment or injections into the eye are likely needed",
                "There is a risk of sudden vision loss from bleeding — act now",
                "Surgery may be required — your specialist will assess this",
                "Do NOT wait. Every day of delay increases the risk of permanent vision loss",
            ]
        }
    }
    rec = base[severity].copy()

    # Add personalised advice based on clinical data
    personal = []
    hba1c = clinical_data.get('hba1c', 0)
    if hba1c >= 8:
        personal.append(f"Your HbA1c of {hba1c}% is significantly above target. Improving this is the single most important thing you can do to protect your eyes.")
    elif hba1c >= 7:
        personal.append(f"Your HbA1c of {hba1c}% is slightly above the target of 7%. Small improvements will have a real impact on your eye health.")

    duration = clinical_data.get('diabetes_duration', 0)
    if duration >= 15:
        personal.append(f"With {duration} years of diabetes, your cumulative risk is higher. Eye screening every 6 months is strongly recommended regardless of current findings.")

    systolic = clinical_data.get('systolic_bp', 0)
    if systolic >= 140:
        personal.append(f"Your blood pressure of {systolic} mmHg is high. High blood pressure doubles the speed of diabetic eye disease progression.")

    rec['personalisedAdvice'] = personal
    rec['riskLevel']  = risk_info['level']
    rec['riskColor']  = risk_info['color']
    rec['riskPlain']  = risk_info['plainSummary']
    return rec


# =============================================================================
# MAIN
# =============================================================================

def predict(image_path, clinical_data_json):
    try:
        # Gate 1: image
        is_valid, val_score, val_message = _validate_is_fundus(image_path)
        if not is_valid:
            return {
                'error': 'INVALID_IMAGE', 'errorType': 'validation',
                'message': val_message, 'validationScore': val_score,
                'stage': 'Error', 'severity': -1,
            }

        # Gate 2: clinical
        clinical_data = json.loads(clinical_data_json)

        # Diabetes duration is REQUIRED
        if not clinical_data.get('age') or not clinical_data.get('hba1c'):
            return {
                'error': 'INCOMPLETE_DATA', 'errorType': 'validation',
                'message': 'Age and HbA1c are required.',
                'stage': 'Error', 'severity': -1,
            }
        if not clinical_data.get('diabetes_duration'):
            return {
                'error': 'INCOMPLETE_DATA', 'errorType': 'validation',
                'message': 'Diabetes duration is required for diabetic retinopathy screening. Please enter how long you have had diabetes.',
                'stage': 'Error', 'severity': -1,
            }

        # Clinical alerts
        clinical_alerts = get_clinical_alerts(clinical_data)

        # Run model
        image_model, _, device = load_models()
        image_tensor = preprocess_image(image_path).to(device)

        with torch.no_grad():
            output     = image_model(image_tensor)
            probs      = torch.softmax(output, dim=1)[0]
            image_pred = torch.argmax(probs).item()
            image_conf = probs[image_pred].item() * 100

        # Fuse: if clinical risk is very high and image shows mild, bump one level
        hba1c    = clinical_data.get('hba1c', 0)
        duration = clinical_data.get('diabetes_duration', 0)
        systolic = clinical_data.get('systolic_bp', 0)
        high_clinical = (hba1c >= 9) or (duration >= 15 and hba1c >= 7.5) or (systolic >= 160)

        final_pred = image_pred
        if high_clinical and image_pred < 2:
            final_pred = min(image_pred + 1, 4)
        final_conf = max(image_conf - (5 if high_clinical and final_pred != image_pred else 0), 50)

        # Risk score
        risk_info = calculate_combined_risk(final_pred, final_conf, clinical_data)

        # Recommendations
        recommendations = get_recommendations(final_pred, clinical_data, risk_info)

        return {
            'stage':         DR_CLASSES[final_pred],
            'plainLabel':    PLAIN_LABELS[final_pred],
            'plainDesc':     PLAIN_DESCRIPTIONS[final_pred],
            'severity':      final_pred,
            'confidence':    round(final_conf, 1),
            'riskScore':     risk_info['score'],
            'riskInfo':      risk_info,
            'recommendations': recommendations,
            'clinicalAlerts':  clinical_alerts,
            'imageValidation': val_message,
            'probabilities': {DR_CLASSES[i]: round(p.item() * 100, 1) for i, p in enumerate(probs)},
            'clinicalFactors': [{'factor': f, 'status': 'Risk Factor'} for f in risk_info['riskFactors']],
            'modelDetails': {
                'imagePrediction': DR_CLASSES[image_pred],
                'imageConfidence': round(image_conf, 1),
                'clinicalAdjusted': final_pred != image_pred,
            },
            'qualityAssurance': {
                'imageValidationScore': val_score,
                'imageQualityPassed':   True,
                'clinicalDataComplete': True,
                'confidenceThreshold':  'PASSED' if final_conf >= 60 else 'WARNING',
            },
        }

    except Exception as e:
        return {
            'error': 'PROCESSING_ERROR', 'errorType': 'system',
            'message': str(e), 'stage': 'Error', 'severity': -1,
        }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(json.dumps({'error': 'Usage: python predict.py <image_path> <clinical_json>'}))
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1], sys.argv[2]), indent=2))