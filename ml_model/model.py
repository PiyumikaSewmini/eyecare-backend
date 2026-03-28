# EyeCare+ ML Model - ResNet50 with Clinical Data Integration
# Save as: backend/ml_model/model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class DRMultimodalModel:
    """
    Multimodal Diabetic Retinopathy Detection Model
    Combines fundus image features with clinical data
    """
    
    def __init__(self, image_size=(224, 224, 3), num_classes=5):
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build fusion architecture:
        - Image branch: ResNet50 (pretrained on ImageNet)
        - Clinical branch: Dense neural network
        - Fusion: Concatenate and classify
        """
        
        # IMAGE BRANCH - ResNet50
        image_input = Input(shape=self.image_size, name='image_input')
        
        # Load pretrained ResNet50 (without top layer)
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=image_input,
            pooling=None
        )
        
        # Fine-tune last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Image feature extraction
        x = base_model.output
        x = GlobalAveragePooling2D(name='image_pool')(x)
        x = Dense(512, activation='relu', name='image_dense1')(x)
        x = Dropout(0.5, name='image_dropout1')(x)
        x = Dense(256, activation='relu', name='image_dense2')(x)
        image_features = Dropout(0.3, name='image_dropout2')(x)
        
        # CLINICAL BRANCH
        clinical_input = Input(shape=(7,), name='clinical_input')  # 7 clinical features
        
        # Clinical feature processing
        y = Dense(64, activation='relu', name='clinical_dense1')(clinical_input)
        y = Dropout(0.3, name='clinical_dropout1')(y)
        y = Dense(32, activation='relu', name='clinical_dense2')(y)
        clinical_features = Dropout(0.2, name='clinical_dropout2')(y)
        
        # FUSION LAYER
        # Concatenate image and clinical features
        combined = Concatenate(name='fusion_concat')([image_features, clinical_features])
        
        # Final classification layers
        z = Dense(128, activation='relu', name='fusion_dense1')(combined)
        z = Dropout(0.4, name='fusion_dropout')(z)
        z = Dense(64, activation='relu', name='fusion_dense2')(z)
        
        # Output layer - 5 classes
        # 0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR
        output = Dense(self.num_classes, activation='softmax', name='output')(z)
        
        # Create model
        self.model = Model(
            inputs=[image_input, clinical_input],
            outputs=output,
            name='DRMultimodalModel'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile model with optimizer and loss"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
    def get_summary(self):
        """Print model architecture"""
        return self.model.summary()


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess fundus image for model input
    
    Args:
        image_path: Path to fundus image
        target_size: Target image size
        
    Returns:
        Preprocessed image array
    """
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Load image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for ResNet50
    img_array = preprocess_input(img_array)
    
    return img_array


def preprocess_clinical_data(clinical_dict):
    """
    Normalize clinical data for model input
    
    Args:
        clinical_dict: Dictionary with clinical features
        {
            'age': int,
            'gender': str,  # 'male', 'female', 'other'
            'diabetes_duration': int (years),
            'hba1c': float (percentage),
            'systolic_bp': int (mmHg),
            'diastolic_bp': int (mmHg),
            'cholesterol': int (mg/dL)
        }
        
    Returns:
        Normalized feature array
    """
    
    # Gender encoding
    gender_map = {'male': 0, 'female': 1, 'other': 2}
    gender_encoded = gender_map.get(clinical_dict['gender'].lower(), 0)
    
    # Normalization ranges (based on medical standards)
    age_norm = clinical_dict['age'] / 100.0  # Normalize to 0-1
    gender_norm = gender_encoded / 2.0
    duration_norm = min(clinical_dict['diabetes_duration'] / 30.0, 1.0)  # Cap at 30 years
    hba1c_norm = clinical_dict['hba1c'] / 15.0  # Normalize (normal: 4-6%, diabetic: 7-15%)
    systolic_norm = clinical_dict['systolic_bp'] / 200.0  # Normalize (normal: 120, high: 180+)
    diastolic_norm = clinical_dict['diastolic_bp'] / 120.0  # Normalize (normal: 80, high: 110+)
    cholesterol_norm = clinical_dict['cholesterol'] / 400.0  # Normalize (normal: <200, high: 240+)
    
    # Create feature array
    features = np.array([
        age_norm,
        gender_norm,
        duration_norm,
        hba1c_norm,
        systolic_norm,
        diastolic_norm,
        cholesterol_norm
    ])
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)
    
    return features


def train_model(model, train_generator, val_generator, epochs=50):
    """
    Train the model
    
    Args:
        model: Compiled model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# Example usage
if __name__ == "__main__":
    # Create model
    print("Building multimodal DR detection model...")
    dr_model = DRMultimodalModel()
    model = dr_model.build_model()
    dr_model.compile_model()
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    dr_model.get_summary()
    
    print("\n" + "="*50)
    print("Model built successfully!")
    print("Total parameters:", model.count_params())
    print("="*50)
    
    # Save model architecture
    model.save('dr_model_architecture.h5')
    print("\nModel architecture saved to: dr_model_architecture.h5")
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Prepare dataset (EyePACS, Messidor-2, APTOS)")
    print("2. Create data generators")
    print("3. Train model using train_model()")
    print("4. Use predict.py for inference")
    print("="*50)