import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random

class MedicalDataSimulator:
    """Simulates realistic medical data for FHE demonstrations"""
    
    @staticmethod
    def generate_patient_records(num_patients: int) -> List[Dict]:
        """Generate synthetic but realistic patient data"""
        records = []
        
        for i in range(num_patients):
            # Realistic medical parameter ranges
            age = random.randint(18, 90)
            is_male = random.random() > 0.5
            
            # Generate correlated medical data
            bp_systolic = max(90, min(180, int(np.random.normal(120, 20))))
            bp_diastolic = max(60, min(120, int(np.random.normal(80, 15))))
            
            cholesterol = max(150, min(300, int(np.random.normal(200, 40))))
            glucose = max(70, min(300, int(np.random.normal(100, 30))))
            bmi = max(18, min(45, np.random.normal(25, 5)))
            
            # Simulate disease probabilities based on risk factors
            diabetes_risk = min(0.8, (glucose - 70) / 400 + (bmi - 25) / 100 + age / 500)
            heart_disease_risk = min(0.7, (cholesterol - 150) / 600 + (bp_systolic - 120) / 300 + age / 400)
            
            has_diabetes = 1.0 if random.random() < diabetes_risk else 0.0
            has_heart_disease = 1.0 if random.random() < heart_disease_risk else 0.0
            
            record = {
                'patient_id': f"P{10000 + i}",
                'age': age,
                'is_male': 1.0 if is_male else 0.0,
                'bp_systolic': bp_systolic,
                'bp_diastolic': bp_diastolic,
                'cholesterol': cholesterol,
                'glucose': glucose,
                'bmi': bmi,
                'has_diabetes': has_diabetes,
                'has_heart_disease': has_heart_disease
            }
            records.append(record)
            
        return records
    
    @staticmethod
    def records_to_feature_matrix(records: List[Dict]) -> np.ndarray:
        """Convert records to feature matrix for encryption"""
        features = []
        for record in records:
            feature_vector = [
                record['age'],
                record['is_male'],
                record['bp_systolic'],
                record['bp_diastolic'], 
                record['cholesterol'],
                record['glucose'],
                record['bmi'],
                record['has_diabetes'],
                record['has_heart_disease']
            ]
            features.append(feature_vector)
        return np.array(features)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        return ['age', 'is_male', 'bp_systolic', 'bp_diastolic', 'cholesterol', 
                'glucose', 'bmi', 'has_diabetes', 'has_heart_disease']