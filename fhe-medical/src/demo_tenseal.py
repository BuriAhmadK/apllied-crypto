import tenseal as ts
import numpy as np
import time
from typing import List, Dict, Tuple
from .data_simulator import MedicalDataSimulator

class TenSEALMedicalDemo:
    def __init__(self):
        self.context = None
        self.setup_encryption()
        self.performance_metrics = {}
        
    def setup_encryption(self):
        """Setup CKKS encryption context"""
        # CKKS scheme for floating-point operations
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
    def encrypt_batch_data(self, data_matrix: np.ndarray) -> List[ts.CKKSVector]:
        """Encrypt batch of patient data"""
        encrypted_records = []
        for i in range(data_matrix.shape[0]):
            encrypted_vector = ts.ckks_vector(self.context, data_matrix[i].tolist())
            encrypted_records.append(encrypted_vector)
        return encrypted_records
    
    def compute_encrypted_statistics(self, encrypted_records: List[ts.CKKSVector]) -> Dict:
        """Compute comprehensive statistics on encrypted data"""
        start_time = time.time()
        
        if not encrypted_records:
            return {}
        
        # Initialize with first record
        sum_vector = encrypted_records[0].copy()
        count = len(encrypted_records)
        
        # Sum all records
        for i in range(1, count):
            sum_vector += encrypted_records[i]
        
        encryption_time = time.time() - start_time
        
        self.performance_metrics['encryption_time'] = encryption_time
        self.performance_metrics['record_count'] = count
        
        return {
            'encrypted_sum': sum_vector,
            'count': count,
            'computation_time': encryption_time
        }
    
    def compute_risk_score(self, encrypted_records: List[ts.CKKSVector]) -> List[ts.CKKSVector]:
        """Compute encrypted diabetes risk score using formula"""
        start_time = time.time()
        
        risk_scores = []
        # Simplified risk formula: 0.3*glucose + 0.2*age + 0.2*bmi
        for encrypted_patient in encrypted_records:
            # Assuming indices: 0:age, 4:cholesterol, 5:glucose, 6:bmi
            glucose_component = encrypted_patient * 0.3  # This is simplified
            risk_score = glucose_component  # In reality, we'd need more complex operations
            risk_scores.append(risk_score)
        
        self.performance_metrics['risk_score_time'] = time.time() - start_time
        return risk_scores
    
    def demonstrate_operations(self, sample_data: List[float]):
        """Demonstrate various encrypted operations"""
        print("\n--- TenSEAL Operations Demonstration ---")
        
        # Encrypt sample data
        encrypted = ts.ckks_vector(self.context, sample_data)
        
        # Demonstrate operations
        encrypted_add = encrypted + encrypted
        encrypted_mult = encrypted * 2.5
        encrypted_poly = encrypted * encrypted + encrypted * 1.5 + 10
        
        # Decrypt results
        original_decrypted = encrypted.decrypt()
        add_decrypted = encrypted_add.decrypt()
        mult_decrypted = encrypted_mult.decrypt()
        poly_decrypted = encrypted_poly.decrypt()
        
        print(f"Original: {[round(x, 2) for x in original_decrypted]}")
        print(f"After Addition: {[round(x, 2) for x in add_decrypted]}")
        print(f"After Multiplication: {[round(x, 2) for x in mult_decrypted]}")
        print(f"After Polynomial: {[round(x, 2) for x in poly_decrypted]}")
        
        return {
            'operations': ['addition', 'multiplication', 'polynomial'],
            'results': [add_decrypted, mult_decrypted, poly_decrypted]
        }