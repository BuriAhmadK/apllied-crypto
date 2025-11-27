from phe import paillier
import numpy as np
import time
from typing import List, Dict, Tuple
from .data_simulator import MedicalDataSimulator

class PaillierMedicalDemo:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.performance_metrics = {}
    
    def encrypt_single_value(self, value: float) -> paillier.EncryptedNumber:
        """Encrypt single value using Paillier"""
        # Paillier works with integers, so we scale floats
        return self.public_key.encrypt(int(value * 100))
    
    def encrypt_patient_data(self, records: List[Dict], selected_field: str) -> List[paillier.EncryptedNumber]:
        """Encrypt a specific field from patient records"""
        start_time = time.time()
        
        encrypted_values = []
        for record in records:
            value = record[selected_field]
            encrypted_val = self.encrypt_single_value(value)
            encrypted_values.append(encrypted_val)
        
        self.performance_metrics['encryption_time'] = time.time() - start_time
        self.performance_metrics['record_count'] = len(records)
        
        return encrypted_values
    
    def compute_average(self, encrypted_values: List[paillier.EncryptedNumber]) -> Tuple:
        """Compute average from encrypted values"""
        start_time = time.time()
        
        if not encrypted_values:
            return None, 0
        
        # Sum all encrypted values
        encrypted_sum = encrypted_values[0]
        for encrypted_val in encrypted_values[1:]:
            encrypted_sum += encrypted_val
        
        count = len(encrypted_values)
        
        self.performance_metrics['computation_time'] = time.time() - start_time
        
        return encrypted_sum, count
    
    def compute_variance(self, encrypted_values: List[paillier.EncryptedNumber], mean: float) -> paillier.EncryptedNumber:
        """Compute variance using Paillier properties"""
        start_time = time.time()
        
        # Variance = E[X^2] - E[X]^2
        # We can compute E[X^2] by encrypting squares
        encrypted_squares = []
        for enc_val in encrypted_values:
            # For variance computation, we'd need different approach
            # This is simplified for demo
            pass
        
        self.performance_metrics['variance_time'] = time.time() - start_time
        return None
    
    def demonstrate_operations(self, sample_values: List[float]):
        """Demonstrate Paillier operations"""
        print("\n--- Paillier Operations Demonstration ---")
        
        # Encrypt sample values
        encrypted_values = [self.encrypt_single_value(val) for val in sample_values]
        
        # Demonstrate addition
        encrypted_sum = encrypted_values[0]
        for ev in encrypted_values[1:]:
            encrypted_sum += ev
        
        # Demonstrate scalar multiplication
        encrypted_scaled = encrypted_values[0] * 3
        
        # Decrypt results
        original_decrypted = [self.private_key.decrypt(ev) / 100 for ev in encrypted_values]
        sum_decrypted = self.private_key.decrypt(encrypted_sum) / 100
        scaled_decrypted = self.private_key.decrypt(encrypted_scaled) / 100
        
        print(f"Original values: {sample_values}")
        print(f"Encrypted Sum: {sum_decrypted:.2f} (Actual: {sum(sample_values):.2f})")
        print(f"Encrypted Scaled: {scaled_decrypted:.2f} (Actual: {sample_values[0] * 3:.2f})")
        
        return {
            'operations': ['addition', 'scalar_multiplication'],
            'results': [sum_decrypted, scaled_decrypted]
        }