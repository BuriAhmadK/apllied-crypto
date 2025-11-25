#!/usr/bin/env python3
"""
Main runner for FHE Medical Research Demo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.demo_tenseal import TenSEALMedicalDemo
from src.demo_paillier import PaillierMedicalDemo
from src.advanced_performance_analyzer import AdvancedPerformanceAnalyzer
from src.data_simulator import MedicalDataSimulator

def main():
    print("🔬 FHE Medical Research Demo - Comprehensive Comparison")
    print("=" * 60)
    
    # Initialize components
    print("Initializing components...")
    tenseal_demo = TenSEALMedicalDemo()
    paillier_demo = PaillierMedicalDemo()
    analyzer = AdvancedPerformanceAnalyzer()
    data_simulator = MedicalDataSimulator()
    
    # Generate sample data
    print("\n1. Generating synthetic medical data...")
    sample_records = data_simulator.generate_patient_records(100)
    feature_matrix = data_simulator.records_to_feature_matrix(sample_records[:5])  # Small sample for demo
    
    print(f"Generated {len(sample_records)} patient records")
    print(f"Sample record: {sample_records[0]}")
    
    # Run individual demos
    print("\n2. Running TenSEAL (FHE) Demo...")
    tenseal_results = tenseal_demo.demonstrate_operations(feature_matrix[0])
    
    print("\n3. Running Paillier (Partial HE) Demo...")
    paillier_results = paillier_demo.demonstrate_operations([rec['age'] for rec in sample_records[:5]])
    
    print("\n4. Running Research-Grade Performance Benchmark...")
    advanced_analyzer = AdvancedPerformanceAnalyzer()

    comprehensive_metrics = advanced_analyzer.comprehensive_benchmark(
        tenseal_demo, paillier_demo, [10, 50, 100]
    )

    print("\n5. Generating Research-Grade Graphs...")
    advanced_analyzer.create_research_grade_graphs(comprehensive_metrics)

    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("📊 Check 'results/graphs/' directory for performance charts")
    print("💡 Review the recommendations above for choosing between FHE schemes")
    print("=" * 60)

if __name__ == "__main__":
    main()