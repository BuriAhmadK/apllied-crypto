"""
FHE Softmax Benchmark
Compares Taylor vs Chebyshev approximations across different polynomial degrees
"""

import tenseal as ts
import numpy as np
import pandas as pd
from datetime import datetime
import time
from approximations import get_budget_config, compute_softmax_fhe


def generate_test_data(num_samples=10, vector_size=10, seed=42):
    """Generate test data for ML classification logits scenario."""
    np.random.seed(seed)
    test_vectors = []
    for i in range(num_samples):
        logits = np.random.normal(loc=1.5, scale=1.2, size=vector_size)
        test_vectors.append(logits)
    return test_vectors


def compute_true_softmax(vector):
    """Compute ground truth softmax using standard numpy."""
    exp_x = np.exp(vector - np.max(vector))
    return exp_x / np.sum(exp_x)


def create_context():
    """
    Create a TenSEAL encryption context with proper budget allocation.
    Uses SINGLE shared context for all tests to avoid expensive re-creation.
    """
    print("[CONTEXT] Creating encryption context...")
    
    coeff_mod = get_budget_config()
    print(f"[CONTEXT] Coefficient modulus: {coeff_mod}")
    print(f"[CONTEXT] Total bits: {sum(coeff_mod)} (limit: 218 for poly=8192)")
    print(f"[CONTEXT] Levels: {len(coeff_mod)} (allows {len(coeff_mod)-1} multiplications)")
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=coeff_mod
    )
    context.global_scale = 2**26
    
    print("[CONTEXT] Generating Galois keys...")
    context.generate_galois_keys()
    
    print("[CONTEXT] Generating relinearization keys...")
    context.generate_relin_keys()
    
    print("[CONTEXT] Ready!\n")
    return context


def run_single_test(context, method, degree, test_vector, sample_id, test_num, total):
    """Run a single FHE softmax computation."""
    try:
        # Ground truth
        true_softmax = compute_true_softmax(test_vector)
        true_argmax = np.argmax(true_softmax)
        
        # Encrypt
        encrypted_vector = ts.ckks_vector(context, test_vector.tolist())
        
        # FHE computation
        start_time = time.time()
        print(f"[{test_num:3d}/{total}] [{method.upper():10s} deg={degree} sample={sample_id:2d}] ", end="", flush=True)
        encrypted_result = compute_softmax_fhe(encrypted_vector, degree, method)
        elapsed = time.time() - start_time
        
        # Decrypt
        fhe_result = np.array(encrypted_result.decrypt()[:len(test_vector)])
        
        # Metrics
        l1_error = np.mean(np.abs(fhe_result - true_softmax))
        fhe_argmax = np.argmax(fhe_result)
        argmax_match = (fhe_argmax == true_argmax)
        
        status = "HIT" if argmax_match else "MISS"
        print(f"L1={l1_error:.6f} argmax={status} ({elapsed:.2f}s)")
        
        return {
            'method': method,
            'degree': degree,
            'sample_id': sample_id,
            'l1_error': l1_error,
            'argmax_match': argmax_match,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            'method': method,
            'degree': degree,
            'sample_id': sample_id,
            'l1_error': np.nan,
            'argmax_match': False,
            'success': False
        }


def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("FHE SOFTMAX BENCHMARK")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Config
    methods = ['taylor', 'chebyshev']
    degrees = [1, 2, 3, 4, 5]
    num_samples = 10
    
    total_tests = len(methods) * len(degrees) * num_samples
    print(f"Tests: {len(methods)} methods x {len(degrees)} degrees x {num_samples} samples = {total_tests}\n")
    
    # Generate data
    print("[DATA] Generating test vectors...")
    test_vectors = generate_test_data(num_samples=num_samples)
    print(f"[DATA] {len(test_vectors)} vectors, size {len(test_vectors[0])}\n")
    
    # Create context ONCE
    print("=" * 70)
    print("PHASE 1: Context Creation")
    print("=" * 70)
    context = create_context()
    
    # Run tests
    print("=" * 70)
    print("PHASE 2: Benchmark")
    print("=" * 70)
    
    results = []
    test_num = 0
    
    for method in methods:
        for degree in degrees:
            for sample_id, test_vector in enumerate(test_vectors):
                test_num += 1
                result = run_single_test(context, method, degree, test_vector, sample_id, test_num, total_tests)
                results.append(result)
    
    # Save
    print("\n" + "=" * 70)
    print("PHASE 3: Results")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    print(f"\nSaved: benchmark_results.csv")
    
    # Summary
    successful = df[df['success'] == True]
    failed = df[df['success'] == False]
    print(f"Success: {len(successful)}/{len(df)}")
    if len(failed) > 0:
        print(f"Failed: {len(failed)}")
    
    print("\n--- Summary ---")
    for method in methods:
        print(f"\n{method.upper()}:")
        for degree in degrees:
            subset = successful[(successful['method'] == method) & (successful['degree'] == degree)]
            if len(subset) > 0:
                mean_err = subset['l1_error'].mean()
                std_err = subset['l1_error'].std()
                hits = int(subset['argmax_match'].sum())
                total = len(subset)
                print(f"  Deg {degree}: L1={mean_err:.4f}±{std_err:.4f} | Argmax={hits}/{total} ({100*hits/total:.0f}%)")
    
    print(f"\n{'=' * 70}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nNext: python visualize_results.py")


if __name__ == "__main__":
    main()
