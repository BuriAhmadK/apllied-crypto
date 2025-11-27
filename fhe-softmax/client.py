import tenseal as ts
import numpy as np
import httpx
import base64
import time


def compute_softmax_plaintext(x):
    """Compute softmax on plaintext data for verification."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    """Main client function - encrypts data, sends to server, decrypts result."""
    
    print_section("FHE SOFTMAX CLIENT")
    
    # Step 1: Initialize encryption context
    print("\n[CLIENT] Step 1: Initializing encryption context...")
    print("[CLIENT] Using CKKS scheme (supports floating point operations)")
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,  # Increased for more budget (slower but more capacity)
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]  # 8 levels, 400 bits total
    )
    context.global_scale = 2**40
    
    print("[CLIENT] Generating Galois keys (for rotation operations)...")
    context.generate_galois_keys()
    
    print("[CLIENT] Generating relinearization keys (for multiplication)...")
    context.generate_relin_keys()
    
    print("[CLIENT] ✓ Encryption context created successfully!")
    print(f"[CLIENT]   - Poly modulus degree: 16384 (high security)")
    print(f"[CLIENT]   - Global scale: 2^40")
    print(f"[CLIENT]   - Coefficient modulus chain: 8 levels (400 bits total budget)")
    
    # Step 2: Create dummy data
    print("\n[CLIENT] Step 2: Creating dummy data...")
    original_data = [1.0, 2.5, 0.5, 3.0, 1.5]
    print(f"[CLIENT] Original data: {original_data}")
    
    # Compute expected softmax for verification
    expected_softmax = compute_softmax_plaintext(np.array(original_data))
    print(f"[CLIENT] Expected softmax: {expected_softmax.tolist()}")
    
    # Step 3: Encrypt data
    print("\n[CLIENT] Step 3: Encrypting data...")
    encrypted_vector = ts.ckks_vector(context, original_data)
    print(f"[CLIENT] ✓ Data encrypted! (Size: {encrypted_vector.size()} elements)")
    
    # Step 4: Serialize for transmission
    print("\n[CLIENT] Step 4: Preparing data for transmission...")
    
    # Make context public (remove secret key)
    public_context = context.copy()
    public_context.make_context_public()
    
    context_bytes = public_context.serialize()
    encrypted_data_bytes = encrypted_vector.serialize()
    
    context_base64 = base64.b64encode(context_bytes).decode('utf-8')
    encrypted_data_base64 = base64.b64encode(encrypted_data_bytes).decode('utf-8')
    
    print(f"[CLIENT] Context size: {len(context_bytes)} bytes")
    print(f"[CLIENT] Encrypted data size: {len(encrypted_data_bytes)} bytes")
    
    # Step 5: Send to server
    print("\n[CLIENT] Step 5: Sending encrypted data to server...")
    print("[CLIENT] Server URL: http://localhost:8000/compute_softmax")
    print("[CLIENT] Note: Large polynomial degree (16384) may take 30-60 seconds...")
    
    try:
        start_time = time.time()
        
        response = httpx.post(
            "http://localhost:8000/compute_softmax",
            json={
                "context": context_base64,
                "encrypted_data": encrypted_data_base64
            },
            timeout=120.0  # Increased to 2 minutes for large polynomial degree
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"[CLIENT] ✓ Received encrypted result from server ({elapsed_time:.2f}s)")
        else:
            print(f"[CLIENT] ✗ Error: Server returned status {response.status_code}")
            print(f"[CLIENT] Response: {response.text}")
            return
            
    except httpx.ConnectError:
        print("[CLIENT] ✗ ERROR: Could not connect to server!")
        print("[CLIENT] Make sure the server is running: python server.py")
        return
    except Exception as e:
        print(f"[CLIENT] ✗ ERROR: {str(e)}")
        return
    
    # Step 6: Decrypt result
    print("\n[CLIENT] Step 6: Decrypting result...")
    
    result_base64 = response.json()["encrypted_result"]
    result_bytes = base64.b64decode(result_base64)
    
    encrypted_result = ts.ckks_vector_from(context, result_bytes)
    decrypted_result = encrypted_result.decrypt()
    
    print(f"[CLIENT] ✓ Decryption complete!")
    
    # Step 7: Display results
    print_section("RESULTS")
    
    print(f"\nOriginal Data:        {original_data}")
    print(f"Expected Softmax:     {[f'{x:.6f}' for x in expected_softmax]}")
    print(f"FHE Computed Softmax: {[f'{x:.6f}' for x in decrypted_result[:len(original_data)]]}")
    
    # Calculate error
    error = np.abs(expected_softmax - np.array(decrypted_result[:len(original_data)]))
    print(f"\nAbsolute Error:       {[f'{x:.6f}' for x in error]}")
    print(f"Max Error:            {np.max(error):.6f}")
    print(f"Mean Error:           {np.mean(error):.6f}")
    
    # Check relative ordering
    expected_order = np.argsort(expected_softmax)[::-1]  # Descending order
    computed_order = np.argsort(decrypted_result[:len(original_data)])[::-1]
    order_correct = np.array_equal(expected_order, computed_order)
    
    print(f"\nExpected ranking:     {expected_order.tolist()} (highest to lowest)")
    print(f"Computed ranking:     {computed_order.tolist()} (highest to lowest)")
    print(f"Ranking preserved:    {'✓ YES' if order_correct else '✗ NO'}")
    
    if np.max(error) < 0.05 and order_correct:
        print("\n✓ SUCCESS: FHE softmax is accurate! (Error < 0.05, ranking preserved)")
    elif order_correct:
        print("\n✓ PARTIAL SUCCESS: Ranking is correct but values have some error.")
        print("  This is acceptable for many ML applications (classification, etc.)")
    else:
        print("\n⚠ WARNING: Ranking not preserved - need better approximation.")
        print("  Reasons:")
        print("  - Polynomial approximation limitations")
        print("  - Normalization approximation errors")
        print("  - CKKS precision constraints")
    
    print_section("FHE DEMO COMPLETE")
    print("\nKey Points:")
    print("  • The server NEVER saw the plaintext data")
    print("  • All computation was performed on encrypted values")
    print("  • Only the client with the secret key can decrypt results")
    print()


if __name__ == "__main__":
    main()
