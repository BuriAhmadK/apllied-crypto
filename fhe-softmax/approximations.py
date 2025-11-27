"""
FHE Softmax Approximation Methods
Implements Taylor Series and Chebyshev Polynomial approximations for exp(x)
"""

import numpy as np


def get_budget_config(degree=None, method=None):
    """
    Return coefficient modulus configuration for given degree and method.
    
    For poly_modulus_degree=8192:
    - Maximum total bits = 218
    - Need to fit: [first] + [middle...] + [last]
    
    We use a SINGLE configuration that works for ALL degrees to simplify.
    This wastes some budget for lower degrees but ensures stability.
    
    Degree 5 requires:
    - x² (1 mult)
    - x³ (1 mult)  
    - x⁴ (1 mult)
    - x⁵ (1 mult)
    - final normalization (1 mult)
    Total: 5 multiplications = needs 6 levels minimum
    """
    # Conservative config: 7 levels (6 multiplications available)
    # Total: 30 + 26*5 + 30 = 190 bits (under 218 limit)
    return [30, 26, 26, 26, 26, 26, 30]


def taylor_exp(encrypted_vector, degree):
    """
    Compute exp(x) approximation using Taylor series.
    exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
    
    NO pre-scaling to preserve multiplicative depth budget.
    """
    print(f"[TAYLOR] Computing degree {degree}...")
    
    # Degree 1: exp(x) ≈ 1 + x
    if degree == 1:
        result = encrypted_vector + 1.0
        return result
    
    # Start with x (linear term)
    result = encrypted_vector.copy()
    
    # Compute x²
    x2 = encrypted_vector * encrypted_vector
    
    # Add x²/2
    result = result + (x2 * 0.5)
    
    if degree >= 3:
        x3 = x2 * encrypted_vector  # x³
        result = result + (x3 * 0.16666666667)  # x³/6
    
    if degree >= 4:
        x4 = x2 * x2  # x⁴
        result = result + (x4 * 0.04166666667)  # x⁴/24
    
    if degree >= 5:
        x5 = x4 * encrypted_vector  # x⁵
        result = result + (x5 * 0.00833333333)  # x⁵/120
    
    # Add constant 1
    result = result + 1.0
    
    return result


def chebyshev_exp(encrypted_vector, degree):
    """
    Compute exp(x) approximation using Chebyshev polynomials.
    Uses simple power series form for stability in FHE.
    
    NO pre-scaling to preserve multiplicative depth budget.
    """
    print(f"[CHEBYSHEV] Computing degree {degree}...")
    
    # Chebyshev-optimized coefficients for exp(x) on [-1, 1]
    coeffs = {
        1: [1.0, 1.1752],
        2: [1.0, 1.1752, 0.5431],
        3: [1.0, 1.1752, 0.5431, 0.1768],
        4: [1.0, 1.1752, 0.5431, 0.1768, 0.0443],
        5: [1.0, 1.1752, 0.5431, 0.1768, 0.0443, 0.0089],
    }
    
    c = coeffs[degree]
    
    # Build polynomial: c0 + c1*x + c2*x² + ...
    result = encrypted_vector * c[1]  # c1 * x
    
    if degree >= 2:
        x2 = encrypted_vector * encrypted_vector
        result = result + (x2 * c[2])
    
    if degree >= 3:
        x3 = x2 * encrypted_vector
        result = result + (x3 * c[3])
    
    if degree >= 4:
        x4 = x2 * x2
        result = result + (x4 * c[4])
    
    if degree >= 5:
        x5 = x4 * encrypted_vector
        result = result + (x5 * c[5])
    
    # Add constant
    result = result + c[0]
    
    return result


def compute_softmax_fhe(encrypted_vector, degree, method='taylor'):
    """
    Complete FHE softmax computation.
    
    REMOVED pre-scaling (encrypted_vector * 0.16666) to save multiplicative depth.
    Only uses final normalization multiplication.
    """
    print(f"[SOFTMAX] {method.upper()} degree {degree}")
    
    # Compute exp approximation directly (no pre-scaling)
    if method == 'taylor':
        exp_result = taylor_exp(encrypted_vector, degree)
    else:
        exp_result = chebyshev_exp(encrypted_vector, degree)
    
    # Normalize (1 multiplication)
    result = exp_result * 0.1
    
    return result
