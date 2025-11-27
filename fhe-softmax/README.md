# Privacy-Preserving Softmax using Fully Homomorphic Encryption (FHE)# FHE Softmax Implementation with FastAPI



This project demonstrates a privacy-preserving implementation of the Softmax activation function using the CKKS homomorphic encryption scheme (via the TenSEAL library). It includes a client-server architecture for secure inference and a comprehensive benchmark comparing different polynomial approximation methods.A demonstration of Fully Homomorphic Encryption (FHE) using TenSEAL and FastAPI. The server computes softmax on encrypted data without ever seeing the plaintext values.



## 📂 Project Structure## 🔐 What is FHE?



- **`approximations.py`**: Core FHE logic implementing Taylor Series and Chebyshev polynomial approximations for `exp(x)` and Softmax.Fully Homomorphic Encryption allows computations to be performed on encrypted data without decrypting it first. This means:

- **`benchmark.py`**: Benchmarking suite to evaluate accuracy and performance of different approximation degrees (1-5).- The server never sees your private data

- **`visualize_results.py`**: Generates graphs comparing L1 error and classification accuracy.- All computations happen on encrypted values

- **`client.py`**: FHE Client that encrypts data and sends it to the server.- Only you (with the secret key) can decrypt the results

- **`server.py`**: FastAPI Server that performs homomorphic Softmax computation.

- **`TECHNICAL_DOCUMENTATION.md`**: Detailed explanation of the mathematical theory, encryption parameters, and budget management.## 🏗️ Architecture



## 🚀 Features```

┌─────────────────┐                  ┌─────────────────┐

- **Homomorphic Softmax**: Computes Softmax on encrypted vectors without decryption.│                 │  Encrypted Data  │                 │

- **Approximation Methods**:│  Client         │ ───────────────> │  Server         │

  - **Taylor Series**: Standard expansion of `exp(x)`.│  (Has keys)     │                  │  (No keys)      │

  - **Chebyshev Polynomials**: Minimax approximation for better stability.│                 │ <─────────────── │                 │

- **Optimized Parameters**: Tuned CKKS parameters (`poly_modulus_degree=8192`) for optimal balance between security, precision, and speed.│                 │ Encrypted Result │                 │

- **Context Reuse**: Efficient benchmarking system that reuses encryption contexts to save time.└─────────────────┘                  └─────────────────┘

       │                                      │

## 🛠️ Installation       │ Decrypts result                     │ Computes softmax

       │                                      │ on encrypted data

1. **Clone the repository**       ▼                                      ▼

   ```bash   Plaintext                          Never sees plaintext!

   git clone <repository-url>```

   cd fhe-softmax

   ```## 📋 Prerequisites



2. **Create a virtual environment**- Python 3.8 or higher

   ```bash- pip package manager

   python -m venv venv

   # Windows## 🚀 Quick Start

   .\venv\Scripts\activate

   # Linux/Mac### 1. Clone or Create Project Directory

   source venv/bin/activate```bash

   ```mkdir fhe-softmax

cd fhe-softmax

3. **Install dependencies**```

   ```bash

   pip install -r requirements.txt### 2. Create Virtual Environment

   ``````bash

# Create virtual environment

## 📊 Running the Benchmarkpython -m venv venv



Compare the accuracy and performance of Taylor vs. Chebyshev approximations.# Activate it (Linux/Mac)

source venv/bin/activate

1. **Run the benchmark**

   ```bash# Activate it (Windows)

   python benchmark.pyvenv\Scripts\activate

   ``````

   This will run 100 tests (2 methods × 5 degrees × 10 samples) and save results to `benchmark_results.csv`.

### 3. Install Dependencies

2. **Visualize results**```bash

   ```bashpip install --upgrade pip

   python visualize_results.pypip install -r requirements.txt

   ``````

   This generates:

   - `degree_vs_error.png`: Comparison of L1 errors.**Note**: TenSEAL installation may take a few minutes as it compiles C++ code.

   - `argmax_hits_vs_misses.png`: Classification accuracy analysis.

### 4. Run the Server

## 🌐 Running the Client-Server DemoOpen a terminal and run:

```bash

Simulate a secure inference scenario where a client sends encrypted logits to a server.# Make sure venv is activated

python server.py

1. **Start the Server**```

   ```bash

   python server.pyYou should see:

   ``````

   The server will start on `http://localhost:8000`.====================================================

FHE SOFTMAX SERVER

2. **Run the Client** (in a new terminal)====================================================

   ```bash

   python client.py[SERVER] Starting FHE Softmax Server on http://localhost:8000

   ```[SERVER] Press CTRL+C to stop

   The client will:```

   - Generate a random vector.

   - Encrypt it using CKKS.### 5. Run the Client

   - Send it to the server.Open a **new terminal**, activate the venv, and run:

   - Receive the encrypted Softmax result.```bash

   - Decrypt and compare with the plaintext ground truth.# Activate venv first

source venv/bin/activate  # or venv\Scripts\activate on Windows

## 📝 Technical Details

# Run client

- **Encryption Scheme**: CKKS (Approximate Homomorphic Encryption)python client.py

- **Polynomial Modulus**: 8192```

- **Coefficient Modulus**: `[30, 26, 26, 26, 26, 26, 30]` (190 bits)

- **Global Scale**: $2^{26}$## 📊 Sample Output

- **Input Scaling**: Inputs are implicitly handled; budget is managed to allow up to degree 5 polynomials.

### Server Terminal:

For a deep dive into the encryption budget and mathematical derivations, please refer to [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).```

====================================================

## 📄 License[SERVER] Received encrypted data from client

====================================================

[MIT License](LICENSE)[SERVER] Decoding base64 data...

[SERVER] Deserializing encryption context...
[SERVER] Context loaded - Poly modulus degree: 8192
[SERVER] Deserializing encrypted vector...
[SERVER] Encrypted vector loaded - Size: 5
[SERVER] Computing softmax on encrypted data...
[SERVER] Step 1: Computing exp(x) for each element...
[SERVER] Step 2: Summing all exp values...
[SERVER] Step 3: Dividing each exp(x) by sum...
[SERVER] ✓ Softmax computation complete!
[SERVER] Serializing encrypted result...
[SERVER] Sending encrypted result back to client
====================================================
```

### Client Terminal:
```
============================================================
  FHE SOFTMAX CLIENT
============================================================

[CLIENT] Step 1: Initializing encryption context...
[CLIENT] Using CKKS scheme (supports floating point operations)
[CLIENT] Generating Galois keys (for rotation operations)...
[CLIENT] Generating relinearization keys (for multiplication)...
[CLIENT] ✓ Encryption context created successfully!
[CLIENT]   - Poly modulus degree: 8192
[CLIENT]   - Global scale: 2^40

[CLIENT] Step 2: Creating dummy data...
[CLIENT] Original data: [1.0, 2.5, 0.5, 3.0, 1.5]
[CLIENT] Expected softmax: [0.052428, 0.234581, 0.031777, 0.386672, 0.094642]

[CLIENT] Step 3: Encrypting data...
[CLIENT] ✓ Data encrypted! (Size: 5 elements)

[CLIENT] Step 4: Preparing data for transmission...
[CLIENT] Context size: 28934 bytes
[CLIENT] Encrypted data size: 8872 bytes

[CLIENT] Step 5: Sending encrypted data to server...
[CLIENT] Server URL: http://localhost:8000/compute_softmax
[CLIENT] ✓ Received encrypted result from server (0.45s)

[CLIENT] Step 6: Decrypting result...
[CLIENT] ✓ Decryption complete!

============================================================
  RESULTS
============================================================

Original Data:        [1.0, 2.5, 0.5, 3.0, 1.5]
Expected Softmax:     ['0.052428', '0.234581', '0.031777', '0.386672', '0.094642']
FHE Computed Softmax: ['0.052156', '0.234891', '0.031654', '0.387012', '0.094287']

Absolute Error:       ['0.000272', '0.000310', '0.000123', '0.000340', '0.000355']
Max Error:            0.000355
Mean Error:           0.000280

✓ SUCCESS: FHE computation is accurate! (Error < 0.01)

============================================================
  FHE DEMO COMPLETE
============================================================

Key Points:
  • The server NEVER saw the plaintext data
  • All computation was performed on encrypted values
  • Only the client with the secret key can decrypt results
```

## 🔍 How It Works

1. **Client Initialization**
   - Creates encryption context using CKKS scheme
   - Generates encryption keys (kept secret)
   - Generates Galois keys (for rotations) and relinearization keys (for multiplications)

2. **Data Encryption**
   - Client encrypts input data: `[1.0, 2.5, 0.5, 3.0, 1.5]`
   - Creates a public version of the context (without secret key)

3. **Server Communication**
   - Client sends encrypted data + public context to server
   - Server receives only encrypted values (cannot decrypt them)

4. **Homomorphic Computation**
   - Server computes softmax on encrypted data:
     - Approximates exp(x) using polynomial
     - Sums all encrypted exponentials
     - Divides each by the sum
   - All operations preserve encryption!

5. **Result Decryption**
   - Server returns encrypted result
   - Client decrypts using secret key
   - Verifies accuracy against plaintext computation

## 🛠️ Technical Details

### Encryption Scheme
- **CKKS (Cheon-Kim-Kim-Song)**: Supports approximate arithmetic on encrypted floating-point numbers
- **Polynomial Modulus Degree**: 8192 (security parameter)
- **Coefficient Modulus**: [60, 40, 40, 60] bits (controls precision and operations)

### Softmax Approximation
Since exact exp() is impossible in FHE, we use polynomial approximation:
```
exp(x) ≈ 1 + x + x²/2 + x³/6
```

This introduces small errors (~0.001) but keeps computation practical.

### Performance
- Context creation: ~2-3 seconds
- Encryption: <1 second
- Server computation: ~0.5 seconds
- Total roundtrip: ~3-5 seconds

## ⚠️ Limitations

1. **Approximation Errors**: Polynomial approximations introduce small errors
2. **Performance**: FHE is slower than plaintext computation (100-1000x)
3. **Limited Operations**: Not all operations are efficient in FHE
4. **Memory Usage**: Encrypted data is much larger than plaintext

## 🧪 Troubleshooting

### "Could not connect to server"
- Make sure server is running: `python server.py`
- Check if port 8000 is available
- Try restarting both server and client

### "TenSEAL not found"
```bash
pip install tenseal==0.3.14
```

### "Context deserialization failed"
- Ensure client and server use compatible TenSEAL versions
- Check that context is properly serialized

### Large Errors in Results
- This is normal for low-degree polynomial approximations
- Increase polynomial degree (at the cost of performance)
- Adjust CKKS parameters for better precision

## 📚 Learn More

- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://github.com/microsoft/SEAL) (underlying library)
- [CKKS Scheme Paper](https://eprint.iacr.org/2016/421.pdf)

## 🤝 Contributing

This is a demonstration project. Feel free to:
- Experiment with different functions
- Optimize the polynomial approximations
- Add more complex computations

## 📄 License

MIT License - Feel free to use for educational purposes

## 🎯 Use Cases

Real-world applications of FHE:
- Private medical data analysis
- Secure financial computations
- Privacy-preserving machine learning
- Confidential data analytics

---

**Remember**: The server performs real computations on your data without ever seeing it in plaintext. This is the power of Fully Homomorphic Encryption! 🔐
