# Applied Cryptography Projects

This repository contains a collection of projects demonstrating the practical applications of Fully Homomorphic Encryption (FHE) and Partial Homomorphic Encryption (PHE). These projects explore privacy-preserving computations in medical research, machine learning (Softmax), and financial services (Tax Calculation).

## 📂 Projects Overview

### 1. [FHE Medical Research Demo](./fhe-medical)
A comprehensive demonstration comparing Fully Homomorphic Encryption (TenSEAL/CKKS) against Partial Homomorphic Encryption (Paillier) for private medical research.
- **Key Features**:
  - Performance comparison between TenSEAL and Paillier.
  - Realistic synthetic patient data simulation.
  - Visualizations of performance metrics.

### 2. [FHE Softmax](./fhe-softmax)
A privacy-preserving implementation of the Softmax activation function using CKKS homomorphic encryption with a client-server architecture.
- **Key Features**:
  - Secure inference using FastAPI.
  - Polynomial approximations (Taylor Series & Chebyshev) for `exp(x)`.
  - Benchmarking suite for accuracy and performance.

### 3. [FHE Tax Calculator](./fhe-tax_calculator)
A secure tax calculator demonstrating three different approaches to privacy: Fully Homomorphic Encryption, Partial Homomorphic Encryption, and standard Transport Encryption.
- **Key Features**:
  - **Model A**: TenSEAL (BFV) for full homomorphic computation.
  - **Model B**: Paillier for partial homomorphic encryption.
  - **E2E**: Fernet for standard transport-layer encryption comparison.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd apllied-crypto
   ```

2. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the unified requirements for all projects:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

Each project is self-contained in its own directory. Navigate to the specific project folder to run its demos.

### Running FHE Medical Demo
```bash
cd fhe-medical
python run_demo.py
```

### Running FHE Softmax
This project requires running a server and a client in separate terminals.
```bash
cd fhe-softmax
# Terminal 1: Start Server
python server.py

# Terminal 2: Run Client
python client.py
```

### Running FHE Tax Calculator
Choose a model (Model_A, Model_B, or e2e) and run the server and client.
```bash
cd fhe-tax_calculator
# Example for Model A (TenSEAL)
# Terminal 1
python Model_A/server_ts.py

# Terminal 2
python Model_A/client_ts.py
```

For more detailed instructions, please refer to the `README.md` file within each project's directory.
