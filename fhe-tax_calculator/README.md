

## Repository layout
```
applied_crypto/
├─ Model_A/    # TenSEAL BFV client/server
├─ Model_B/    # Paillier client/server
├─ e2e/        # Fernet client/server
├─ demo.py     # scratchpad (optional)
├─ requirements.txt
└─ README.md
```

## 1. One-time setup (PowerShell)
Run these commands from the repo root .

```powershell
# Create a virtual environment (once)
python -m venv .venv

# Activate it in the current shell (run in EVERY shell you use)
.\.venv\Scripts\Activate.ps1

# Install the shared dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs:
- `tenseal` + `numpy` for Model A (fully homomorphic demo)
- `phe` for Model B (Paillier demo)
- `cryptography` for the Fernet transport-encrypted demo

> **Tip:** When you open a new PowerShell window, activate the virtual environment again before running any Python commands.

## 2. Running each model
All models follow the same pattern: start the server in one shell and the client in another shell. Make sure both shells have the virtual environment activated.

### Model A — TenSEAL BFV (folder `Model_A/`)
**Shell A (server):**
```powershell
.\.venv\Scripts\Activate.ps1
python Model_A\server_ts.py
```

**Shell B (client):**
```powershell
.\.venv\Scripts\Activate.ps1
python Model_A\client_ts.py
```
Follow the prompts in the client shell (enter income + deductions). The client encrypts taxable income, the server performs the entire tax computation on ciphertexts, and the client decrypts the final tax result. Packet metrics are printed for bandwidth analysis.

### Model B — Paillier partially homomorphic (folder `Model_B/`)
**Shell A (server):**
```powershell
.\.venv\Scripts\Activate.ps1
python Model_B\server_phe.py
```

**Shell B (client):**
```powershell
.\.venv\Scripts\Activate.ps1
python Model_B\client_phe.py
```
The client splits taxable income into bands locally, encrypts each slice with Paillier, and the server homomorphically scales/sums the encrypted slices. This demonstrates partial homomorphic encryption and shows the difference in what the server learns compared to Model A.

### End-to-end transport encryption (folder `e2e/`)
**Shell A (server):**
```powershell
.\.venv\Scripts\Activate.ps1
python e2e\server_e2e.py
```

**Shell B (client):**
```powershell
.\.venv\Scripts\Activate.ps1
python e2e\client_e2e.py
```
The client encrypts request/response packets with a shared Fernet key. The server decrypts values, computes tax in plaintext, and re-encrypts its reply. Use this model to contrast transport-only encryption versus true homomorphic privacy.


## 3. Easy running demo

```powershell
.\.venv\Scripts\Activate.ps1
python demo.py
```