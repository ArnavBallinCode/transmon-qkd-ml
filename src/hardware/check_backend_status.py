import os
from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    backend_name = os.getenv("IBM_BACKEND", "ibm_marrakesh")
    
    print("Authenticating to IBM Quantum...")
    if token and token != "your_token_here":
        # Save or use token directly
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        # Fallback to saved account
        service = QiskitRuntimeService()
    
    print(f"Fetching backend '{backend_name}'...")
    backend = service.backend(backend_name)
    
    status = backend.status()
    print("\n--- Backend Status ---")
    print(f"Backend Name    : {backend.name}")
    print(f"Operational     : {status.operational}")
    print(f"Pending Jobs    : {status.pending_jobs}")
    print(f"Number of Qubits: {backend.num_qubits}")

if __name__ == "__main__":
    main()
