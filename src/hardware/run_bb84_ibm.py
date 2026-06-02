import numpy as np
import pandas as pd
import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def generate_bb84_data(N=50):
    """
    Generate random bits and bases for Alice and Bob.
    Basis Convention:
    0 -> Z basis
    1 -> X basis
    """
    alice_bits = np.random.randint(0, 2, N)
    alice_bases = np.random.randint(0, 2, N)
    bob_bases = np.random.randint(0, 2, N)
    return alice_bits, alice_bases, bob_bases

def build_bb84_circuit(alice_bit, alice_basis, bob_basis):
    """
    Builds a single BB84 quantum circuit.
    
    Alice state preparation:
    - (bit=0, basis=0): |0>
    - (bit=1, basis=0): X gate -> |1>
    - (bit=0, basis=1): H gate -> |+>
    - (bit=1, basis=1): X then H -> |->
    
    Bob basis selection:
    - If bob_basis == 1: apply H before measurement.
    """
    qc = QuantumCircuit(1)
    
    # Alice State Preparation
    if alice_bit == 1:
        qc.x(0)
    if alice_basis == 1:
        qc.h(0)
        
    # Channel
    # For this experiment, the IBM Quantum hardware acts as our noisy channel.
    
    # Bob Basis Selection
    if bob_basis == 1:
        qc.h(0)
        
    # Measure
    qc.measure_all()
    
    return qc




def submit_job(circuits):

    service = QiskitRuntimeService()

    print("Selecting backend...")

    backends = service.backends(
        simulator=False,
        operational=True
    )

    backend = min(
        backends,
        key=lambda b: b.status().pending_jobs
    )

    print(f"Using backend: {backend.name}")

    print("Generating ISA circuits...")

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1
    )

    isa_circuits = [
        pm.run(qc)
        for qc in circuits
    ]

    print("Submitting job...")

    sampler = Sampler(mode=backend)

    job = sampler.run(
        isa_circuits,
        shots=1
    )

    print(f"Job ID: {job.job_id()}")

    print("Waiting for completion...")

    result = job.result()

    return result

def extract_measurements(result, N):
    """
    Extracts the measured bits from SamplerV2 result.
    """
    measured_bits = []
    print("Unique measured bits:")
    print(np.unique(measured_bits, return_counts=True))
    for i in range(N):
        # Retrieve the PubResult for the i-th circuit
        pub_result = result[i]
        
        # qc.measure_all() creates a classical register named 'meas'
        # Extract the BitArray and get counts
        counts = pub_result.data.meas.get_counts()
        
        # Since shots=1, there should be exactly one key (e.g., '0' or '1')
        measured_bit = int(list(counts.keys())[0])
        measured_bits.append(measured_bit)
        
    return np.array(measured_bits)

def compute_qber(df):
    """
    Performs sifting and computes QBER.
    Sifting keeps only the rows where Alice's basis matches Bob's basis.
    """
    # Sifting
    sifted_df = df[df['alice_basis'] == df['bob_basis']].copy()
    sifted_length = len(sifted_df)
    
    if sifted_length == 0:
        return 0, 0, 0.0
        
    # Count errors between Alice's prepared bit and Bob's measured bit
    errors = (sifted_df['alice_bit'] != sifted_df['measured_bit']).sum()
    
    # Compute QBER
    qber = errors / sifted_length
    
    return sifted_length, errors, qber

def main():
    N = 3000
    print(f"Generating {N} BB84 transmissions...")
    alice_bits, alice_bases, bob_bases = generate_bb84_data(N)
    
    print("Building circuits...")
    circuits = []
    for i in range(N):
        qc = build_bb84_circuit(alice_bits[i], alice_bases[i], bob_bases[i])
        circuits.append(qc)
        
    result = submit_job(circuits)
    
    print("Extracting measurements...")
    measured_bits = extract_measurements(result, N)
    
    # Create DataFrame
    df = pd.DataFrame({
        'alice_bit': alice_bits,
        'alice_basis': alice_bases,
        'bob_basis': bob_bases,
        'measured_bit': measured_bits
    })
    
    # Compute QBER
    sifted_length, errors, qber = compute_qber(df)
    
    print("\n--- BB84 Hardware Results ---")
    print(f"Total transmissions: {N}")
    print(f"Sifted key length  : {sifted_length}")
    print(f"Number of errors   : {errors}")
    print(f"QBER               : {qber:.4f}")
    
    print("\nSample of Data:")
    print(df.head())
    
    # Save Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data", "hardware")
    os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, "hardware_bb84_results_3000.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
