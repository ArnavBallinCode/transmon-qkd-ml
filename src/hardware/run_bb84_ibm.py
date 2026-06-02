import numpy as np
import pandas as pd
import os
import sys
import time
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from dotenv import load_dotenv

def generate_bb84_data(N=100):
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
    """
    qc = QuantumCircuit(1)
    
    # Alice State Preparation
    if alice_bit == 1:
        qc.x(0)
    if alice_basis == 1:
        qc.h(0)
        
    # Bob Basis Selection
    if bob_basis == 1:
        qc.h(0)
        
    # Measure
    qc.measure_all()
    
    return qc

def submit_job(circuits, backend_name=None, physical_qubit=None):
    """
    Submits a batch of circuits to IBM Quantum using SamplerV2.
    """
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    
    if token and token != "your_token_here":
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService()
        
    print("Selecting backend...")
    if not backend_name:
        backend_name = os.getenv("IBM_BACKEND", "ibm_marrakesh")
        
    backend = service.backend(backend_name)
    print(f"Using backend: {backend.name}")
    
    if physical_qubit is None:
        # Pick a random physical qubit from available
        physical_qubit = np.random.randint(0, backend.num_qubits)
    print(f"Mapping all circuits to physical qubit: {physical_qubit}")

    print("Generating ISA circuits...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=[physical_qubit])
    isa_circuits = [pm.run(qc) for qc in circuits]

    print("Submitting job...")
    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=1)
    print(f"Job ID: {job.job_id()}")
    
    print("Waiting for completion...")
    result = job.result()
    return result, physical_qubit

def extract_measurements(result, N):
    """
    Extracts the measured bits from SamplerV2 result.
    """
    measured_bits = []
    for i in range(N):
        pub_result = result[i]
        counts = pub_result.data.meas.get_counts()
        measured_bit = int(list(counts.keys())[0])
        measured_bits.append(measured_bit)
    return np.array(measured_bits)

def compute_qber(df):
    """
    Performs sifting and computes QBER.
    """
    sifted_df = df[df['alice_basis'] == df['bob_basis']].copy()
    sifted_length = len(sifted_df)
    
    if sifted_length == 0:
        return 0, 0, 0.0
        
    errors = (sifted_df['alice_bit'] != sifted_df['measured_bit']).sum()
    qber = errors / sifted_length
    return sifted_length, errors, qber

def append_to_summary(run_id, N, physical_qubit, sifted_length, errors, qber, out_dir):
    """
    Appends the summary of a run to the hardware_qber_summary_{N}.csv
    """
    summary_path = os.path.join(out_dir, f"hardware_qber_summary_{N}.csv")
    
    df_new = pd.DataFrame([{
        "run_id": run_id,
        "total_transmissions": N,
        "physical_qubit": physical_qubit,
        "sifted_length": sifted_length,
        "errors": errors,
        "qber": qber
    }])
    
    if not os.path.exists(summary_path):
        df_new.to_csv(summary_path, index=False)
    else:
        df_new.to_csv(summary_path, mode='a', header=False, index=False)

def main():
    run_id = 1
    if len(sys.argv) > 1:
        run_id = int(sys.argv[1])
        
    N = 100
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
        
    print(f"--- Starting BB84 Run {run_id} ---")
    print(f"Generating {N} BB84 transmissions...")
    
    alice_bits, alice_bases, bob_bases = generate_bb84_data(N)
    
    print("Building circuits...")
    circuits = [build_bb84_circuit(alice_bits[i], alice_bases[i], bob_bases[i]) for i in range(N)]
        
    result, physical_qubit = submit_job(circuits)
    
    print("Extracting measurements...")
    measured_bits = extract_measurements(result, N)
    
    df = pd.DataFrame({
        'alice_bit': alice_bits,
        'alice_basis': alice_bases,
        'bob_basis': bob_bases,
        'measured_bit': measured_bits,
        'physical_qubit': physical_qubit
    })
    
    sifted_length, errors, qber = compute_qber(df)
    
    print(f"\n--- BB84 Hardware Results (Run {run_id}) ---")
    print(f"Total transmissions: {N}")
    print(f"Physical Qubit     : {physical_qubit}")
    print(f"Sifted key length  : {sifted_length}")
    print(f"Number of errors   : {errors}")
    print(f"QBER               : {qber:.4f}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    out_dir = os.path.join(project_root, "results", "hardware", f"N_{N}")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"bb84_run_{run_id}_{N}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved run data to {out_path}")
    
    append_to_summary(run_id, N, physical_qubit, sifted_length, errors, qber, out_dir)
    print(f"Appended results to hardware_qber_summary_{N}.csv")

if __name__ == "__main__":
    main()
