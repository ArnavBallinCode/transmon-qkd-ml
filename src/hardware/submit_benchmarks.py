import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def generate_bb84_data(N):
    alice_bits = np.random.randint(0, 2, N).tolist()
    alice_bases = np.random.randint(0, 2, N).tolist()
    bob_bases = np.random.randint(0, 2, N).tolist()
    return alice_bits, alice_bases, bob_bases

def build_bb84_circuits(alice_bits, alice_bases, bob_bases):
    circuits = []
    for bit, a_basis, b_basis in zip(alice_bits, alice_bases, bob_bases):
        qc = QuantumCircuit(1)
        if bit == 1:
            qc.x(0)
        if a_basis == 1:
            qc.h(0)
        if b_basis == 1:
            qc.h(0)
        qc.measure_all()
        circuits.append(qc)
    return circuits

def build_bell_state_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def record_job(job_id, backend_name, qubit_str, benchmark, sample_size, metadata=None):
    pending_csv = "data/jobs/pending_jobs.csv"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Save metadata
    if metadata:
        meta_path = f"data/jobs/metadata/{job_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
            
    # Append to pending jobs
    df = pd.DataFrame([{
        "job_id": job_id,
        "timestamp": timestamp,
        "backend": backend_name,
        "physical_qubit": qubit_str,
        "benchmark": benchmark,
        "sample_size": sample_size,
        "status": "QUEUED"
    }])
    
    file_exists = os.path.isfile(pending_csv)
    df.to_csv(pending_csv, mode='a', index=False, header=not file_exists)
    print(f"Recorded job {job_id} for {benchmark} on {qubit_str}")

def main():
    ibm_token = os.environ.get("QISKIT_IBM_TOKEN")
    if ibm_token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
    else:
        service = QiskitRuntimeService()
        
    backend_name = "ibm_marrakesh"
    backend = service.backend(backend_name)
    sampler = Sampler(mode=backend)
    
    N = 1000 # Sample size requested by user
    
    # --- SUBMIT BB84 ---
    physical_qubit = np.random.randint(0, backend.num_qubits)
    print(f"Submitting BB84 on qubit {physical_qubit}...")
    alice_bits, alice_bases, bob_bases = generate_bb84_data(N)
    bb84_circuits = build_bb84_circuits(alice_bits, alice_bases, bob_bases)
    
    pm_bb84 = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=[physical_qubit])
    isa_bb84 = [pm_bb84.run(qc) for qc in bb84_circuits]
    
    job_bb84 = sampler.run(isa_bb84, shots=1)
    record_job(
        job_id=job_bb84.job_id(),
        backend_name=backend_name,
        qubit_str=str(physical_qubit),
        benchmark="BB84",
        sample_size=N,
        metadata={
            "alice_bits": alice_bits,
            "alice_bases": alice_bases,
            "bob_bases": bob_bases
        }
    )
    
    # --- SUBMIT BELL STATE ---
    # Pick a random edge from the coupling map
    edges = list(backend.coupling_map.get_edges())
    edge = edges[np.random.randint(0, len(edges))]
    qubit_pair = list(edge)
    
    print(f"Submitting Bell State on qubits {qubit_pair}...")
    bell_qc = build_bell_state_circuit()
    
    pm_bell = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=qubit_pair)
    isa_bell = pm_bell.run(bell_qc)
    
    job_bell = sampler.run([isa_bell], shots=N)
    record_job(
        job_id=job_bell.job_id(),
        backend_name=backend_name,
        qubit_str=f"{qubit_pair[0]}_{qubit_pair[1]}",
        benchmark="BellState",
        sample_size=N,
        metadata={} # Bell state doesn't need external random data
    )

if __name__ == "__main__":
    os.makedirs("data/jobs/metadata", exist_ok=True)
    os.makedirs("data/timeseries", exist_ok=True)
    main()
