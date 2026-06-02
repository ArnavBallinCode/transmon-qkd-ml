import os
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv

def main():
    load_dotenv()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.getenv("IBM_QUANTUM_TOKEN"))
    backend = service.backend("ibm_marrakesh")
    
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    
    # Extract physical qubit
    try:
        if hasattr(isa_qc, 'layout') and isa_qc.layout is not None:
            virtual_qubit = qc.qubits[0]
            physical_qubit = isa_qc.layout.final_index_layout()[0]
            print(f"Physical qubit (method 1): {physical_qubit}")
            print(f"Layout mapping: {isa_qc.layout.initial_layout[virtual_qubit]}")
        else:
            print("No layout attribute found.")
    except Exception as e:
        print(f"Error extracting layout: {e}")

if __name__ == "__main__":
    main()
