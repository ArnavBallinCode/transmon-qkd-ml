# pyrefly: ignore [missing-import]
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="StZt3BiP5kSBmg_9M-h-Png1QjiFSKtnzw53AQ1pkZfg"
)

print("Connected!")