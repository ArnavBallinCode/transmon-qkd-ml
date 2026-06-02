# extract_readout_errors.py

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_marrakesh")

target = backend.target

rows = []

for q in range(backend.num_qubits):

    try:

        measure_props = target["measure"][(q,)]

        rows.append({
            "qubit": q,
            "readout_error": measure_props.error
        })

    except Exception:
        pass

df = pd.DataFrame(rows)

print(df.head())

df.to_csv(
    "data/hardware/ibm_marrakesh_readout_errors.csv",
    index=False
)