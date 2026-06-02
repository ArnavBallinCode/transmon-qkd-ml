# extract_calibration_data.py

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_marrakesh")

target = backend.target

rows = []

for q in range(backend.num_qubits):

    try:
        props = target.qubit_properties[q]

        rows.append({
            "qubit": q,
            "t1_us": props.t1 * 1e6 if props.t1 else None,
            "t2_us": props.t2 * 1e6 if props.t2 else None,
            "frequency_ghz": props.frequency / 1e9 if props.frequency else None
        })

    except Exception as e:
        print("Skipping qubit", q, e)

df = pd.DataFrame(rows)

print(df.head())

df.to_csv(
    "data/hardware/ibm_marrakesh_qubit_calibration.csv",
    index=False
)

print("\nSaved calibration data")
print(df.describe())