# extract_calibration_data.py

import os
import pandas as pd
from datetime import datetime, timezone
from qiskit_ibm_runtime import QiskitRuntimeService

# Use token from environment if available (for GitHub Actions)
ibm_token = os.environ.get("QISKIT_IBM_TOKEN")
if ibm_token:
    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
else:
    service = QiskitRuntimeService()

backend = service.backend("ibm_marrakesh")
target = backend.target

rows = []
current_time = datetime.now(timezone.utc).isoformat()

for q in range(backend.num_qubits):
    try:
        props = target.qubit_properties[q]
        rows.append({
            "timestamp": current_time,
            "qubit": q,
            "t1_us": props.t1 * 1e6 if props.t1 else None,
            "t2_us": props.t2 * 1e6 if props.t2 else None,
            "frequency_ghz": props.frequency / 1e9 if props.frequency else None
        })
    except Exception as e:
        print("Skipping qubit", q, e)

df = pd.DataFrame(rows)

csv_path = "data/hardware/ibm_marrakesh_qubit_calibration.csv"
file_exists = os.path.isfile(csv_path)

# Append to CSV if it exists, otherwise write new file with header
df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

print(f"\nSaved calibration data at {current_time}")
print(f"Appended {len(df)} rows to {csv_path}")