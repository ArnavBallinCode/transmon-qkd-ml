# extract_readout_errors.py

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
        measure_props = target["measure"][(q,)]
        rows.append({
            "timestamp": current_time,
            "qubit": q,
            "readout_error": measure_props.error
        })
    except Exception:
        pass

df = pd.DataFrame(rows)

csv_path = "data/hardware/ibm_marrakesh_readout_errors.csv"
file_exists = os.path.isfile(csv_path)

# Append to CSV if it exists, otherwise write new file with header
df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

print(f"\nSaved readout errors at {current_time}")
print(f"Appended {len(df)} rows to {csv_path}")