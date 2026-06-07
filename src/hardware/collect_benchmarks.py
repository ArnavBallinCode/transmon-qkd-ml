import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from qiskit_ibm_runtime import QiskitRuntimeService

def compute_bb84_qber(result, metadata, N):
    alice_bits = np.array(metadata["alice_bits"])
    alice_bases = np.array(metadata["alice_bases"])
    bob_bases = np.array(metadata["bob_bases"])
    
    measured_bits = []
    for i in range(N):
        pub_result = result[i]
        counts = pub_result.data.meas.get_counts()
        measured_bit = int(list(counts.keys())[0])
        measured_bits.append(measured_bit)
        
    measured_bits = np.array(measured_bits)
    
    df = pd.DataFrame({
        'alice_bit': alice_bits,
        'alice_basis': alice_bases,
        'bob_basis': bob_bases,
        'measured_bit': measured_bits
    })
    
    sifted_df = df[df['alice_basis'] == df['bob_basis']]
    sifted_length = len(sifted_df)
    
    if sifted_length == 0:
        return 0.0
        
    errors = (sifted_df['alice_bit'] != sifted_df['measured_bit']).sum()
    qber = errors / sifted_length
    return qber

def compute_bell_fidelity(result):
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    total = sum(counts.values())
    fidelity = (counts.get('00', 0) + counts.get('11', 0)) / total
    return fidelity

def append_to_history(timestamp, backend, qubit_str, benchmark, sample_size, metric_name, metric_value):
    history_csv = "data/timeseries/benchmark_history.csv"
    
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "backend": backend,
        "physical_qubit": qubit_str,
        "benchmark": benchmark,
        "sample_size": sample_size,
        "metric_name": metric_name,
        "metric_value": metric_value
    }])
    
    file_exists = os.path.isfile(history_csv)
    df.to_csv(history_csv, mode='a', index=False, header=not file_exists)

def main():
    pending_csv = "data/jobs/pending_jobs.csv"
    if not os.path.exists(pending_csv):
        print("No pending jobs found.")
        return
        
    df_pending = pd.read_csv(pending_csv)
    if len(df_pending) == 0:
        print("No pending jobs found.")
        return
        
    ibm_token = os.environ.get("QISKIT_IBM_TOKEN")
    if ibm_token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
    else:
        service = QiskitRuntimeService()
        
    still_pending = []
    
    for _, row in df_pending.iterrows():
        job_id = row['job_id']
        benchmark = row['benchmark']
        backend_name = row['backend']
        qubit_str = row['physical_qubit']
        sample_size = row['sample_size']
        
        print(f"Checking job {job_id} ({benchmark})...")
        try:
            job = service.job(job_id)
            status = job.status()
        except Exception as e:
            print(f"Could not retrieve job {job_id}: {e}")
            still_pending.append(row)
            continue
            
        if status == "DONE":
            print(f"Job {job_id} is DONE. Processing...")
            result = job.result()
            
            meta_path = f"data/jobs/metadata/{job_id}.json"
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            
            metric_name = "UNKNOWN"
            metric_value = 0.0
            
            if benchmark == "BB84":
                metric_value = compute_bb84_qber(result, metadata, sample_size)
                metric_name = "QBER"
            elif benchmark == "BellState":
                metric_value = compute_bell_fidelity(result)
                metric_name = "Fidelity"
                
            collection_time = datetime.now(timezone.utc).isoformat()
            append_to_history(collection_time, backend_name, qubit_str, benchmark, sample_size, metric_name, metric_value)
            
            # Cleanup metadata
            if os.path.exists(meta_path):
                os.remove(meta_path)
                
            print(f"Processed {benchmark} -> {metric_name}: {metric_value}")
            
        elif status in ["ERROR", "CANCELLED"]:
            print(f"Job {job_id} failed or cancelled. Discarding.")
            meta_path = f"data/jobs/metadata/{job_id}.json"
            if os.path.exists(meta_path):
                os.remove(meta_path)
        else:
            print(f"Job {job_id} status: {status}. Keeping in queue.")
            still_pending.append(row)
            
    # Rewrite pending jobs
    df_new = pd.DataFrame(still_pending)
    if len(df_new) > 0:
        df_new.to_csv(pending_csv, index=False)
    else:
        # Create empty CSV with headers
        pd.DataFrame(columns=df_pending.columns).to_csv(pending_csv, index=False)
    
if __name__ == "__main__":
    main()
