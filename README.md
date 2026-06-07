# Noise Analysis in Transmon Qubits with QKD & ML-based Error Prediction

**Overview & Motivation:** This project evaluates quantum algorithms (like Quantum Key Distribution and Entanglement Fidelity) on noisy transmon qubits. By continuously monitoring IBM Quantum hardware, we use Machine Learning to predict hardware degradation and eventually build an Algorithm-Aware Qubit Recommendation Engine.

## The Hardware Automation Pipeline
Because IBM Quantum operates on asynchronous queueing systems, we have built a fully automated GitHub Actions pipeline that independently collects both hardware calibration metadata and executes actual quantum circuits 24/7 without timing out.

### 1. Calibration Data Collection (Every 4 Hours)
The pipeline connects to the `ibm_marrakesh` backend and downloads the latest physics calibration metadata for all 156 qubits. This consumes **no QPU time** as it is merely fetching IBM's calibration snapshots.
*   **$T_1$ (Relaxation Time)**: How long a qubit can hold its energy state.
*   **$T_2$ (Dephasing Time)**: How long a qubit can hold its phase/superposition.
*   **Readout Error**: The probability of measurement failure.
*   **Dataset:** Appended to `data/hardware/ibm_marrakesh_calibration_timeseries.csv` and `data/hardware/ibm_marrakesh_readout_timeseries.csv`.

### 2. Automated Algorithm Benchmarking
We submit actual quantum circuits to `ibm_marrakesh` to measure the real-world impact of the noise described above.

#### Asynchronous Submission (Every 4 Hours)
*   **BB84 QKD Benchmark**: Generates $N=1000$ random cryptographic keys, securely encodes them onto a random physical qubit, and submits the job. This tests the **Quantum Bit Error Rate (QBER)** of single qubits.
*   **Bell State Benchmark**: Randomly selects a coupled pair of qubits (an edge) from the hardware map and prepares a maximally entangled state $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$. This tests the **Entanglement Fidelity** of the two-qubit CNOT gates.
*   **Metadata**: The secret Alice/Bob bit strings and job IDs are saved to `data/jobs/pending_jobs.csv` while we wait in the IBM queue.

#### Asynchronous Collection (Every 30 Minutes)
*   The system polls the IBM queue for the status of the submitted jobs.
*   If a job is finished, the pipeline downloads the measurement counts, sifts the keys using the saved metadata, and dynamically calculates the final QBER and Fidelity.
*   **Dataset:** Appended to `data/timeseries/benchmark_history.csv`.

## The Machine Learning Goal
By amassing a massive longitudinal dataset of:
`[Timestamp, Qubit, T1, T2, Readout Error]  ->  [Actual Measured Algorithm Performance]`

We train predictive Machine Learning models (Random Forests, Neural Networks) capable of **Algorithm-Aware Qubit Recommendation**. The final engine will allow a user to input a specific algorithm (e.g., Teleportation) and automatically receive a recommendation for the absolute best physical qubits available on the IBM chip at that exact moment in time based on our historical drift models.

## Local Reproduction
1. Provide your `QISKIT_IBM_TOKEN` in `.env`.
2. Generate base models: `conda run -n transmon-qkd python src/ml/train_models.py`
3. Run baseline inference: `conda run -n transmon-qkd python src/ml/run_inference.py`
4. Run manual BB84 hardware verification: `conda run -n transmon-qkd python src/hardware/run_bb84_ibm.py 1`
