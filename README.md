# Noise Analysis in Transmon Qubits with QKD & ML-based Error Prediction

**Overview & Motivation:** This project evaluates Quantum Key Distribution (BB84) on noisy transmon qubits, using machine learning to predict Quantum Bit Error Rates (QBER) and identify resilient qubits.

**BB84 Workflow:** We implement the BB84 protocol using Qiskit, transmitting basis states over quantum channels and sifting keys based on measurement outcomes.

**Hardware Integration:** Experiments execute directly on the `ibm_marrakesh` backend via Qiskit Runtime SamplerV2, alongside coherence time (T1/T2) and readout error calibration metrics.

**Data & Models:** We train Random Forest and Neural Network models on a modeled BB84 dataset, then infer expected QBER for real IBM Marrakesh hardware configurations.

**Reproduction:**
1. Provide your `IBM_QUANTUM_TOKEN` in `.env`.
2. Generate models: `conda run -n transmon-qkd python src/ml/train_models.py`
3. Run inference: `conda run -n transmon-qkd python src/ml/run_inference.py`
4. Run hardware BB84: `conda run -n transmon-qkd python src/hardware/run_bb84_ibm.py 1`
