from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()

backend = service.backend("ibm_kingston")

qc = QuantumCircuit(1)

# Prepare |1>
qc.x(0)

qc.measure_all()

sampler = Sampler(mode=backend)

job = sampler.run([qc], shots=4096)

print("Job ID:", job.job_id())

result = job.result()

print(result)