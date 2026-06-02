from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

job = service.job("d8fe3507jphs739m47r0")

result = job.result()

bitarray = result[0].data.meas

print(bitarray.get_counts())