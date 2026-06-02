from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_marrakesh")

target = backend.target

with open("backend_target_dump.txt", "w") as f:
    f.write(str(target))

print("saved")