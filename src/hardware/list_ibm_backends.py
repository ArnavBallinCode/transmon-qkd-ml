from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

print("Available Backends:\n")

for backend in service.backends():
    print(backend.name)