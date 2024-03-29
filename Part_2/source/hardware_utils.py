# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
import sys

import numpy as np
import qiskit
import quasar

sys.path.append("..")
from qcware_transpile.translations.quasar.to_qiskit import audit, translate

# fix for older versions of Qiskit
if qiskit.__version__ <= "0.37.1":
    import qiskit.providers.aer.noise as noise
else:
    import qiskit_aer.noise as noise

import copy
import json
import pickle
import time
from pathlib import Path

from tqdm import tqdm


def counter_to_dict(c):
    """Converts counter returned by pytket get_counts function
    to dictionary returned by qiskit
    canonical use:
    >>> result = backend.get_result(handle)
    >>> counts = result.get_counts(basis=BasisOrder.dlo)
    >>> counts_qiskit = counter_to_dict(counts)
    """
    d = {}
    for k, v in c.items():
        d["".join(str(x) for x in k)] = int(v)
    return d


def prepare_circuit_compound(rbs_idxs, time_step, num_qubits, seq_jumps, thetas):
    """
    This function prepares a quantum circuit for a given input and set of parameters,
    which can later be executed on quantum hardware.
    """

    def _get_layer_circuit(params):
        _params = np.array(params).astype("float")
        circuit_layer = quasar.Circuit()
        idx_angle = 0
        for gates_per_timestep in rbs_idxs:
            for gate in gates_per_timestep:
                circuit_layer.add_gate(
                    quasar.Gate.RBS(theta=-_params[idx_angle]), tuple(gate)
                )
                idx_angle += 1
        return circuit_layer

    first_gates = (
        [quasar.Circuit().H(0)] * (num_qubits - 2)
        + [quasar.Circuit().I(0)]
        + [quasar.Circuit().X(0)]
    )
    circuit = quasar.Circuit.join_in_qubits(first_gates)
    if time_step == 0:
        layer_circuit = _get_layer_circuit(thetas[0])
        circuit = quasar.Circuit.join_in_time([circuit, layer_circuit])
    else:
        thetas = thetas.reshape(2, time_step, -1)
        for idx, jump in enumerate(seq_jumps):
            layer_circuit = _get_layer_circuit(thetas[int(jump)][idx])
            circuit = quasar.Circuit.join_in_time([circuit, layer_circuit])
    # Translate from qcware-quasar to qiskit
    qiskit_circuit = translate(circuit)
    # qiskit_circuit.save_statevector()
    qiskit_circuit = qiskit.transpile(qiskit_circuit, optimization_level=3)
    c = qiskit.ClassicalRegister(num_qubits)
    qiskit_circuit.add_register(c)
    qiskit_circuit.barrier()
    qiskit_circuit.measure(qubit=range(num_qubits), cbit=c)
    return qiskit_circuit


def run_circuit_compound(
    circs,
    num_qubits,
    device_id,
    global_info,
    backend_name="quantinuum_H1-1",
):

    global_number_of_circuits_executed, global_hardware_run_results_dict = global_info
    results = np.zeros((len(circs), 2**num_qubits))

    global_number_of_circuits_executed += len(circs)
    num_measurements = 1000

    if "qiskit" in backend_name:
        backend = qiskit.Aer.get_backend("qasm_simulator")
        if backend_name == "qiskit_noiseless":
            measurement = qiskit.execute(circs, backend, shots=num_measurements)
        elif backend_name == "qiskit_noisy":
            # Error probabilities
            prob_1 = 0.001  # 1-qubit gate
            prob_2 = 0.01  # 2-qubit gate
            # Dylan's tunes error probabilities
            # prob_1 = 0  # 1-qubit gate
            # prob_2 = 3.5e-3   # 2-qubit gate

            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)

            # Add errors to noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ["h", "x", "ry"])
            noise_model.add_all_qubit_quantum_error(error_2, ["cz"])

            # Get basis gates from noise model
            basis_gates = noise_model.basis_gates
            measurement = qiskit.execute(
                circs,
                backend,
                basis_gates=basis_gates,
                noise_mode=noise_model,
                shots=num_measurements,
            )
        else:
            raise ValueError(f"Unexpected backend name {backend_name}")
        all_counts = measurement.result().get_counts()
    elif "quantinuum" in backend_name:
        # From docs: "Batches cannot exceed the maximum limit of 500 H-System Quantum Credits (HQCs) total"
        # Therefore batching is more or less useless on quantinuum
        from pytket import OpType
        from pytket.circuit import BasisOrder
        from pytket.extensions.qiskit import qiskit_to_tk
        from pytket.extensions.quantinuum import QuantinuumBackend

        outpath_stem = "_".join(
            [
                device_id,
                global_hardware_run_results_dict["model_type"],
                backend_name,
                global_hardware_run_results_dict["layer_type"],
                str(global_hardware_run_results_dict["epsilon"]),
                str(global_hardware_run_results_dict["batch_idx"]),
            ]
        )

        outpath_result_final = f"data/{outpath_stem}.json"
        outpath_handles = f"data/handles_{outpath_stem}.pickle"

        if Path(outpath_result_final).exists():
            # if precomputed results already present on disk, simply load
            print(f"Using precomputed counts from {outpath_result_final}")
            all_counts = json.load(open(outpath_result_final, "r"))["all_counts"]
        else:
            if backend_name == "quantinuum_H1-2E":
                backend = QuantinuumBackend(device_name="H1-2E")
            elif backend_name == "quantinuum_H1-2":
                backend = QuantinuumBackend(device_name="H1-2")
            elif backend_name == "quantinuum_H1-1E":
                backend = QuantinuumBackend(device_name="H1-1E")
            elif backend_name == "quantinuum_H1-1":
                backend = QuantinuumBackend(device_name="H1-1")
            else:
                raise ValueError(f"Unknown Quantinuum backend: {backend_name}")
            if Path(outpath_handles).exists():
                # if circuits already submitted, simply load from disk
                print(f"Using pickled handles from {outpath_handles}")
                handles = pickle.load(open(outpath_handles, "rb"))
            else:
                # otherwise, submit circuits and pickle handles
                circs_tk = [qiskit_to_tk(circ) for circ in circs]
                for idx, circ in enumerate(circs_tk):
                    circ.name = f"{outpath_stem}_{idx+1}_of_{len(circs)}"
                compiled_circuits = backend.get_compiled_circuits(
                    circs_tk, optimisation_level=2
                )

                #######################################################

                print("Before compilation")
                print(
                    f"Depths: {[circ.depth(filter_function=lambda x: x[0].name == 'cz') for circ in circs]}"
                )
                print("After conversion")
                print(f"Depths: {[circ.depth_by_type(OpType.CZ) for circ in circs_tk]}")
                print("After compilation")
                print(
                    f"Depths ZZMax+ZZPhase: {[circ.depth_by_type(OpType.ZZMax)+circ.depth_by_type(OpType.ZZPhase) for circ in compiled_circuits]}"
                )
                print(
                    f"Depths ZZMax: {[circ.depth_by_type(OpType.ZZMax) for circ in compiled_circuits]}"
                )
                print(
                    f"Depths ZZPhase: {[circ.depth_by_type(OpType.ZZPhase) for circ in compiled_circuits]}"
                )

                #######################################################

                handles = backend.process_circuits(
                    compiled_circuits, n_shots=num_measurements
                )
                pickle.dump(handles, open(outpath_handles, "wb"))
                print(f"Dumped handles to {outpath_handles}")
            # retrieve results from handles
            result_list = []

            with tqdm(total=len(handles), desc="#jobs finished") as pbar:
                for handle in handles:
                    while True:
                        status = backend.circuit_status(handle).status
                        if status.name == "COMPLETED":
                            result = backend.get_result(handle)
                            result_list.append(copy.deepcopy(result))
                            pbar.update(1)
                            break
                        else:
                            assert status.name in ["QUEUED", "RUNNING"]
                        time.sleep(1)
            global_hardware_run_results_dict["result_list"] = [
                x.to_dict() for x in result_list
            ]
            # convert from tket counts format to qiskit
            all_counts = [
                counter_to_dict(result.get_counts(basis=BasisOrder.dlo))
                for result in result_list
            ]
            global_hardware_run_results_dict["all_counts"] = all_counts
            # dump result on disk
            json.dump(global_hardware_run_results_dict, open(outpath_result_final, "w"))
    else:
        raise ValueError(f"Unexpected backend name {backend_name}")
    global_hardware_run_results_dict["batch_idx"] += 1

    # Post processing
    for j in range(len(circs)):
        if len(circs) == 1:
            measurementRes = all_counts
        else:
            measurementRes = all_counts[j]
        num_qubits = len(list(measurementRes)[0])
        filtered_counts = {f"{i:0{num_qubits}b}": 0 for i in range(2**num_qubits)}
        num_postselected = 0
        for bitstring, count in measurementRes.items():
            ham_weight = sum([int(x) for x in bitstring])
            if ham_weight == 0 or ham_weight == num_qubits:
                continue
            filtered_counts[bitstring] = count
            num_postselected += count
        results[j] = np.sqrt(
            [filtered_counts[k] / num_postselected for k in sorted(filtered_counts)]
        )
    return results, (
        global_number_of_circuits_executed,
        global_hardware_run_results_dict,
    )
