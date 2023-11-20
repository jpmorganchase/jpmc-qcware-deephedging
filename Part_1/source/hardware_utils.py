# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co and QC Ware
import os
import sys

sys.path.append("..")
sys.path.append("../..")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import qiskit
import quasar
from qcware_transpile.translations.quasar.to_qiskit import translate
from qio import loader
from source import config
from source.qnn import get_butterfly_idxs, get_pyramid_idxs
from tqdm import tqdm

# fix for older versions of Qiskit
if qiskit.__version__ <= "0.37.1":
    import qiskit.providers.aer.noise as noise
else:
    import qiskit_aer.noise as noise


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


def prepare_circuit(input, params, loader_layout="parallel", layer_layout="butterfly"):
    """
     This function prepares a quantum circuit for a given input and set of parameters,
     which can later be executed on quantum hardware. The function takes three inputs:
    input, params, and optional arguments `loader_layout` and `layer_layout`. The
    function first determines the number of qubits in the circuit based on the length
    of the input array. It then creates a loader circuit using the loader function
    provided in the loader_layout format. The function then calls a `_get_layer_circuit()`
    function that creates a circuit layer based on the params array and the layer_layout
    format. The circuit layer is constructed using the RBS gate defined using `quasar.Gate.RBS()`.
    The `quasar.Gate.RBS()` function is called with a negative value of the params array elements
    (so as to take care of qiskit's reverse qubits notation) and added to the circuit layer for
    each gate in the `rbs_idxs` array. The function then combines the loader and layer circuits
    into a single `quasar.Circuit` object, which is then translated to a Qiskit circuit using the
    `translate()` function. The Qiskit circuit is then optimized and measured, and the function
     returns the resulting circuit.
    """

    def _get_layer_circuit():
        _params = np.array(params).astype("float")
        if layer_layout == "butterfly":
            rbs_idxs = get_butterfly_idxs(num_qubits, num_qubits)
        elif layer_layout == "pyramid":
            rbs_idxs = get_pyramid_idxs(num_qubits, num_qubits)
        circuit_layer = quasar.Circuit()
        idx_angle = 0
        for gates_per_timestep in rbs_idxs[::-1]:
            for gate in gates_per_timestep:
                circuit_layer.add_gate(
                    quasar.Gate.RBS(theta=-_params[::-1][idx_angle]), tuple(gate)
                )
                idx_angle += 1
        return circuit_layer

    num_qubits = len(input)
    loader_circuit = loader(
        np.array(input), mode=loader_layout, initial=True, controlled=False
    )
    layer_circuit = _get_layer_circuit()
    circuit = quasar.Circuit.join_in_time([loader_circuit, layer_circuit])
    # Translate from qcware-quasar to qiskit
    qiskit_circuit = translate(circuit)

    # qiskit_circuit.save_statevector()

    qiskit_circuit = qiskit.transpile(qiskit_circuit, optimization_level=3)
    c = qiskit.ClassicalRegister(num_qubits)
    qiskit_circuit.add_register(c)
    qiskit_circuit.barrier()
    qiskit_circuit.measure(qubit=range(num_qubits), cbit=c)
    return qiskit_circuit


def run_circuit(circs, num_qubits, device_id, backend_name):
    """
    This function accepts a list of `qiskit` circuits (which need to have the same qubit count), the number
    of qubits in each of the given circuit and the backend_name. The circuits are assumed to be hamming-weights
    preserving circuit used as part of quantum orthogonal layers. The function prepares a numpy array called `results` containing
    the output vector in unary basis. Currently this function supports two backends `qiskit` and `quantinuum`. The results
    run on `quantinuum`-based backends are stored using global variables `global_number_of_circuits_executed` and
    `global_hardware_run_results_dict`.This function then runs the circuit on the selected backend,
    and returns the result of the computation. If the backend is 'qiskit', the function uses the Qiskit Aer simulator to simulate
    the circuit and compute the result. If the backend is 'quantinuum', the function uses the Quantinuum simulator to perform the computation.
    Note that this function assumes that the user has already installed the required packages for the specified backend, and that the backend is properly
    configured and accessible.
    """
    global_number_of_circuits_executed = config.global_number_of_circuits_executed
    global_hardware_run_results_dict = config.global_hardware_run_results_dict
    input_size = num_qubits
    results = np.zeros((len(circs), input_size))

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
    # Discard bitstrings that do not correspond to unary encoding (not Hamming weight 1)
    # We build a dictionary with all unary bitstrings and only add counts corresponding to unary bitstrings
    # Note: f"{2**i:0{input_size}b}" converts 2**i to its binary string representation.
    for j in range(len(circs)):
        measurementRes = all_counts[j]
        num_postselected = 0
        filtered_counts = {f"{2**i:0{input_size}b}": 0 for i in range(input_size)}
        for bitstring, count in measurementRes.items():
            if sum([int(x) for x in bitstring]) != 1:
                continue
            filtered_counts[bitstring] += count
            num_postselected += count
        results[j] = [
            filtered_counts[k] / num_postselected for k in sorted(filtered_counts)
        ][::-1]

    config.global_number_of_circuits_executed = global_number_of_circuits_executed
    config.global_hardware_run_results_dict = global_hardware_run_results_dict
    return results
