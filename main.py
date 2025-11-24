"""
Readout Error Mitigation Demonstration in Qiskit

This script demonstrates readout error mitigation by:
1. Creating a Bell state circuit
2. Running it on ideal, noisy, and mitigated simulators
3. Comparing the results with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from scipy.linalg import pinv


def create_bell_state_circuit():
    """
    Create a Bell state circuit: (|00> + |11>)/sqrt(2)
    
    Returns:
        QuantumCircuit: The Bell state circuit
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Apply Hadamard to create superposition
    qc.cx(0, 1)  # CNOT to create entanglement
    qc.measure_all()  # Measure both qubits
    return qc


def create_noise_model():
    """
    Create a realistic noise model with:
    - Depolarizing noise on gates
    - Readout errors on measurements
    
    Returns:
        NoiseModel: A noise model with gate and readout errors
    """
    noise_model = NoiseModel()
    
    # Add depolarizing error to single-qubit gates (H gate)
    depol_error_1q = depolarizing_error(0.01, 1)  # 1% depolarizing error
    noise_model.add_all_qubit_quantum_error(depol_error_1q, ['h'])
    
    # Add depolarizing error to two-qubit gates (CNOT)
    depol_error_2q = depolarizing_error(0.02, 2)  # 2% depolarizing error
    noise_model.add_all_qubit_quantum_error(depol_error_2q, ['cx'])
    
    # Add readout errors for each qubit
    # Readout error: probability of measuring 0 when state is 1, and vice versa
    readout_error_0 = ReadoutError([[0.95, 0.05],  # P(0|0)=0.95, P(1|0)=0.05
                                    [0.08, 0.92]])  # P(0|1)=0.08, P(1|1)=0.92
    
    readout_error_1 = ReadoutError([[0.93, 0.07],  # P(0|0)=0.93, P(1|0)=0.07
                                    [0.06, 0.94]])  # P(0|1)=0.06, P(1|1)=0.94
    
    noise_model.add_readout_error(readout_error_0, [0])
    noise_model.add_readout_error(readout_error_1, [1])
    
    return noise_model


def run_ideal_simulation(circuit, shots=1024):
    """
    Run the circuit on an ideal simulator (no noise).
    
    Args:
        circuit: QuantumCircuit to execute
        shots: Number of measurement shots
        
    Returns:
        dict: Counts dictionary
    """
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    return counts


def run_noisy_simulation(circuit, noise_model, shots=1024):
    """
    Run the circuit on a noisy simulator.
    
    Args:
        circuit: QuantumCircuit to execute
        noise_model: NoiseModel to apply
        shots: Number of measurement shots
        
    Returns:
        dict: Counts dictionary
    """
    simulator = AerSimulator(noise_model=noise_model)
    # Transpile to ensure gates match noise model
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(transpiled_circuit)
    return counts


def characterize_readout_errors(noise_model, num_qubits=2, shots=1024):
    """
    Characterize readout errors by running calibration circuits.
    
    Args:
        noise_model: NoiseModel with readout errors
        num_qubits: Number of qubits
        shots: Number of shots per calibration circuit
        
    Returns:
        numpy.ndarray: Assignment matrix (inverse for mitigation)
    """
    simulator = AerSimulator(noise_model=noise_model)
    
    # Generate calibration circuits: prepare each computational basis state
    # and measure it
    calibration_circuits = []
    calibration_states = []
    
    for state_idx in range(2 ** num_qubits):
        # Create circuit preparing state |state_idx>
        cal_circuit = QuantumCircuit(num_qubits, num_qubits)
        state_binary = format(state_idx, f'0{num_qubits}b')
        
        # Prepare the state
        for qubit_idx, bit in enumerate(state_binary):
            if bit == '1':
                cal_circuit.x(qubit_idx)
        
        cal_circuit.measure_all()
        calibration_circuits.append(cal_circuit)
        calibration_states.append(state_binary)
    
    # Run calibration circuits
    transpiled_cal_circuits = transpile(calibration_circuits, simulator)
    job = simulator.run(transpiled_cal_circuits, shots=shots)
    results = job.result()
    
    # Build assignment matrix: assignment_matrix[i][j] = P(measure j | prepare i)
    # Qiskit uses little-endian: '00' means qubit0=0, qubit1=0
    assignment_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    
    for i, state in enumerate(calibration_states):
        counts = results.get_counts(transpiled_cal_circuits[i])
        total_shots = sum(counts.values())
        
        for measured_state, count in counts.items():
            # Handle measurement format: remove spaces and take only the qubit measurement part
            # Format might be '11 00' (qubit results + classical bits) or '11'
            measured_state_clean = measured_state.split()[0] if ' ' in measured_state else measured_state
            # Convert binary strings to indices
            # Qiskit measurement results are in little-endian format
            # '00' = 0, '01' = 1, '10' = 2, '11' = 3
            measured_idx = int(measured_state_clean, 2)
            assignment_matrix[i][measured_idx] = count / total_shots
    
    # Compute the inverse (pseudo-inverse) of the assignment matrix for mitigation
    # P_measured = A × P_true, so P_true ≈ A^(-1) × P_measured
    mitigation_matrix = pinv(assignment_matrix)
    
    return mitigation_matrix


def run_mitigated_simulation(circuit, noise_model, mitigation_matrix, shots=1024):
    """
    Run the circuit on a noisy simulator and apply readout error mitigation.
    
    Args:
        circuit: QuantumCircuit to execute
        noise_model: NoiseModel to apply
        mitigation_matrix: Inverse assignment matrix for mitigation
        shots: Number of measurement shots
        
    Returns:
        dict: Mitigated counts dictionary
    """
    simulator = AerSimulator(noise_model=noise_model)
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    raw_counts = result.get_counts(transpiled_circuit)
    
    # Get number of qubits from the format of raw_counts keys
    # Handle format like '11 00' (qubit results + classical bits) or '11'
    first_key = list(raw_counts.keys())[0]
    first_key_clean = first_key.split()[0] if ' ' in first_key else first_key
    num_qubits = len(first_key_clean)
    num_states = 2 ** num_qubits
    
    # Convert raw counts to probability vector
    measured_probs = np.zeros(num_states)
    for state_str, count in raw_counts.items():
        # Handle measurement format: remove spaces and take only the qubit measurement part
        state_str_clean = state_str.split()[0] if ' ' in state_str else state_str
        state_idx = int(state_str_clean, 2)
        measured_probs[state_idx] = count / shots
    
    # Apply mitigation: P_true ≈ mitigation_matrix × P_measured
    true_probs = mitigation_matrix @ measured_probs
    
    # Ensure probabilities are non-negative and normalized
    true_probs = np.maximum(true_probs, 0)  # Clip negative values to 0
    true_probs = true_probs / np.sum(true_probs)  # Renormalize
    
    # Convert back to counts dictionary
    mitigated_counts_dict = {}
    for state_idx, prob in enumerate(true_probs):
        count = int(round(prob * shots))
        if count > 0:
            state_str = format(state_idx, f'0{num_qubits}b')
            mitigated_counts_dict[state_str] = count
    
    return mitigated_counts_dict


def normalize_counts(counts, shots):
    """
    Normalize counts to probabilities.
    
    Args:
        counts: Dictionary of counts
        shots: Total number of shots
        
    Returns:
        dict: Normalized probabilities
    """
    return {state: count / shots for state, count in counts.items()}


def plot_comparison(counts_ideal, counts_noisy, counts_mitigated, shots):
    """
    Create bar plots comparing the three probability distributions.
    
    Args:
        counts_ideal: Ideal simulation counts
        counts_noisy: Noisy simulation counts
        counts_mitigated: Mitigated counts
        shots: Total number of shots
    """
    # Normalize to probabilities
    probs_ideal = normalize_counts(counts_ideal, shots)
    probs_noisy = normalize_counts(counts_noisy, shots)
    probs_mitigated = normalize_counts(counts_mitigated, shots)
    
    # Get all possible states
    all_states = sorted(set(list(probs_ideal.keys()) + 
                           list(probs_noisy.keys()) + 
                           list(probs_mitigated.keys())))
    
    # Ensure all dictionaries have all states
    for state in all_states:
        probs_ideal.setdefault(state, 0.0)
        probs_noisy.setdefault(state, 0.0)
        probs_mitigated.setdefault(state, 0.0)
    
    states = all_states
    ideal_vals = [probs_ideal[state] for state in states]
    noisy_vals = [probs_noisy[state] for state in states]
    mitigated_vals = [probs_mitigated[state] for state in states]
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(states))
    width = 0.35
    
    # Plot 1: Ideal vs Noisy
    ax1 = axes[0]
    ax1.bar(x - width/2, ideal_vals, width, label='Ideal', alpha=0.8, color='green')
    ax1.bar(x + width/2, noisy_vals, width, label='Noisy', alpha=0.8, color='red')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Probability')
    ax1.set_title('Ideal vs Noisy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(states)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Noisy vs Mitigated
    ax2 = axes[1]
    ax2.bar(x - width/2, noisy_vals, width, label='Noisy', alpha=0.8, color='red')
    ax2.bar(x + width/2, mitigated_vals, width, label='Mitigated', alpha=0.8, color='blue')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Probability')
    ax2.set_title('Noisy vs Mitigated')
    ax2.set_xticks(x)
    ax2.set_xticklabels(states)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Ideal vs Mitigated
    ax3 = axes[2]
    ax3.bar(x - width/2, ideal_vals, width, label='Ideal', alpha=0.8, color='green')
    ax3.bar(x + width/2, mitigated_vals, width, label='Mitigated', alpha=0.8, color='blue')
    ax3.set_xlabel('State')
    ax3.set_ylabel('Probability')
    ax3.set_title('Ideal vs Mitigated')
    ax3.set_xticks(x)
    ax3.set_xticklabels(states)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('readout_mitigation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'readout_mitigation_comparison.png'")
    plt.show()


def main():
    """
    Main function to run the complete readout error mitigation demonstration.
    """
    print("=" * 60)
    print("Readout Error Mitigation Demonstration")
    print("=" * 60)
    
    shots = 1024
    
    # Step 1: Create Bell state circuit
    print("\n1. Creating Bell state circuit: (|00> + |11>)/sqrt(2)")
    circuit = create_bell_state_circuit()
    print("   Circuit created successfully!")
    
    # Step 2: Create noise model
    print("\n2. Creating noise model with gate and readout errors...")
    noise_model = create_noise_model()
    print("   Noise model created successfully!")
    
    # Step 3: Run ideal simulation
    print("\n3. Running ideal simulation (no noise)...")
    counts_ideal = run_ideal_simulation(circuit, shots=shots)
    print(f"   Ideal counts: {counts_ideal}")
    
    # Step 4: Run noisy simulation
    print("\n4. Running noisy simulation...")
    counts_noisy = run_noisy_simulation(circuit, noise_model, shots=shots)
    print(f"   Noisy counts: {counts_noisy}")
    
    # Step 5: Characterize readout errors
    print("\n5. Characterizing readout errors with calibration circuits...")
    mitigation_matrix = characterize_readout_errors(noise_model, num_qubits=2, shots=shots)
    print("   Readout error characterization complete!")
    
    # Step 6: Run mitigated simulation
    print("\n6. Running mitigated simulation...")
    counts_mitigated = run_mitigated_simulation(circuit, noise_model, mitigation_matrix, shots=shots)
    print(f"   Mitigated counts: {counts_mitigated}")
    
    # Step 7: Create visualizations
    print("\n7. Creating comparison plots...")
    plot_comparison(counts_ideal, counts_noisy, counts_mitigated, shots)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nIdeal counts: {counts_ideal}")
    print(f"Noisy counts: {counts_noisy}")
    print(f"Mitigated counts: {counts_mitigated}")
    
    # Calculate fidelities
    def calculate_fidelity(counts1, counts2, shots):
        """Calculate fidelity between two probability distributions."""
        probs1 = normalize_counts(counts1, shots)
        probs2 = normalize_counts(counts2, shots)
        all_states = set(list(probs1.keys()) + list(probs2.keys()))
        fidelity = sum(np.sqrt(probs1.get(s, 0) * probs2.get(s, 0)) for s in all_states)
        return fidelity ** 2
    
    fidelity_noisy = calculate_fidelity(counts_ideal, counts_noisy, shots)
    fidelity_mitigated = calculate_fidelity(counts_ideal, counts_mitigated, shots)
    
    print(f"\nFidelity (Ideal vs Noisy): {fidelity_noisy:.4f}")
    print(f"Fidelity (Ideal vs Mitigated): {fidelity_mitigated:.4f}")
    print(f"Improvement: {fidelity_mitigated - fidelity_noisy:.4f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

