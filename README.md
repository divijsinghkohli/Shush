# Readout Error Mitigation in Qiskit

This project demonstrates readout error mitigation in quantum computing using Qiskit. It shows how to characterize and mitigate measurement errors that occur when reading out quantum states.

## What is Readout Error?

Readout error (also called measurement error) occurs when the measurement process itself introduces errors. When measuring a qubit, there's a probability that:

- A qubit in state |0⟩ is measured as |1⟩ (false positive)
- A qubit in state |1⟩ is measured as |0⟩ (false negative)

These errors are characterized by an **assignment matrix** (also called a confusion matrix) where:
- `assignment_matrix[i][j]` = Probability of measuring state `j` when the qubit was actually in state `i`

For example, if a qubit has:
- 95% chance of correctly measuring |0⟩ when it's |0⟩
- 5% chance of incorrectly measuring |1⟩ when it's |0⟩
- 8% chance of incorrectly measuring |0⟩ when it's |1⟩
- 92% chance of correctly measuring |1⟩ when it's |1⟩

Then the assignment matrix would be:
```
[[0.95, 0.05],
 [0.08, 0.92]]
```

## How Error Mitigation Works

Readout error mitigation works in two steps:

### 1. Characterization
We run **calibration circuits** that prepare each computational basis state (|00⟩, |01⟩, |10⟩, |11⟩) and measure them. By comparing what we prepare with what we measure, we can build the assignment matrix that describes the readout errors.

### 2. Mitigation
Once we have the assignment matrix, we can invert it (or use it to solve a linear system) to estimate what the true probabilities were, given the noisy measurement results. This is done using a **readout mitigator** that applies the inverse transformation to the measured counts.

The mathematical relationship is:
```
P_measured = Assignment_Matrix × P_true
```

To recover the true probabilities:
```
P_true ≈ Assignment_Matrix^(-1) × P_measured
```

## How This Code Implements Mitigation

This implementation uses the following approach:

1. **Bell State Circuit**: Creates the entangled state (|00⟩ + |11⟩)/√2, which ideally should only produce |00⟩ or |11⟩ measurements.

2. **Noise Model**: Creates a realistic noise model with:
   - **Gate errors**: Depolarizing noise on Hadamard (1%) and CNOT (2%) gates
   - **Readout errors**: Independent readout errors for each qubit (5-8% error rates)

3. **Three Simulation Modes**:
   - **Ideal**: No noise, shows the perfect quantum behavior
   - **Noisy**: Includes both gate and readout errors
   - **Mitigated**: Applies readout error mitigation to the noisy results

4. **Calibration Process**:
   - Generates 4 calibration circuits (one for each basis state: |00⟩, |01⟩, |10⟩, |11⟩)
   - Runs each circuit on the noisy simulator
   - Builds the assignment matrix from the calibration results
   - Computes the pseudo-inverse of the assignment matrix for mitigation

5. **Mitigation Application**:
   - Runs the Bell state circuit on the noisy simulator
   - Converts noisy measurement counts to a probability vector
   - Applies the mitigation matrix: `P_true ≈ mitigation_matrix × P_measured`
   - Clips negative probabilities and renormalizes to get valid probabilities
   - Converts back to counts dictionary

6. **Visualization**: Creates three comparison plots:
   - Ideal vs Noisy: Shows how noise affects the results
   - Noisy vs Mitigated: Shows the improvement from mitigation
   - Ideal vs Mitigated: Shows how close mitigation gets us to the ideal result

## Expected Results

For a Bell state (|00⟩ + |11⟩)/√2:

- **Ideal simulation**: Should show approximately 50% probability for |00⟩ and 50% for |11⟩, with near-zero probability for |01⟩ and |10⟩.

- **Noisy simulation**: Will show:
  - Reduced probabilities for |00⟩ and |11⟩ (due to readout errors)
  - Non-zero probabilities for |01⟩ and |10⟩ (due to readout errors flipping bits)
  - Overall lower fidelity compared to ideal

- **Mitigated simulation**: Should show:
  - Restored probabilities closer to 50% for |00⟩ and |11⟩
  - Reduced probabilities for |01⟩ and |10⟩
  - Higher fidelity compared to noisy results
  - Results closer to the ideal case

The mitigation won't be perfect because:
1. Gate errors (depolarizing noise) cannot be corrected by readout mitigation alone
2. Statistical noise from finite shot counts
3. The mitigation process itself introduces some approximation errors

However, you should see a clear improvement in the mitigated results compared to the noisy results.

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Create a Bell state circuit
2. Run simulations (ideal, noisy, mitigated)
3. Print the counts for each simulation
4. Generate comparison plots saved as `readout_mitigation_comparison.png`
5. Display fidelity metrics

## Output

The script prints:
- `counts_ideal`: Measurement counts from ideal simulation
- `counts_noisy`: Measurement counts from noisy simulation  
- `counts_mitigated`: Measurement counts after readout error mitigation

It also calculates and displays:
- Fidelity between ideal and noisy results
- Fidelity between ideal and mitigated results
- Improvement from mitigation

## Files

- `main.py`: Main script with all the implementation
- `requirements.txt`: Python package dependencies
- `README.md`: This file

## Dependencies

- `qiskit`: Core Qiskit framework
- `qiskit-aer`: Qiskit Aer simulator (includes noise models)
- `matplotlib`: For plotting comparison graphs
- `numpy`: For numerical operations
- `scipy`: For matrix pseudo-inverse computation

## Notes

- This implementation uses modern Qiskit APIs (no deprecated functions)
- Uses `qiskit-aer` for simulators and noise models
- Uses `CorrelatedReadoutMitigator` for readout error mitigation
- The code is designed to run end-to-end without modification

