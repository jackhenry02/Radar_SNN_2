
# Design Specification: Resonant-and-Fire (RF) Cochlear Receiver

This document details the transition from a classical Matched Filter receiver to an asynchronous, spiking "Digital Cochlea" using **Resonator-and-Fire (RF)** neurons. This architecture shifts the processing from 1D time-domain signals to a **2D Tonotopic Map** (Frequency $\times$ Time).

## 1. System Comparison: Classical vs. Biological

| Component | Classical Receiver (Current) | RF Cochlear Receiver (Proposed) |
| --- | --- | --- |
| **Input** | Modulated/Carrier Wave | Modulated/Carrier Wave |
| **Demodulation** | Math-based (Mixer + LPF) | **LIF Envelope Detector** (Rectification + Integration) |
| **Frequency Analysis** | Matched Filter (knows exact pulse) | **RF Neuron Bank** (Tuned to frequencies) |
| **Output** | Single Spike Train | **Parallel Tonotopic Spike Trains** |

---

## 2. Stage 1: The "LIF" Demodulator & Envelope Detector

Instead of a digital mixer, we use the non-linear dynamics of a Leaky Integrate-and-Fire (LIF) neuron to extract the envelope.

### The Physics

The **Inner Hair Cells** in the ear act as half-wave rectifiers. In your SNN, we replicate this by taking the positive part of the signal and feeding it into an LIF neuron with a **slow decay**.

### The Discrete Equation

For the input signal $x[t]$ (the modulated wave):

1. **Rectification:** $I[t] = \max(0, x[t])$
2. **Integration (Envelope):** $$V_{env}[t+1] = \beta_{slow} V_{env}[t] + w_{in} I[t]$$


> **Note:** Here, $\beta_{slow}$ is chosen to be much larger than your carrier period. This "smooths" the high-frequency oscillations, leaving only the "hump" of the chirp. This $V_{env}$ is then fed as the input current to the RF neurons.

---

## 3. Stage 2: The Resonator-and-Fire (RF) Bank

The core of the new receiver is a bank of $N$ neurons, each tuned to a specific frequency $f_i$  present in your chirp.

### The 2nd-Order Dynamics (Izhikevich-style)

To achieve resonance, we need a 2-variable system. $V$ represents the membrane potential, and  $U$ represents a recovery variable (ion channel current) that creates the oscillation.

#### Discrete-Time State Update:

For a neuron tuned to frequency $\omega_i = 2\pi f_i$:
$$V[t+1] = V[t] + \Delta t \left( A(V[t] - V_{rest}) - U[t] + V_{env}[t] \right)$$
$$U[t+1] = U[t] + \Delta t \left( B(V[t] - V_{rest}) \right)$$

* **Tuning:** The frequency of oscillation is determined by the parameters $A$ and $B$. Specifically, the resonant frequency is approximately $f \approx \frac{1}{2\pi}\sqrt{B}$.
* **Damping:** The parameter $A$ controls how "sharp" the filter is (the Q-factor). A small  $A$ makes the neuron a very narrow-band filter; a larger  $A$ makes it more robust to slight frequency shifts.

### Spiking Logic

If  $V[t] > V_{thr}$:

1. **Emit Spike:** $S_i[t] = 1$
2. **Reset:** $V \leftarrow V_{reset}$, $U \leftarrow U + d$ (where $d$ is a damping constant).

---

## 4. Stage 3: The Tonotopic Output

Instead of one spike at the "peak" of a matched filter, your receiver now outputs a **matrix of spikes**.

* As the chirp sweeps from **High** to **Low** frequency:
* The **High-$f$ neurons** spike first.
* The **Mid-$f$ neurons** spike next.
* The **Low-$f$ neurons** spike last.



This creates a "diagonal" pattern in your spike plot. Your **Jeffress Coincidence Bank** now looks for this *specific sequence* of spikes across multiple channels to confirm a distance, making it incredibly robust to random noise.

---

## 5. Implementation Experiment Plan

To ensure a stable transition, follow these three experiments:

### Experiment A: The Frequency Sweep Check

* **Input:** Your standard modulated chirp.
* **Metric:** Plot the $V_{mem}$ of 5 RF neurons tuned to different parts of the chirp.
* **Success:** You should see each neuron's $V_{mem}$ "hum" and peak only when the chirp passes through its specific frequency band.

### Experiment B: Single-Ear Spiking

* **Input:** Echo from 5 meters.
* **Metric:** Raster plot of the Tonotopic Bank.
* **Success:** A clear "diagonal" line of spikes representing the temporal order of the frequency sweep.

### Experiment C: Coincidence Integration

* **Action:** Feed the parallel channels into your existing `Processing` unit.
* **Adjustment:** You will need to sum or "AND" the signals from the different frequency channels.
* **Success:** The distance estimate remains accurate within $<1\%$ RMSE.

---

## 6. Biological Justification for Report

> "By utilizing Resonator-and-Fire neurons, the receiver replaces the non-biological Matched Filter with a **biophysically plausible tonotopic map**. This mimics the mechanical resonance of the **Basilar Membrane**, where frequency selection is an inherent property of the neural dynamics rather than an external algorithmic step."


