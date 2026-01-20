# Spiking Radar/Sonar Pipeline (1D and 2D) - Step-by-step with Equations

This document explains the full pipeline implemented in this repo, with a strong focus on the `SpikingLIFDelayEstimator`. It is written to be report-ready and includes the key math for each stage.

## 1) Notation

- Sampling rate: $ f_s $, time step $ \Delta t = 1/f_s $.
- Speed of propagation: $ c $ (sound or light).
- Discrete-time index: $ n $, continuous time: $ t $.
- Transmit spikes: $ x[n] \in \{0,1\} $.
- Received spikes (single channel): $ r[n] \in \{0,1\} $.
- Binaural spikes: left $ r_L[n] $, right $ r_R[n] $.
- LIF membrane potential: $ V[n] $.
- LIF threshold: $ V_{th} $.
- Time constant: $ \tau_m $, decay factor $ \alpha = e^{-\Delta t/\tau_m} $.

Coordinate convention for 2D:
- Object position $ (x, y) $, where $ x $ is forward range, $ y $ is lateral (positive toward the left receiver).
- Receivers at $ (0, \pm d/2) $, where $ d $ is receiver spacing.

## 2) Transmitter: spike generation and waveform synthesis

### 2.1 Poisson spike generation

The transmitter generates a stochastic spike train using a Poisson model. In discrete time, this is implemented as a Bernoulli trial per sample:

$$
x[n] \sim \text{Bernoulli}(p), \quad p = \lambda \Delta t
$$

where $ \lambda $ is the desired spike rate (Hz). In the code, this is equivalent to
`rng.random(N) < spike_probability_per_sample`.

**Why Poisson?**
- Memoryless: spike times are independent, which is a common first-order model of irregular biological spiking.
- Maximum entropy: given only a mean rate, a Poisson process makes the fewest additional assumptions.
- Analytical tractability: expected counts and correlations are easy to compute.

The inter-spike interval is exponential:
$$
P(\text{ISI} > t) = e^{-\lambda t}
$$

### 2.2 Chirp template and baseband encoding

A linear chirp $ c(t) $ is used as the base waveform template. A standard continuous form is:

$$
c(t) = \cos\left(2\pi\left(f_0 t + \frac{K}{2} t^2\right)\right)
$$

where $ K = \frac{B}{T} $ is the chirp rate, $ B $ is bandwidth, and $ T $ is chirp duration.

The baseband spike-driven signal is the convolution of spikes and the chirp template:

$$
b[n] = (x * c)[n] = \sum_k x[k]\,c[n-k]
$$

### 2.3 Modulation

The baseband is modulated onto a carrier:

$$
s_{tx}[n] = b[n] \cos(2\pi f_c n \Delta t)
$$

This mirrors the `SpikingRadarTransmitter` pipeline.

## 3) Environment / propagation

### 3.1 1D propagation (monostatic echo)

Round-trip time-of-flight:
$$
\tau = \frac{2R}{c}
$$

Discrete delay: $ d = \lfloor \tau f_s \rceil $.

Received signal (with attenuation $ a $ and noise $ \eta $):
$$
s_{rx}[n] = a\,s_{tx}[n - d] + \eta[n]
$$

### 3.2 2D propagation (binaural)

Let object position be $ p = (x, y) $.

Transmitter at $ (0,0) $, receivers at $ (0, \pm d/2) $:

$$
L_L = \|p - (0,0)\| + \|p - (0, d/2)\|
$$
$$
L_R = \|p - (0,0)\| + \|p - (0, -d/2)\|
$$

Delays:
$$
\tau_L = \frac{L_L}{c}, \quad \tau_R = \frac{L_R}{c}
$$

Received signals:
$$
s_{rx,L}[n] = a\,s_{tx}[n - d_L] + \eta_L[n]
$$
$$
s_{rx,R}[n] = a\,s_{tx}[n - d_R] + \eta_R[n]
$$

with $ d_{L/R} = \lfloor \tau_{L/R} f_s \rceil $.

## 4) Receiver: demodulation, filtering, matched filter, spike recovery

### 4.1 Demodulation

Multiply by carrier:
$$
s_{bb}[n] = s_{rx}[n] \cos(2\pi f_c n \Delta t)
$$

### 4.2 Low-pass filtering

An IIR low-pass is applied to recover the baseband envelope.

### 4.3 Matched filtering

The matched filter is the time-reversed chirp template:

$$
h[n] = c[-n]
$$

The matched filter output is:

$$
y[n] = (s_{bb} * h)[n]
$$

This is equivalent to a correlation with the template and maximizes SNR in white noise.

### 4.4 Spike recovery (thresholding)

The matched-filter output is normalized and thresholded:

$$
r[n] = \mathbb{I}\left(\frac{y[n]}{\max|y|} > \theta\right)
$$

This yields the spikes used by the LIF estimator.

## 5) SpikingLIFDelayEstimator (core)

This is the main computation stage: a bank of LIF coincidence detectors (Jeffress model).

### 5.1 LIF dynamics (continuous time)

$$
\tau_m \frac{dV(t)}{dt} = -V(t) + w_x x(t) + w_y r(t)
$$

When $ V(t) \ge V_{th} $, a spike is emitted and $ V $ is reset to 0.

### 5.2 LIF dynamics (discrete time)

$$
V[n+1] = \alpha V[n] + w_x x[n] + w_y r[n]
$$

where:
$$
\alpha = e^{-\Delta t/\tau_m}
$$

In the implementation, a **coincidence gate** is applied to avoid baseline firing:

$$
r_{gated}[n] = r[n] \cdot x_{delayed}[n]
$$

$$
V[n+1] = \alpha V[n] + w_x x_{delayed}[n] + w_y r_{gated}[n]
$$

This ensures spikes are only counted on true TX/RX coincidences.

### 5.3 Jeffress coincidence model for range (distance)

Each neuron corresponds to a delay hypothesis $ d_i $ (samples):

$$
x_i[n] = x[n - d_i]
$$

Coincidence at neuron $ i $ is driven by:
$$
V_i[n+1] = \alpha V_i[n] + w_x x_i[n] + w_y r_i[n]
$$

The spike count $ N_i $ approximates a correlation:

$$
N_i \propto \sum_n x[n - d_i]\,r[n] = R_{xr}[d_i]
$$

Estimated delay:
$$
\hat{d} = \arg\max_i N_i
$$

Estimated time-of-flight:
$$
\hat{\tau} = \hat{d}\,\Delta t
$$

Estimated distance:
$$
\hat{R} = \frac{c \hat{\tau}}{2}
$$

### 5.4 Jeffress coincidence model for angle (ITD)

For binaural input, each neuron corresponds to a signed interaural delay $ \delta_i $:

$$
r_R[n + \delta_i] \quad \text{aligned with} \quad r_L[n]
$$

The coincidence count approximates a cross-correlation:

$$
N_i \propto \sum_n r_L[n]\,r_R[n+\delta_i] = R_{LR}[\delta_i]
$$

The best ITD is:
$$
\hat{\delta} = \arg\max_i N_i
$$

Convert to time:
$$
\widehat{\text{ITD}} = \hat{\delta}\,\Delta t
$$

Geometry relates ITD to angle:

$$
\theta = \arcsin\left(\frac{c\,\widehat{\text{ITD}}}{d}\right)
$$

where $ d $ is receiver spacing. The sign of $ \theta $ is set by which receiver is leading.

### 5.5 Relationship to classical cross-correlation

For spike trains, cross-correlation is:

$$
R_{xr}[k] = \sum_n x[n]\,r[n+k]
$$

The LIF bank implements a *spiking approximation* of this correlation:

1. Each neuron represents one lag.
2. Coincidence spiking is the discrete analog of correlation peaks.
3. The argmax across neurons recovers the peak lag.

This provides a direct neural interpretation of matched filtering and correlation.

## 6) Output stage

The output packages the estimated distance and angle along with the underlying signals. In 2D:

- Estimated range from averaged left/right delay.
- Estimated angle from ITD.
- Data stored in `SpikingRadarResult_2D` for reporting/plots.

## 7) Resolution limits (useful for reporting)

Time resolution:
$$
\Delta t = 1/f_s
$$

Range resolution:
$$
\Delta R \approx \frac{c}{2f_s}
$$

Angle resolution (small angles):
$$
\Delta \theta \approx \frac{c}{d\,f_s}
$$

At larger angles, divide by $ \cos\theta $ due to the arcsin nonlinearity.

## 8) Summary (pipeline as a sequence)

1. **Generate spikes** using Poisson model (or deterministic spikes for debugging).
2. **Convolve** spikes with chirp template to form baseband.
3. **Modulate** baseband onto a carrier.
4. **Propagate** through environment with delay, attenuation, and noise.
5. **Demodulate** and **low-pass** to recover baseband.
6. **Matched filter** to maximize echo SNR.
7. **Recover spikes** by thresholding.
8. **LIF range bank**: coincidence detection vs delayed TX.
9. **LIF ITD bank**: coincidence detection between left/right channels.
10. **Estimate distance and angle** from peak spike counts.

This pipeline is mathematically consistent with classical radar/sonar correlation while remaining biologically interpretable through Jeffress-style coincidence detection.
