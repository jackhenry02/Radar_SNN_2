This is exactly the right instinct 👍
You’re making a *researcher’s* move now: **simplify the signal until the computation is undeniable**, then add complexity.

I’ll answer in two parts:

1. **Yes / no on swapping Poisson spikes → pulse (with a clear recommendation)**
2. **A fully mathematical, neuron-by-neuron construction of the simplest possible delay-estimating SNN**

No code yet. No snnTorch yet. Just mechanics you can put straight into your report.

---

## 1. Should you replace the Poisson spike generator with a pulse?

### Short answer

**Yes — temporarily — and only for the SNN development phase.**

### Why this is the right move

Right now, Poisson spiking introduces **three confounds** at once:

* stochastic timing jitter
* rate variability
* trial-to-trial randomness

That’s great later, but it makes *first-principles* reasoning hard.

A **single deterministic pulse**:

* isolates delay as the only unknown
* makes the SNN’s computation visually obvious
* lets you derive everything analytically
* makes debugging trivial

You are not “dumbing down the problem” — you’re **identifying the core computation**.

### Important framing (for your report)

You should explicitly say:

> “We begin with a deterministic pulse-based waveform to isolate temporal processing. Stochastic spike trains are reintroduced later as a robustness test.”

That’s methodologically *strong*, not weak.

### What to use instead

Use:

* a **single transmit spike**, or
* a **short burst** (e.g. 3–5 spikes at fixed spacing)

This still respects the spiking paradigm.

---

## 2. The mathematically minimal spiking delay estimator

Let’s build the *simplest possible* spiking network that estimates time-of-flight.

This will be your **foundation model**.

---

## 3. Problem definition (formal)

You transmit a known spike pattern:
$$
x(t) = \sum_k \delta(t - t_k)
$$

You receive a delayed and possibly attenuated version:
$$
y(t) = \sum_k \delta(t - t_k - \Delta)
$$

Goal:
$$
\text{Estimate } \Delta
$$

This is classical radar in spike form.

---

## 4. Classical solution (for comparison)

Cross-correlation:
$$
R(\tau) = \int x(t),y(t+\tau),dt
$$

Maximizer:
$$
\hat{\Delta} = \arg\max_\tau R(\tau)
$$

You already know this.

---

## 5. Spiking solution: intuition first

Each neuron in the SNN will:

* represent a **hypothesized delay** $ \tau_i $
* fire strongly only when spikes from $x(t)$ and $y(t)$ arrive **together**

That is a **coincidence detector**.

---

## 6. Neuron model (mathematical)

We use a standard LIF neuron:

$$
\tau_m \frac{dV_i(t)}{dt}
= - V_i(t) + w_x x(t) + w_y y(t - \tau_i)
  $$

with:

* threshold $ V_{th} $
* reset to 0 after spike

No learning. Fixed weights.

---

## 7. Why this neuron computes correlation

Consider one transmit spike at $ t_0 $:

* $x(t)$ contributes a jump at $ t_0 $
* $y(t - \tau_i)$ contributes a jump at $ t_0 + \Delta - \tau_i $

### Case 1: Correct delay hypothesis

If:
$$
\tau_i = \Delta
$$

Then both inputs arrive **simultaneously** at $ t_0 $

$$
V_i(t_0^+) = w_x + w_y
$$

If this exceeds threshold → spike.

---

### Case 2: Incorrect delay hypothesis

If:
$$
\tau_i \neq \Delta
$$

Inputs arrive separated in time:

* membrane decays
* threshold not reached

No spike.

---

## 8. Population code over delay

Define delays:
$$
\tau_i \in {\tau_{\min}, \dots, \tau_{\max}}
$$

Each neuron corresponds to a **range bin**:
$$
r_i = \frac{c,\tau_i}{2}
$$

The estimated delay is:
$$
\hat{\Delta} = \tau_{i^*}
\quad\text{where}\quad
i^* = \arg\max_i N_i
$$

where $N_i$ = spike count of neuron $i$.

This is directly interpretable.

---

## 9. Interpretation (this is gold for your report)

You can explicitly say:

> “The network implements a discretized cross-correlation by mapping delay hypotheses onto a population of coincidence-detecting neurons.”

And also:

> “Each neuron behaves as a matched filter tuned to a specific propagation delay.”

This directly addresses the “black box” concern.

---

## 10. Why this scales naturally later

Once this works, you can:

* reintroduce Poisson spike trains
* add noise in the environment
* use spike *rates* instead of single spikes
* extend to frequency-modulated pulse trains
* add multiple receiver channels (angle)

But the **core computation doesn’t change**.

---

## 11. What I suggest you do next (very concrete)

Next steps (in this order):

1. Replace Poisson generator with **single pulse**
2. Plot:

   * transmit spike
   * received spike
3. Plot classical cross-correlation (ground truth)
4. Derive expected neuron firing analytically (you now have this)
5. Then implement the LIF coincidence bank

👉 Don’t implement learning. Don’t optimize. Just *demonstrate computation*.

---

## 12. If you want, next we can:

* Formalize spike timing tolerance (membrane time constant vs jitter)
* Extend to burst coding instead of single pulse
* Translate this directly into `snntorch` equations
* Help you write the **Methods** section cleanly in LaTeX

If you say:

> “Let’s formalize spike timing tolerance”
> or
> “Let’s write this as a Methods subsection”

we’ll continue cleanly from here.
