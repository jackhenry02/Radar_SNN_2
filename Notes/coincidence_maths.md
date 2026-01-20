Perfect — this is exactly the right next step.
We’ll do this **slow, explicit, and defensible**, so you can lift chunks straight into your report.

I’ll structure this as:

1. Discrete-time LIF model (no ambiguity)
2. Membrane response to two spikes
3. Coincidence condition (derived, not guessed)
4. Parameter selection strategy
5. What range to test — and **where wave speed actually matters (and where it doesn’t)**

---

## 1. Discrete-time LIF neuron (canonical form)

We assume a fixed simulation timestep $ \Delta t $.

Define membrane potential $ V_i[n] $ for neuron $ i $:

$$
V_i[n+1]
=

\alpha V_i[n]
+
w_x x[n]
+
w_y y[n - d_i]
-

S_i[n] V_{\text{reset}}
$$

where:

* $ \alpha = e^{-\Delta t/\tau_m} $
* $ x[n] \in {0,1} $ transmit spike
* $ y[n] \in {0,1} $ received spike
* $ d_i $ = delay (in timesteps) for neuron $ i $
* $ S_i[n] = \mathbb{I}(V_i[n] \ge V_{th}) $
* reset is typically to 0

No refractory period for now (simplifies analysis).

---

## 2. Response to two spikes separated by Δn

Assume:

* one transmit spike at $ n = 0 $
* one received spike at $ n = \Delta $

Neuron $ i $ hypothesizes delay $ d_i $.

So it receives:

* spike from $x$ at $ n = 0 $
* spike from $y$ at $ n = \Delta - d_i $

---

### Case of interest: **near coincidence**

Let:
$$
\delta = \Delta - d_i
$$

This is the **timing mismatch**.

---

### Membrane evolution

At $ n = 0 $:
$$
V[1] = w_x
$$

At $ n = \delta $:
$$
V[\delta+1]
=

\alpha^{\delta} w_x + w_y
$$

This is the **peak membrane potential** before any reset.

---

## 3. Coincidence firing condition (key result)

The neuron fires **if and only if**:

$$
\alpha^{|\delta|} w_x + w_y ;\ge; V_{th}
$$

This is your core equation.

---

### Exact coincidence (ideal case)

If $ \delta = 0 $:

$$
V_{\text{coinc}} = w_x + w_y
$$

Require:
$$
w_x + w_y > V_{th}
$$

---

### Non-coincident case (suppression)

For all $ |\delta| \ge \delta_{\min} $:

$$
\alpha^{\delta_{\min}} w_x + w_y < V_{th}
$$

This inequality **defines your timing selectivity window**.

---

## 4. Solving for the coincidence window

Let:

* desired tolerance = $ \delta_{\max} $ timesteps

Solve:
$$
\alpha^{\delta_{\max}} w_x + w_y = V_{th}
$$
Rearrange:
$$
\delta_{\max}
=

\frac{
\ln\left(\frac{V_{th}-w_y}{w_x}\right)
}{
\ln \alpha
}
$$

Since $ \alpha < 1 $, denominator is negative → correct sign.

---

### Interpretation (this is report-worthy)

* **Smaller ( \tau_m )** → sharper coincidence window
* **Larger weights** → broader tolerance
* **Higher threshold** → more selective neuron

This is *precise*, not heuristic.

---

## 5. Choosing parameters (practical + principled)

### Step 1: Fix time resolution

You choose:
$$
\Delta t = \text{simulation timestep}
$$

Typical values:

* Sonar (audio): $ \Delta t = 1\text{–}10,\mu s $
* Radar (RF envelope / baseband): $ \Delta t = 1\text{–}10,ns $

But **this is numerical**, not biological.

---

### Step 2: Choose desired delay resolution

Let:
$$
\Delta \tau_{\text{bin}} = k,\Delta t
$$

Common choice:

* $ k = 1\text{–}3 $

You want neurons to fire **only within one delay bin**.

---

### Step 3: Choose membrane time constant

Use:
$$
\tau_m \approx (1\text{–}3),\Delta \tau_{\text{bin}}
$$

This guarantees:

* coincidence-sensitive
* no long integration tails

---

### Step 4: Choose weights and threshold

A simple, robust choice:

$$
w_x = w_y = 1
\qquad
V_{th} = 1.5
$$

This ensures:

* single spike never fires neuron
* exact coincidence always fires
* tolerance controlled only by $ \tau_m $

You can **prove** this satisfies the inequalities above.

---

## 6. What range of delays should you test?

This is where physics *enters*, but only here.

### Delay → range mapping

$$
r = \frac{c,\tau}{2}
$$

So:

* **Speed of sound** $ c \approx 343,\text{m/s} $
* **Speed of light** $ c \approx 3\times10^8,\text{m/s} $

---

### Important insight (this matters)

> **The neuron dynamics do *not* depend on wave speed.**

Wave speed only affects:

* how delays map to physical distance
* how wide a delay range is physically meaningful

The SNN itself doesn’t “know” if it’s sonar or radar.

---

### Example ranges

| Modality | Max range | Max delay |
| -------- | --------- | --------- |
| Sonar    | 10 m      | ≈ 58 ms   |
| Radar    | 10 m      | ≈ 67 ns   |

So:

* sonar → many more timesteps, slow simulation
* radar → very fine timestep, short windows

For **results quickly**:
👉 **Start with sonar-scale delays**, even if your end goal is radar.

That’s completely legitimate.

---

## 7. Suggested test grid (concrete)

I recommend:

* Fix $ \Delta t $
* Test delays:
  $$
  d_i \in [0, D_{\max}]
  \quad\text{with}\quad
  D_{\max} = 50\text{–}200 \text{ bins}
  $$

Plot:

* spike count vs $ d_i $
* compare to correlation curve

You’ll get a *beautiful* match.

---

## 8. What we achieved (important checkpoint)

You now have:

* a **derived coincidence condition**
* exact formulas for timing tolerance
* parameter choices justified mathematically
* a clean separation between physics and computation

This is already **publishable-level reasoning** for a methods section.

---

## 9. Next steps (choose one)

If you want, next we can:

1. **Add Poisson spike trains and analyze robustness**
2. **Translate this exactly into `snntorch`**
3. **Add inhibitory competition (WTA)**
4. **Write this as a LaTeX Methods subsection**
5. **Extend from pulse → burst coding**

Just tell me which one you want to do next.
