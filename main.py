from src.config import PhysicsConfig, RadarConfig
from src.signal_generator import SignalGenerator
from src.transmitter import Transmitter
from src.environment import Environment
from src.dsp import Receiver
from src.detector import ClassicalDetector
import matplotlib.pyplot as plt

def run_simulation():
    # 1. Setup
    p_cfg = PhysicsConfig(wave_type="sound") # Easy toggle to "light"
    r_cfg = RadarConfig()
    
    # 2. Initialize Modules
    gen = SignalGenerator(p_cfg, r_cfg)
    chirp_template = gen.generate_chirp_template()
    
    tx = Transmitter(r_cfg, chirp_template)
    env = Environment(p_cfg)
    rx = Receiver(r_cfg, chirp_template)
    detector = ClassicalDetector(p_cfg)

    # 3. Run Pipeline
    source_spikes = gen.generate_spike_train()       # Step 1: Bio Input
    tx_signal = tx.process(source_spikes)            # Step 2: Tx
    rx_signal = env.propagate(tx_signal, target_distance=5.0) # Step 3: Channel
    recovered = rx.process(rx_signal)                # Step 4: Rx
    
    # 4. Detect
    estimated_dist = detector.detect(source_spikes, recovered)
    
    print(f"Target: 5.0m | Estimated: {estimated_dist:.4f}m")
    
    # Optional: Quick debug plot
    plt.plot(recovered.time, recovered.data)
    plt.title("Recovered Bio-Signature")
    plt.show()

if __name__ == "__main__":
    run_simulation()

