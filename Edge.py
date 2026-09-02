"""
Edge Vision RL Training Harness (DDS dataset)

CHANGELOG (code review fixes applied):
  1. load_real_frame / get_dataset_frame_paths now check OpenCV availability
     explicitly instead of crashing with a cryptic AttributeError when cv2
     is missing but a real dataset directory is present.
  2. Bandwidth is now pre-generated once per episode into a trace array, so
     the "next state" bandwidth used to build the TD target is exactly the
     bandwidth that will actually be used next step, instead of an
     independently-redrawn random sample.
  3. FarsightedA2CAgent now batches updates every `update_every` steps from
     a small rolling buffer (default 5), matching the "update every five
     time steps from a set of sampled transitions" pattern used in the
     DNN-Scissor system this work is positioned against, instead of a
     single-sample online update every step.
  4. get_action() no longer returns an unused `value`; its forward pass is
     wrapped in torch.no_grad() since it's only used for sampling, not
     backprop. train_on_batch() recomputes values from the batch directly.
  5. all_accuracies no longer risks NaN when every frame in an episode is
     dropped (safe_mean() guards the empty-list case).
  6. Action mask is precomputed once (it never actually depended on state),
     and exploration now samples directly from the precomputed list of
     valid actions instead of rejection-sampling in a `while` loop.
  7. Gradient clipping added before each optimizer step.
  8. gamma / critic-loss weight / drop penalty are now named, configurable
     values instead of magic numbers.
  9. Added seeding per run + a multi-seed aggregation path
     (run_multi_seed_experiment), since the related work this is compared
     against (e.g. DNN-Scissor) reports results averaged over five
     independently seeded runs, not a single seed=42 run.
 10. Added argparse for dataset dir / episodes / seed(s) / learning rate /
     gamma / update_every, instead of hardcoded values.
 11. plt.show() is now optional (--no-show) so this can run headless.
 12. Renamed the `flureflection` variable to `fluctuation`.
 13. Reward function now normalizes delay/energy into comparable 0-1-ish
     ranges before weighting (delay/timeout, energy/ENERGY_MAX_REF),
     instead of mixing raw seconds/joules with accuracy percentages. The
     old `acc*10 - delay*2.5 - energy*0.4` reward let latency/energy
     dominate purely because their raw ranges were much larger than
     accuracy's -- on a Jetson Orin Nano test run, this drove accuracy
     from ~82% down to ~64.5% over 50 episodes as the policy converged to
     always picking the cheapest (light, cutpoint~4) action regardless of
     scene difficulty, even though episode reward kept climbing. New
     weights (w_acc/w_delay/w_energy, default 0.5/0.25/0.25, summing to 1)
     mirror EdgeRL's normalized weighted-sum reward and are exposed as
     CLI flags for a sensitivity sweep. DROP_PENALTY was rescaled from
     -8.0 to -1.0 to match the new reward magnitude.
 14. Added per-episode model-choice (light/medium/heavy) distribution
     tracking, printed in the training log and plotted as a stacked-area
     panel, so a collapse to always picking the cheapest model is visible
     directly rather than inferred from the accuracy curve.

STILL UNRESOLVED (flagged, not fixed here -- needs a design decision):
  DAGVisionModel (Section 3) defines a real DAG-structured CNN with
  partition-point-aware forward passes, but it is NOT called anywhere in
  EdgeSystemSimulator.step(). Delay/energy/accuracy are currently produced
  by fixed per-complexity-class lookup tables (self.model_computations,
  self.tensor_sizes, base_accuracies), not by actually running this
  network. If the paper's Methodology section describes DAG-aware
  partitioning as something that was executed on real frames, that claim
  is not yet backed by this script -- either wire DAGVisionModel into
  EdgeSystemSimulator.step(), or describe this explicitly as a cost-model
  simulation used to train the policy prior to full network integration.
"""
import argparse
import math
import os
import random
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Check if OpenCV is available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("OpenCV not found! Please run: pip install opencv-python")
    print("Falling back to synthetic frame generation only.")
    cv2 = None
    CV2_AVAILABLE = False

# ==========================================
# SECTION 0: PER-RUN LOGGING
# ==========================================
class _Tee:
    """Mirrors writes to multiple streams (e.g. the real console + a log
    file) so every existing print() call is captured in the run's log file
    without having to touch each call site."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_logging(log_dir="logs"):
    """Starts logging this run to a uniquely-named file (timestamp-based,
    so earlier runs' logs are never overwritten) and mirrors all stdout
    into it. The exact command line used to launch this run is written as
    the first line, so a log file is self-describing on its own.

    Returns (run_id, log_path); run_id is reused to name this run's result
    image(s) so a log file and its chart share a timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{run_id}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    command = "python " + " ".join(sys.argv)
    print(f"Command: {command}")
    print(f"Run ID:  {run_id}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Log file: {log_path}")
    return run_id, log_path


# ==========================================
# CONFIGURATION: PATH TO YOUR DOWNLOADED DDS DATASET
# ==========================================
# Default path; overridable via --dataset-dir on the command line.
DATASET_DIR = "./trafficcam_1/src"


def set_seed(seed):
    """(Re)seed all sources of randomness -- called once per run so that
    multi-seed experiments actually get independent runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_mean(values):
    """Mean over positive (i.e. non-dropped-frame) entries, returning 0.0
    instead of NaN when the list is empty or all entries are non-positive."""
    valid = [v for v in values if v > 0.0]
    return float(np.mean(valid)) if valid else 0.0


# ==========================================
# SECTION 1: REAL-WORLD FRAME LOADER (REPLACES SYNTHETIC GENERATOR)
# ==========================================
def load_real_frame(frame_path):
    """
    Loads a real video frame from disk and preprocesses it.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV is required to load real dataset frames but is not "
            "installed. Install it with `pip install opencv-python`, or "
            "run with an empty/missing --dataset-dir to use synthetic "
            "frames only."
        )
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    # Read the image in BGR format
    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"Failed to decode image at: {frame_path}")

    # Resize to 640x640 to match standard input size
    img = cv2.resize(img, (640, 640))
    return img


def get_dataset_frame_paths(directory):
    """
    Scans the dataset directory *recursively* and returns a sorted list of
    frame file paths. Returns an empty list (triggering the synthetic
    fallback) if OpenCV is unavailable, the directory is missing, or it
    contains no supported images.

    Recursive because datasets such as TuSimple nest frames two levels
    down (clips/<batch>/<clip_id>/1.jpg .. 20.jpg) rather than storing them
    flat in the given directory; a plain os.listdir() would silently find
    nothing and fall back to synthetic frames without erroring.
    """
    if not CV2_AVAILABLE:
        print("\n[Warning] OpenCV unavailable; cannot load real dataset frames.")
        print("Falling back to synthetic frame generator...\n")
        return []

    if not os.path.exists(directory):
        print(f"\n[Warning] Dataset directory '{directory}' not found.")
        print("Falling back to synthetic frame generator for testing...\n")
        return []

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    frame_paths = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(supported_extensions):
                frame_paths.append(os.path.join(root, f))
    frame_paths.sort()

    if len(frame_paths) == 0:
        print(f"\n[Warning] No images found under '{directory}' (searched recursively).")
        print("Falling back to synthetic frame generator...\n")

    return frame_paths


def generate_synthetic_frame(complexity_type='low'):
    """
    Fallback synthetic generator if a real dataset directory is not
    configured, or OpenCV/the dataset is unavailable.
    """
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    if complexity_type == 'low':
        if cv2:
            cv2.circle(img, (320, 320), 120, (255, 255, 255), -1)
    elif complexity_type == 'medium':
        if cv2:
            cv2.rectangle(img, (150, 150), (490, 490), (255, 255, 255), 5)
            cv2.putText(img, "Edge Vision", (200, 320), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 3)
    else:
        if cv2:
            for _ in range(80):
                pt1 = (random.randint(0, 639), random.randint(0, 639))
                pt2 = (random.randint(0, 639), random.randint(0, 639))
                color = (random.randint(100, 255), random.randint(100, 255),
                         random.randint(100, 255))
                cv2.line(img, pt1, pt2, color, random.randint(1, 3))
        else:
            img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    return img


# ==========================================
# SECTION 2: LIGHTWEIGHT IMAGE ANALYZER
# ==========================================
class ImageAnalyzer:
    """
    Extracts image complexity features in milliseconds (Shannon Entropy & Canny Edge).
    """
    @staticmethod
    def compute_entropy(gray_img):
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]) if cv2 \
            else np.histogram(gray_img, bins=256, range=(0, 256))[0]
        hist = hist.ravel() / (hist.sum() + 1e-10)
        prob = hist[hist > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    @staticmethod
    def compute_edge_density(gray_img):
        if cv2:
            edges = cv2.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            grad_x = np.diff(gray_img, axis=1)
            edge_density = np.mean(np.abs(grad_x) > 40)
        return edge_density

    def analyze(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY) if cv2 \
            else bgr_img.mean(axis=2).astype(np.uint8)
        entropy = self.compute_entropy(gray)
        edge_density = self.compute_edge_density(gray)
        norm_entropy = min(entropy / 8.0, 1.0)
        return norm_entropy, edge_density


# ==========================================
# SECTION 3: TARGET DNN WITH DAG STRUCTURE
# ==========================================
class DAGVisionModel(nn.Module):
    """
    NOTE: as of this revision, this network is defined but NOT invoked by
    EdgeSystemSimulator.step() -- see the module-level docstring. It is
    kept here so that wiring it in later (real forward passes instead of
    the lookup-table cost model) is a contained change.
    """
    def __init__(self, complexity='medium'):
        super(DAGVisionModel, self).__init__()
        self.complexity = complexity
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        channels = 16 if complexity == 'light' else (32 if complexity == 'medium' else 64)
        self.conv3 = nn.Conv2d(16, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(16, channels, kernel_size=1)

        self.conv6 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x, partition_point=None, intermediate_tensor=None, side='both'):
        if side == 'remote':
            assert intermediate_tensor is not None
            out = intermediate_tensor
            if partition_point <= 1:
                out = F.relu(self.conv2(out))
            if partition_point <= 2:
                skip = self.skip_conv(out)
                out = F.relu(self.conv3(out))
            else:
                skip = None
            if partition_point <= 3:
                out = F.relu(self.conv4(out))
            if partition_point <= 4:
                out = F.relu(self.conv5(out))
                if skip is not None:
                    out = out + skip
            if partition_point <= 5:
                out = F.relu(self.conv6(out))
            if partition_point <= 6:
                out = self.pool(out)
                out = torch.flatten(out, 1)
                out = self.fc(out)
            return out

        out = x
        if partition_point == 0:
            return out
        out = F.relu(self.conv1(out))
        if partition_point == 1:
            return out
        out = F.relu(self.conv2(out))
        if partition_point == 2:
            return out
        skip = self.skip_conv(out)
        out = F.relu(self.conv3(out))
        if partition_point == 3:
            return out
        out = F.relu(self.conv4(out))
        if partition_point == 4:
            return out
        out = F.relu(self.conv5(out))
        out = out + skip
        if partition_point == 5:
            return out
        out = F.relu(self.conv6(out))
        if partition_point == 6:
            return out
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ==========================================
# SECTION 4: EDGE-ASSISTED SYSTEM SIMULATOR
# ==========================================
class EdgeSystemSimulator:
    # Named constants. DROP_PENALTY and ENERGY_MAX_REF are calibrated against
    # the realistic min/max delay and energy achievable across all 17 valid
    # actions (computed empirically -- see conversation notes / Methodology
    # Section E): delay ranges ~0.05-2.5s, energy ranges ~0.1-6.5J. Both
    # accuracy and the drop penalty are now on the same normalized scale as
    # delay/energy, which the raw-unit version was not (see reward-collapse
    # discussion: the old version let a plain latency/energy grab dominate
    # accuracy because their raw ranges weren't comparable).
    DROP_PENALTY = -1.0
    JOULES_TO_WH = 1.0 / 3600.0  # 1 J = 1/3600 Wh
    ENERGY_MAX_REF = 6.5  # J per frame; worst realistic case across valid actions

    def __init__(self):
        self.client_capacity = 3.5 * (10 ** 9)  # Client Compute Capacity (FLOP/s)
        self.edge_capacity = 40.0 * (10 ** 9)   # Remote GPU Server Capacity (FLOP/s)

        self.p_idle = 1.77
        self.p_compute = 4.48
        self.p_transmit = 2.50

        self.local_queue_time = 0.0
        self.offload_queue_time = 0.0

        self.timeout_threshold = 1.5
        self.initial_battery = 5.0
        self.remaining_battery = 5.0

        self.model_computations = {
            'light':  [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
            'medium': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            'heavy':  [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
        }

        self.tensor_sizes = {
            'light':  [0.5, 1.2, 1.2, 1.0, 0.8, 0.6, 0.1],
            'medium': [1.0, 2.4, 2.4, 2.0, 1.6, 1.2, 0.1],
            'heavy':  [2.0, 4.8, 4.8, 4.0, 3.2, 2.4, 0.1]
        }

    def reset(self):
        """Reset per-episode state. Battery/queues previously had to be
        reset by hand in the training loop; centralizing it here avoids
        that being done inconsistently."""
        self.remaining_battery = self.initial_battery
        self.local_queue_time = 0.0
        self.offload_queue_time = 0.0

    def simulate_bandwidth(self, time_step):
        base_rate = 8.0
        fluctuation = 4.0 * np.sin(time_step / 4.0) + np.random.normal(0, 0.8)
        bandwidth = max(1.0, base_rate + fluctuation)
        return bandwidth

    def generate_bandwidth_trace(self, num_steps):
        """Pre-generate one bandwidth sample per timestep (0..num_steps
        inclusive) up front. This ensures the 'next state' bandwidth used
        to build a TD target is exactly the bandwidth that will actually be
        used once that timestep becomes 'current', rather than an
        independently redrawn random sample (simulate_bandwidth() is
        stochastic, so calling it twice for the same t previously gave two
        different values)."""
        return [self.simulate_bandwidth(t) for t in range(num_steps + 1)]

    def step(self, action, frame_idx, complexity_type, bandwidth, frame_interval=0.15):
        complexity_name = ['light', 'medium', 'heavy'][action[0]]
        split_point = action[1]

        # Base Accuracy Mapping
        base_accuracies = {'light': 0.72, 'medium': 0.85, 'heavy': 0.95}
        acc = base_accuracies[complexity_name]
        if complexity_type == 'high' and complexity_name == 'light':
            acc -= 0.15
        elif complexity_type == 'low' and complexity_name == 'heavy':
            acc = 0.96

        # Local compute delay
        local_flops = sum(self.model_computations[complexity_name][:split_point]) * (10 ** 9)
        local_compute_delay = local_flops / self.client_capacity

        # Drain queues over frame interval
        self.local_queue_time = max(0.0, self.local_queue_time - frame_interval)
        self.offload_queue_time = max(0.0, self.offload_queue_time - frame_interval)

        # Queue check
        total_local_waiting = self.local_queue_time + local_compute_delay
        if total_local_waiting > self.timeout_threshold:
            total_delay = self.timeout_threshold
            energy_consumed = self.p_idle * self.timeout_threshold
            return total_delay, energy_consumed, 0.0, True

        self.local_queue_time = total_local_waiting

        # Offload delay
        tensor_size_mb = self.tensor_sizes[complexity_name][split_point]
        transmission_delay = tensor_size_mb / bandwidth

        total_offload_waiting = self.offload_queue_time + transmission_delay
        if total_offload_waiting > self.timeout_threshold:
            total_delay = self.timeout_threshold
            energy_consumed = self.p_idle * self.timeout_threshold
            return total_delay, energy_consumed, 0.0, True

        self.offload_queue_time = total_offload_waiting

        # Remote compute delay
        remote_flops = sum(self.model_computations[complexity_name][split_point:]) * (10 ** 9)
        remote_compute_delay = remote_flops / self.edge_capacity

        total_delay = (local_compute_delay + transmission_delay + remote_compute_delay
                       + self.local_queue_time + self.offload_queue_time)

        # Energy consumption model
        e_compute = self.p_compute * local_compute_delay
        e_transmit = self.p_transmit * transmission_delay
        e_idle = self.p_idle * (total_delay - local_compute_delay - transmission_delay)
        energy_consumed = e_compute + e_transmit + e_idle

        self.remaining_battery -= energy_consumed * self.JOULES_TO_WH
        self.remaining_battery = max(0.0, self.remaining_battery)

        return total_delay, energy_consumed, acc, False


# ==========================================
# SECTION 5: DEEP REINFORCEMENT LEARNING AGENT (A2C)
# ==========================================
class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU()),
            nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class FarsightedA2CAgent:
    def __init__(self, state_dim=7, num_complexities=3, num_cutpoints=7,
                 gamma=0.95, critic_loss_weight=0.5, lr=0.003,
                 update_every=5, grad_clip_norm=5.0):
        self.num_complexities = num_complexities
        self.num_cutpoints = num_cutpoints
        self.action_dim = num_complexities * num_cutpoints
        self.gamma = gamma
        self.critic_loss_weight = critic_loss_weight
        self.update_every = update_every
        self.grad_clip_norm = grad_clip_norm

        self.network = A2CNetwork(state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # The action mask (Liu et al., 2026) depends only on the
        # (complexity, cut-point) index, never on state -- so it is built
        # once here instead of being rebuilt from scratch on every call to
        # get_action().
        self.mask = torch.ones(self.action_dim)
        for c in range(self.num_complexities):
            for p in range(self.num_cutpoints):
                action_idx = c * self.num_cutpoints + p
                if c == 0 and p > 4:
                    self.mask[action_idx] = 0.0
                if c == 2 and p < 2:
                    self.mask[action_idx] = 0.0
        # Precomputed so exploration can sample directly from valid actions
        # instead of rejection-sampling in a `while` loop.
        self.valid_action_indices = torch.nonzero(self.mask, as_tuple=False).squeeze(-1)

        # Small rolling buffer of (state, action_idx, reward, next_state,
        # done) transitions, flushed into a batched update every
        # `update_every` steps (or at episode end), mirroring the "update
        # every five time steps from a set of sampled transitions" pattern
        # used by DNN-Scissor rather than a single-sample update per step.
        self.buffer = []

    def get_action(self, state_vector, epsilon=0.1):
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        # No gradient needed here -- this pass is only used for sampling an
        # action; train_on_batch() recomputes policy/value on-graph later.
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
            masked_policy = policy.squeeze(0) * self.mask
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                # Degenerate numerical case: fall back to a uniform
                # distribution over valid actions only, never an unmasked
                # (potentially invalid) action.
                masked_policy = self.mask / self.mask.sum()

            if random.random() < epsilon:
                idx = torch.randint(len(self.valid_action_indices), (1,)).item()
                action_idx = self.valid_action_indices[idx].item()
            else:
                action_idx = torch.multinomial(masked_policy, 1).item()

        complexity_choice = action_idx // self.num_cutpoints
        split_point_choice = action_idx % self.num_cutpoints
        return (complexity_choice, split_point_choice), action_idx

    def remember(self, state, action_idx, reward, next_state, done):
        """Add a transition to the buffer; triggers a batched update every
        `update_every` steps, or immediately at episode end so no
        transitions are silently dropped. Returns the training loss if an
        update ran, else None."""
        self.buffer.append((state, action_idx, reward, next_state, done))
        if len(self.buffer) >= self.update_every or done:
            return self.train_on_batch()
        return None

    def train_on_batch(self):
        if not self.buffer:
            return None

        states = torch.tensor([t[0] for t in self.buffer], dtype=torch.float32)
        action_idxs = torch.tensor([t[1] for t in self.buffer], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in self.buffer], dtype=torch.float32)
        next_states = torch.tensor([t[3] for t in self.buffer], dtype=torch.float32)
        dones = torch.tensor([t[4] for t in self.buffer], dtype=torch.float32)

        self.optimizer.zero_grad()

        policies, values = self.network(states)
        with torch.no_grad():
            _, next_values = self.network(next_states)

        td_targets = rewards + self.gamma * next_values.squeeze(-1) * (1.0 - dones)
        advantages = td_targets - values.squeeze(-1)

        log_probs = torch.log(
            policies.gather(1, action_idxs.unsqueeze(1)).squeeze(1) + 1e-10
        )
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(-1), td_targets.detach())

        total_loss = actor_loss + self.critic_loss_weight * critic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        loss_value = total_loss.item()
        self.buffer = []
        return loss_value


# ==========================================
# SECTION 6: TRAINING HARNESS ON DDS DATASET
# ==========================================
def run_dataset_training(num_episodes=50, seed=42, dataset_dir=DATASET_DIR,
                          steps_per_episode=30, lr=0.003, gamma=0.95,
                          update_every=5, w_acc=0.7, w_delay=0.15, w_energy=0.15,
                          frame_sampling='sequential', verbose=True):
    """
    w_acc / w_delay / w_energy weight the three (now comparably-scaled,
    0-1-ish) reward components. Defaults sum to 1.0, mirroring the
    normalized weighted-sum reward used by EdgeRL (Mounesan et al., 2024)
    rather than combining raw seconds/joules/percentages directly. These
    specific default weights (0.7/0.15/0.15) were chosen empirically: on
    synthetic frames, w_acc=0.5 still let the policy converge to ~100%
    "light" model usage and ~72% accuracy by episode 35; w_acc=0.7 instead
    converges (consistently across 3 seeds) to 85% accuracy with zero
    drops and moderate latency/energy. Pass different weights to run your
    own sensitivity sweep, as EdgeRL and DNN-Scissor do.

    frame_sampling controls which real-dataset frames get visited (no
    effect in synthetic-fallback mode, where frames are generated fresh
    every step):
      'sequential' -- deterministic: frame_idx = ((ep-1)*steps_per_episode
          + t) % len(frame_paths). With a real dataset much larger than
          num_episodes*steps_per_episode (e.g. TuSimple's ~72k frames),
          this only ever visits the first num_episodes*steps_per_episode
          frames in sorted-path order and never wraps -- fine for a quick
          run, but it is not a representative sample of the whole dataset.
      'random' -- each step draws a uniformly random frame (with
          replacement) from the full frame_paths list, so even a modest
          episode count sees a mixed sample across all clips/batches
          instead of one alphabetically-first slice.
      'full-pass' -- like 'sequential', but num_episodes is overridden to
          ceil(len(frame_paths) / steps_per_episode) so every frame in the
          dataset gets visited at least once across the run (the final
          episode may re-visit a few frames from the start to fill out its
          last batch). Much longer run time for large datasets.
    """
    set_seed(seed)

    if frame_sampling not in ('sequential', 'random', 'full-pass'):
        raise ValueError(
            f"frame_sampling must be 'sequential', 'random', or 'full-pass', got {frame_sampling!r}"
        )

    if verbose:
        print("=============================================================")
        print("DNN-Scissor Farsighted A2C Local Dataset Execution Loop")
        print(f"(seed={seed}, w_acc={w_acc}, w_delay={w_delay}, w_energy={w_energy}, "
              f"frame_sampling={frame_sampling})")
        print("=============================================================")

    analyzer = ImageAnalyzer()
    sim = EdgeSystemSimulator()
    agent = FarsightedA2CAgent(lr=lr, gamma=gamma, update_every=update_every)

    # Scan for real DDS frames
    frame_paths = get_dataset_frame_paths(dataset_dir)
    use_real_dataset = len(frame_paths) > 0

    if use_real_dataset and frame_sampling == 'full-pass':
        needed_episodes = math.ceil(len(frame_paths) / steps_per_episode)
        if needed_episodes != num_episodes and verbose:
            print(f"[full-pass] Overriding num_episodes {num_episodes} -> {needed_episodes} "
                  f"so all {len(frame_paths)} frames are visited at least once "
                  f"({steps_per_episode} frames/episode).")
        num_episodes = needed_episodes
    elif not use_real_dataset and frame_sampling == 'full-pass' and verbose:
        print("[Warning] frame_sampling='full-pass' has no effect without a real dataset "
              "(synthetic frames are generated fresh every step); using num_episodes as given.")

    if verbose:
        if use_real_dataset:
            print(f"Successfully loaded real dataset. Found {len(frame_paths)} frames.")
            print(f"Simulation will process batches of {min(steps_per_episode, len(frame_paths))} "
                  f"frames per episode.")
        else:
            print("Dataset directory is empty, not found, or OpenCV is unavailable. "
                  "Operating on synthetic frame generator mode.")

    all_rewards, all_latencies, all_energy, all_drops, all_accuracies = [], [], [], [], []
    all_complexity_fracs = []  # [{'light':frac,'medium':frac,'heavy':frac}, ...] per episode

    for ep in range(1, num_episodes + 1):
        ep_reward = 0
        ep_latencies, ep_energy, ep_accuracies = [], [], []
        ep_drops = 0
        ep_complexity_counts = {'light': 0, 'medium': 0, 'heavy': 0}
        ep_completed_steps = 0

        sim.reset()

        # Pre-generate this episode's bandwidth trace once, so the
        # "next state" bandwidth used for the TD target matches exactly
        # what will be used when that timestep becomes current.
        bandwidth_trace = sim.generate_bandwidth_trace(steps_per_episode)

        for t in range(steps_per_episode):
            # Load real or synthetic frame
            if use_real_dataset:
                if frame_sampling == 'random':
                    frame_idx = random.randrange(len(frame_paths))
                else:  # 'sequential' or 'full-pass'
                    frame_idx = ((ep - 1) * steps_per_episode + t) % len(frame_paths)
                try:
                    frame = load_real_frame(frame_paths[frame_idx])
                except Exception as e:
                    print(f"[Error] Failed to load frame {frame_paths[frame_idx]}: {e}. "
                          f"Falling back to synthetic.")
                    frame = generate_synthetic_frame('medium')
            else:
                complexity_type = random.choice(['low', 'medium', 'high'])
                frame = generate_synthetic_frame(complexity_type)

            # Extract real image metrics frame-by-frame
            norm_entropy, edge_density = analyzer.analyze(frame)

            # Map edge density to complexity class
            if edge_density < 0.04:
                complexity_type = 'low'
            elif edge_density < 0.12:
                complexity_type = 'medium'
            else:
                complexity_type = 'high'

            bandwidth = bandwidth_trace[t]

            state = [
                sim.local_queue_time,
                sim.offload_queue_time,
                bandwidth,
                sim.local_queue_time + sim.offload_queue_time,
                norm_entropy,
                edge_density,
                sim.remaining_battery
            ]

            epsilon = max(0.01, 1.0 - ep / 35.0)
            action, action_idx = agent.get_action(state, epsilon=epsilon)
            ep_complexity_counts[['light', 'medium', 'heavy'][action[0]]] += 1

            delay, energy, acc, dropped = sim.step(action, t, complexity_type, bandwidth)

            if dropped:
                reward = EdgeSystemSimulator.DROP_PENALTY
                ep_drops += 1
            else:
                # Normalized reward: delay and energy are rescaled into
                # comparable 0-1-ish ranges before weighting, instead of
                # mixing raw seconds/joules/percentages directly (the old
                # `acc*10 - delay*2.5 - energy*0.4` let latency/energy
                # dominate purely because their raw ranges were larger than
                # accuracy's, causing the policy to always pick the
                # cheapest action regardless of scene difficulty -- see
                # conversation notes / Methodology Section E). Mirrors the
                # normalized weighted-sum reward used by EdgeRL.
                delay_norm = min(delay / sim.timeout_threshold, 1.0)
                energy_norm = min(energy / EdgeSystemSimulator.ENERGY_MAX_REF, 1.0)
                reward = (w_acc * acc) - (w_delay * delay_norm) - (w_energy * energy_norm)
                ep_completed_steps += 1

            ep_reward += reward
            ep_latencies.append(delay if not dropped else sim.timeout_threshold)
            ep_energy.append(energy)
            ep_accuracies.append(acc if not dropped else 0.0)

            next_bandwidth = bandwidth_trace[t + 1]
            next_state = [
                sim.local_queue_time,
                sim.offload_queue_time,
                next_bandwidth,
                sim.local_queue_time + sim.offload_queue_time,
                norm_entropy,
                edge_density,
                sim.remaining_battery
            ]

            done = (t == steps_per_episode - 1) or (sim.remaining_battery <= 0.0)
            agent.remember(state, action_idx, reward, next_state, done)

            if sim.remaining_battery <= 0.0:
                break

        all_rewards.append(ep_reward)
        all_latencies.append(np.mean(ep_latencies))
        all_energy.append(np.sum(ep_energy))
        all_accuracies.append(safe_mean(ep_accuracies))
        all_drops.append(ep_drops)

        total_choices = sum(ep_complexity_counts.values())
        complexity_fracs = {
            k: (v / total_choices if total_choices > 0 else 0.0)
            for k, v in ep_complexity_counts.items()
        }
        all_complexity_fracs.append(complexity_fracs)

        if verbose and (ep % 5 == 0 or ep == 1):
            acc_percentage = safe_mean(ep_accuracies) * 100
            print(f"Episode {ep:02d}/{num_episodes} | Reward: {ep_reward:7.2f} | "
                  f"Drops: {ep_drops:2d} | E2E Latency: {np.mean(ep_latencies):5.3f}s | "
                  f"Energy: {np.sum(ep_energy):5.2f}J | Acc: {acc_percentage:4.1f}% | "
                  f"Model choice L/M/H: {complexity_fracs['light']*100:4.1f}%/"
                  f"{complexity_fracs['medium']*100:4.1f}%/{complexity_fracs['heavy']*100:4.1f}%")

    if verbose:
        print("\nTraining loop complete.")
    return all_rewards, all_latencies, all_energy, all_drops, all_accuracies, all_complexity_fracs


def run_multi_seed_experiment(num_seeds=5, num_episodes=50, dataset_dir=DATASET_DIR,
                               base_seed=42, **kwargs):
    """
    Repeats training across `num_seeds` independent random seeds and
    aggregates the results (mean/std per episode). Mirrors the "average of
    five independent training runs, each initialized with a different
    random seed" practice reported by DNN-Scissor and similar cited work,
    rather than reporting a single seed=42 run.
    """
    all_runs = []
    for i in range(num_seeds):
        seed = base_seed + i
        print(f"\n===== Run {i + 1}/{num_seeds} (seed={seed}) =====")
        results = run_dataset_training(num_episodes=num_episodes, seed=seed,
                                        dataset_dir=dataset_dir, **kwargs)
        all_runs.append(results)

    rewards = np.array([r[0] for r in all_runs])
    latencies = np.array([r[1] for r in all_runs])
    energy = np.array([r[2] for r in all_runs])
    drops = np.array([r[3] for r in all_runs])
    accuracies = np.array([r[4] for r in all_runs])
    # Track the "light" fraction specifically -- it's the direct signal for
    # whether the policy has collapsed to always picking the cheapest model
    # regardless of scene difficulty (see conversation notes).
    light_frac = np.array([[ep['light'] for ep in r[5]] for r in all_runs])

    return {
        'rewards_mean': rewards.mean(axis=0), 'rewards_std': rewards.std(axis=0),
        'latencies_mean': latencies.mean(axis=0), 'latencies_std': latencies.std(axis=0),
        'energy_mean': energy.mean(axis=0), 'energy_std': energy.std(axis=0),
        'drops_mean': drops.mean(axis=0), 'drops_std': drops.std(axis=0),
        'accuracies_mean': accuracies.mean(axis=0), 'accuracies_std': accuracies.std(axis=0),
        'light_frac_mean': light_frac.mean(axis=0), 'light_frac_std': light_frac.std(axis=0),
        'num_seeds': num_seeds,
    }


# ==========================================
# SECTION 7: PLOTTING CONVERGENCE CHARTS
# ==========================================
def plot_results(all_rewards, all_latencies, all_energy, all_drops, all_accuracies,
                  all_complexity_fracs, show=True,
                  out_path="dds_edge_vision_simulation_results.png"):
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))

    axs[0, 0].plot(all_rewards, color='tab:blue', linewidth=2)
    axs[0, 0].set_title("A2C Policy Reward Convergence", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Episode Reward")
    axs[0, 0].grid(True, linestyle='--')

    axs[0, 1].plot(all_latencies, color='tab:orange', linewidth=2)
    axs[0, 1].set_title("Average E2E Latency", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Inference Latency (seconds)")
    axs[0, 1].grid(True, linestyle='--')

    axs[0, 2].plot([a * 100 for a in all_accuracies], color='tab:green', linewidth=2)
    axs[0, 2].set_title("Vision Inference Accuracy", fontsize=12, fontweight='bold')
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Average Accuracy (%)")
    axs[0, 2].grid(True, linestyle='--')

    axs[1, 0].bar(range(len(all_drops)), all_drops, color='tab:red', alpha=0.7)
    axs[1, 0].set_title("Local OS Frame Drops (Queue Timeouts)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Dropped Frames Count")
    axs[1, 0].grid(True, linestyle='--')

    # Diagnostic panel: fraction of light/medium/heavy chosen per episode.
    # A policy that has collapsed to "always pick the cheapest model" will
    # show light -> ~100% here, which is the direct signal for the
    # accuracy-collapse failure mode discussed in the review.
    episodes = range(len(all_complexity_fracs))
    light = [f['light'] * 100 for f in all_complexity_fracs]
    medium = [f['medium'] * 100 for f in all_complexity_fracs]
    heavy = [f['heavy'] * 100 for f in all_complexity_fracs]
    axs[1, 1].stackplot(episodes, light, medium, heavy,
                         labels=['light', 'medium', 'heavy'],
                         colors=['tab:cyan', 'tab:purple', 'tab:brown'], alpha=0.8)
    axs[1, 1].set_title("Model-Choice Distribution", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Share of Frames (%)")
    axs[1, 1].legend(loc='upper right', fontsize=8)
    axs[1, 1].set_ylim(0, 100)

    axs[1, 2].plot(all_energy, color='tab:pink', linewidth=2)
    axs[1, 2].set_title("Total Episode Energy", fontsize=12, fontweight='bold')
    axs[1, 2].set_xlabel("Episode")
    axs[1, 2].set_ylabel("Energy (J)")
    axs[1, 2].grid(True, linestyle='--')

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Successfully saved performance chart to '{out_path}'.")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_aggregated_results(agg, show=True,
                             out_path="dds_edge_vision_multiseed_results.png"):
    """Same panels as plot_results, but with a shaded +/- 1 std band across
    seeds instead of a single-seed line."""
    episodes = np.arange(len(agg['rewards_mean']))
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))

    def band(ax, mean, std, color, ylabel, title):
        ax.plot(episodes, mean, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--')

    band(axs[0, 0], agg['rewards_mean'], agg['rewards_std'], 'tab:blue',
         "Total Episode Reward", f"A2C Reward Convergence (n={agg['num_seeds']} seeds)")
    band(axs[0, 1], agg['latencies_mean'], agg['latencies_std'], 'tab:orange',
         "Inference Latency (s)", "Average E2E Latency")
    band(axs[0, 2], agg['accuracies_mean'] * 100, agg['accuracies_std'] * 100, 'tab:green',
         "Average Accuracy (%)", "Vision Inference Accuracy")
    band(axs[1, 0], agg['drops_mean'], agg['drops_std'], 'tab:red',
         "Dropped Frames Count", "Local OS Frame Drops (Queue Timeouts)")
    band(axs[1, 1], agg['light_frac_mean'] * 100, agg['light_frac_std'] * 100, 'tab:cyan',
         "Light-Model Share (%)", "Light-Model Selection Frequency")
    band(axs[1, 2], agg['energy_mean'], agg['energy_std'], 'tab:pink',
         "Energy (J)", "Total Episode Energy")

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Successfully saved multi-seed performance chart to '{out_path}'.")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Edge Vision RL Training Harness (DDS dataset)")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
                         help="Path to DDS frame directory")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per run")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-seeds", type=int, default=1,
                         help="Number of independently seeded runs to average over "
                              "(use >1 to match multi-seed reporting practice)")
    parser.add_argument("--steps-per-episode", type=int, default=30,
                         help="Frames processed per episode")
    parser.add_argument("--frame-sampling", type=str, default="sequential",
                         choices=["sequential", "random", "full-pass"],
                         help="How to pick real-dataset frames per step: 'sequential' "
                              "(deterministic, only visits the first "
                              "episodes*steps_per_episode frames), 'random' (uniform "
                              "sample across the whole dataset every step), or "
                              "'full-pass' (auto-sets episodes so every frame in the "
                              "dataset is visited at least once). No effect in "
                              "synthetic-fallback mode.")
    parser.add_argument("--lr", type=float, default=0.003, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--update-every", type=int, default=5,
                         help="Steps between batched A2C updates")
    parser.add_argument("--w-acc", type=float, default=0.7,
                         help="Reward weight on (normalized) accuracy")
    parser.add_argument("--w-delay", type=float, default=0.15,
                         help="Reward weight on normalized delay")
    parser.add_argument("--w-energy", type=float, default=0.15,
                         help="Reward weight on normalized energy")
    parser.add_argument("--no-show", action="store_true",
                         help="Do not call plt.show() (headless-safe)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Every run gets a unique timestamp-based ID: it names this run's log
    # file (logs/run_<id>.log, mirroring all console output and starting
    # with the exact command line used) and its result chart, so repeated
    # runs accumulate side by side instead of overwriting each other.
    run_id, log_path = setup_run_logging()

    if args.num_seeds > 1:
        agg = run_multi_seed_experiment(
            num_seeds=args.num_seeds, num_episodes=args.episodes,
            dataset_dir=args.dataset_dir, base_seed=args.seed,
            steps_per_episode=args.steps_per_episode, frame_sampling=args.frame_sampling,
            lr=args.lr, gamma=args.gamma, update_every=args.update_every,
            w_acc=args.w_acc, w_delay=args.w_delay, w_energy=args.w_energy,
        )
        plot_aggregated_results(
            agg, show=not args.no_show,
            out_path=os.path.join("results", f"dds_edge_vision_multiseed_results_{run_id}.png"),
        )
    else:
        results = run_dataset_training(
            num_episodes=args.episodes, seed=args.seed, dataset_dir=args.dataset_dir,
            steps_per_episode=args.steps_per_episode, frame_sampling=args.frame_sampling,
            lr=args.lr, gamma=args.gamma, update_every=args.update_every,
            w_acc=args.w_acc, w_delay=args.w_delay, w_energy=args.w_energy,
        )
        plot_results(
            *results, show=not args.no_show,
            out_path=os.path.join("results", f"dds_edge_vision_simulation_results_{run_id}.png"),
        )

    print(f"\nRun {run_id} complete. Log: {log_path}")
