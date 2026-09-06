# Edge Vision RL Training Harness

An A2C reinforcement-learning harness that trains a policy to choose, per video
frame, **which DNN complexity to run (light / medium / heavy)** and **where to
split the computation between client and edge server** — trading off accuracy,
latency, and energy. It runs against real frames from the TuSimple lane-detection
dataset (`TuSimpleDatasetarchive/`), falling back to synthetic frames if the
dataset or OpenCV isn't available.

Core harness: [`Edge.py`](Edge.py). A companion script,
[`baseline_comparison.py`](baseline_comparison.py), evaluates fixed baseline
policies under the same simulator for comparison against a trained policy —
see [Baseline comparison](#baseline-comparison) below.

## Setup

```bash
pip install opencv-python-headless torch numpy matplotlib
```

(`opencv-python-headless` is enough — the script only calls `imread` / `resize` /
`Canny`, no GUI windows.)

## Dataset

### Downloading

The dataset used here is the TuSimple lane-detection set, mirrored on Kaggle:
**https://www.kaggle.com/datasets/manideep1108/tusimple**

Option A — Kaggle CLI (needs a Kaggle account + API token, `kaggle.json`, see
[Kaggle's API docs](https://www.kaggle.com/docs/api) for how to generate one
and where to place it):

```bash
pip install kaggle
kaggle datasets download -d manideep1108/tusimple -p TuSimpleDatasetarchive --unzip
```

Option B — manual: open the Kaggle page above, click **Download**, then unzip
the archive so its contents land under `TuSimpleDatasetarchive/` in this
project. Either way, verify afterwards that the paths match the layout below
(the Kaggle zip's top-level folder name has changed before between dataset
revisions — if it extracts as something other than `TUSimple/`, rename/move
it so the paths line up).

### Layout

The script expects `--dataset-dir` to contain images **nested arbitrarily deep**
(it scans recursively). TuSimple's actual layout is:

```
TuSimpleDatasetarchive/TUSimple/train_set/clips/<batch>/<clip_id>/1.jpg .. 20.jpg
TuSimpleDatasetarchive/TUSimple/test_set/clips/<batch>/<clip_id>/1.jpg .. 20.jpg
```

~3,626 clips × 20 frames ≈ 72,520 frames in `train_set/clips` alone.

## Running it

```bash
python Edge.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips" --episodes 50 --no-show
```

Drop `--no-show` to also pop up the matplotlib window; either way a PNG chart is
saved. **Every run gets its own timestamped log and chart — nothing gets
overwritten** (see [Per-run logging](#per-run-logging) below).

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-dir` | `./trafficcam_1/src` | Root folder to scan recursively for `.png/.jpg/.jpeg/.bmp` frames |
| `--episodes` | `50` | Episodes per run (overridden by `--frame-sampling full-pass`) |
| `--steps-per-episode` | `30` | Frames processed per episode |
| `--frame-sampling` | `sequential` | `sequential` \| `random` \| `full-pass` — see below |
| `--seed` | `42` | Base random seed |
| `--num-seeds` | `1` | Average over N independently seeded runs |
| `--lr` | `0.003` | Adam learning rate |
| `--gamma` | `0.95` | Discount factor |
| `--update-every` | `5` | Steps between batched A2C updates |
| `--w-acc` / `--w-delay` / `--w-energy` | `0.7 / 0.15 / 0.15` | Reward weights (should sum to ~1.0) |
| `--no-show` | off | Skip `plt.show()` (headless-safe); chart is still saved |

### Frame-sampling modes

`TuSimpleDatasetarchive` is much bigger than a typical `episodes × steps_per_episode`
budget, so how frames get picked matters:

```bash
# Deterministic, cheap — but only ever touches the first episodes*steps_per_episode
# frames in sorted-path order (e.g. 50*30=1500 out of 72,520). Good for a quick smoke test.
python Edge.py --dataset-dir ".../clips" --episodes 50 --frame-sampling sequential --no-show

# Uniform random draw across the WHOLE dataset every step — representative
# training with a normal episode count. Recommended for real training runs.
python Edge.py --dataset-dir ".../clips" --episodes 50 --frame-sampling random --no-show

# Auto-expands --episodes so every frame in the dataset is visited at least
# once. Exhaustive but slow at full scale (~2,418 episodes for 72,520 frames).
python Edge.py --dataset-dir ".../clips" --frame-sampling full-pass --steps-per-episode 30 --no-show
```

For multi-seed runs, add `--num-seeds N` (results are mean/±1-std across seeds).

## Per-run logging

Every run generates a timestamp `run_id` (e.g. `20260902_105642`) shared by its
log file and its result chart, so repeated runs accumulate side by side
instead of clobbering each other:

```
logs/run_20260902_105642.log
results/dds_edge_vision_simulation_results_20260902_105642.png       # single-seed
results/dds_edge_vision_multiseed_results_20260902_105642.png        # --num-seeds > 1
```

The log file mirrors everything printed to the console during that run, and
its first lines record exactly how it was launched:

```
Command: python Edge.py --dataset-dir TuSimpleDatasetarchive/TUSimple/train_set/clips --episodes 200 --frame-sampling random --no-show
Run ID:  20260902_105642
Started: 2026-09-02T10:56:42.037301
Log file: logs\run_20260902_105642.log
```

Implementation — a `_Tee` stream duplicates every `print()` into both the
real console and the log file, so no existing call site had to change:

```python
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_run_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{run_id}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"Command: python {' '.join(sys.argv)}")
    return run_id, log_path
```

## Baseline comparison

[`baseline_comparison.py`](baseline_comparison.py) evaluates three fixed,
non-learning policies under the *identical* simulator, cost model, and
frame-sampling as `Edge.py`, so they can be compared fairly against a trained
policy rather than only against itself at different points in training:

| Policy | Behavior |
|---|---|
| Random | Uniform choice among the 17 valid `(complexity, cut_point)` actions every step |
| Cheapest | Always `(light, cut_point=0)` — maximal offloading, minimal local compute |
| Static Medium | Always `(medium, cut_point=3)` — the middle of the valid range |
| Our RL | A trained `FarsightedA2CAgent` run greedily (`epsilon=0`) |

It imports directly from `Edge.py` (`EdgeSystemSimulator`, `ImageAnalyzer`,
`FarsightedA2CAgent`, the frame loaders, `set_seed`, `safe_mean`) rather than
reimplementing the cost model, so there's exactly one source of truth for how
delay/energy/accuracy get computed. Run it from the same directory as
`Edge.py`:

```bash
python baseline_comparison.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips" --episodes 20
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-dir` | `TuSimpleDatasetarchive/TUSimple/train_set/clips` | Same dataset root you'd pass to `Edge.py` |
| `--episodes` | `20` | Evaluation episodes to average over |
| `--seed` | `42` | Random seed |
| `--frame-sampling` | `random` | `random` \| `sequential` — should match whichever `Edge.py` run you're comparing against, so both sides see a comparable slice of the dataset |

### Adding the "Our RL" row

Run standalone, the script only reports Random / Cheapest / Static Medium —
it has no access to a trained policy on its own. To include one:

- **(a)** capture the `agent` that `run_dataset_training()` builds internally
  and call `run_baseline_comparison(dataset_dir, trained_agent=agent,
  frame_sampling='random')` right after training, in the same Python
  session, or
- **(b)** reuse a converged checkpoint already reported by `Edge.py`'s own
  console/log output — once the model-choice distribution shows the policy
  has settled on a fixed action (see [Known limitation](#known-limitation)),
  its later logged episodes are a reasonable stand-in for a greedy
  evaluation, since $\epsilon$ has already decayed to its floor by then.

### Example output

Evaluated on 20 episodes of real TuSimple frames (`--frame-sampling random`,
seed 42), against a policy trained for 200 episodes on the same dataset:

| Policy | Accuracy | Latency (s) | Energy (J) | Drops/ep |
|---|---|---|---|---|
| Random | 83.6% | 1.518 | 94.16 | 3.65 |
| Cheapest | 72.0% | 0.155 | 9.79 | 0.00 |
| Static Medium | 85.0% | 1.335 | 80.68 | 6.05 |
| Our RL | 85.0% | 0.474 | 28.17 | 0.00 |

Against Static Medium — the same backbone variant Our RL converges to — the
trained policy matches accuracy exactly while cutting latency 64.5% and
energy 65.1%, and eliminating drops entirely. That gap is attributable to
*where* it cuts the model rather than *which* variant it picks, since both
policies select medium almost exclusively; see [Known
limitation](#known-limitation) below for what this comparison does and
doesn't establish.

## Key code blocks

### Recursive dataset scan

TuSimple nests frames two levels below `clips/`, so the scanner walks the tree
instead of doing a flat `os.listdir`:

```python
def get_dataset_frame_paths(directory):
    if not CV2_AVAILABLE:
        return []
    if not os.path.exists(directory):
        return []

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    frame_paths = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(supported_extensions):
                frame_paths.append(os.path.join(root, f))
    frame_paths.sort()
    return frame_paths
```

### Frame selection per training step

```python
if use_real_dataset:
    if frame_sampling == 'random':
        frame_idx = random.randrange(len(frame_paths))
    else:  # 'sequential' or 'full-pass'
        frame_idx = ((ep - 1) * steps_per_episode + t) % len(frame_paths)
    frame = load_real_frame(frame_paths[frame_idx])
```

`full-pass` doesn't change this formula — it just pre-computes `num_episodes`
so the modulo sweeps the whole dataset exactly once:

```python
if use_real_dataset and frame_sampling == 'full-pass':
    needed_episodes = math.ceil(len(frame_paths) / steps_per_episode)
    num_episodes = needed_episodes
```

### Normalized, weighted reward

Delay and energy are rescaled to ~0-1 before weighting, so they're on the same
scale as accuracy (mixing raw seconds/joules with an accuracy percentage
previously let latency dominate the reward and collapsed the policy to always
picking the cheapest model):

```python
delay_norm = min(delay / sim.timeout_threshold, 1.0)
energy_norm = min(energy / EdgeSystemSimulator.ENERGY_MAX_REF, 1.0)
reward = (w_acc * acc) - (w_delay * delay_norm) - (w_energy * energy_norm)
```

### Masked action sampling (A2C agent)

Actions are `(complexity, cut_point)` pairs; a precomputed mask rules out
invalid combinations (e.g. a "light" model can't split past cut-point 4), and
exploration samples directly from the valid-action list instead of
rejection-sampling:

```python
if random.random() < epsilon:
    idx = torch.randint(len(self.valid_action_indices), (1,)).item()
    action_idx = self.valid_action_indices[idx].item()
else:
    action_idx = torch.multinomial(masked_policy, 1).item()
```

### Batched updates every N steps

Transitions accumulate in a small rolling buffer and are flushed into one
batched A2C update every `update_every` steps (or at episode end), instead of
a single-sample update per step:

```python
def remember(self, state, action_idx, reward, next_state, done):
    self.buffer.append((state, action_idx, reward, next_state, done))
    if len(self.buffer) >= self.update_every or done:
        return self.train_on_batch()
    return None
```

## Output

- **`Edge.py`**: a console log line every 5 episodes (reward, drops, latency,
  energy, accuracy, light/medium/heavy split), plus a 6-panel chart — reward
  convergence, latency, accuracy, frame drops, model-choice distribution
  over episodes, and total energy — saved to
  `results/dds_edge_vision_simulation_results_<run_id>.png` (single-seed) or
  `results/dds_edge_vision_multiseed_results_<run_id>.png`
  (`--num-seeds > 1`), alongside a matching timestamped log in `logs/` (see
  [Per-run logging](#per-run-logging)).
- **`baseline_comparison.py`**: a printed markdown table (accuracy, latency,
  energy, drops/episode per policy) to the console — no chart or log file of
  its own.

## Known limitation

`DAGVisionModel` (a real DAG-structured CNN with partition-point-aware forward
passes) is defined in `Edge.py` but **not invoked** anywhere in
`EdgeSystemSimulator.step()`. Delay/energy/accuracy — for both `Edge.py`'s
training loop and `baseline_comparison.py`'s policy evaluations — currently
come from fixed per-complexity-class lookup tables, not from actually running
that network on frames. This applies equally to the baseline comparison
above: it establishes that the trained policy finds a better *cost-model*
operating point than the fixed baselines, not that it is more accurate on
real inference. If a paper/report describes DAG-aware partitioning as
something executed on real frames, that claim isn't backed by this code yet
— either wire `DAGVisionModel` into `EdgeSystemSimulator.step()`, or describe
this explicitly as a cost-model simulation used to pretrain the policy.
