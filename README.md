# ProjectEdge

Code for our paper on choosing a DNN variant and a split point per video frame with reinforcement learning. An A2C agent picks one of three backbone variants (light, medium, heavy) and one of seven cut points, which decides how much of the network runs on the device and how much on the edge server. The reward trades off accuracy, delay and energy.

The experiments run in a simulator. Delay, energy and accuracy come from lookup tables (FLOP counts and tensor sizes per variant and cut point), and accuracy is assigned per variant and scene class instead of being measured by a detector. None of this has been validated against real hardware, so the numbers describe the simulator and nothing more. Frames come from the TuSimple lane-detection dataset, or from synthetic frames if the dataset or OpenCV is missing.

## What we found

The policy settles on a single action, the medium variant with cut point 0, and does not change it with the scene. Evaluating all 17 valid fixed actions shows this is also the best fixed action, so the agent finds a good static configuration but does not beat one. About 99.8% of TuSimple frames fall in the low or medium complexity class, which leaves the complexity features with almost no reward-relevant information. Details are in `reviewer1_experiments.py` and the paper.

## Files

| File | What it does |
|---|---|
| `Edge.py` | Simulator, A2C agent, training loop, logging and plots |
| `baseline_comparison.py` | Random, Cheapest and Static Medium policies under the same simulator |
| `reviewer1_experiments.py` | All 17 fixed actions, scene class mix, and the frozen policy over several seeds |
| `Edge_Vision_RL.ipynb` | Colab notebook version |
| `logs/`, `results/` | One log and one chart per run |

## Setup

```bash
pip install opencv-python-headless torch numpy matplotlib
```

## Dataset

We use TuSimple from Kaggle: https://www.kaggle.com/datasets/manideep1108/tusimple

Unzip it so the frames end up under `TuSimpleDatasetarchive/`. The scripts scan the folder recursively, so the exact nesting does not matter, but the Kaggle archive has changed its top-level folder name between versions, so check the path after unzipping. We train on `train_set/clips` (3,626 clips, about 72,520 frames).

```
TuSimpleDatasetarchive/TUSimple/train_set/clips/<batch>/<clip_id>/1.jpg .. 20.jpg
```

## Training

```bash
python Edge.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips" --episodes 200 --frame-sampling random --no-show
```

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-dir` | `./trafficcam_1/src` | Folder scanned for `.png`, `.jpg`, `.jpeg`, `.bmp` frames |
| `--episodes` | 50 | Episodes per run (ignored with `full-pass`) |
| `--steps-per-episode` | 30 | Frames per episode |
| `--frame-sampling` | `sequential` | `sequential`, `random` or `full-pass` |
| `--seed` | 42 | Base seed |
| `--num-seeds` | 1 | Average over this many seeded runs |
| `--lr` | 0.003 | Adam learning rate |
| `--gamma` | 0.95 | Discount factor |
| `--update-every` | 5 | Steps between A2C updates |
| `--w-acc`, `--w-delay`, `--w-energy` | 0.7, 0.15, 0.15 | Reward weights, roughly summing to 1 |
| `--no-show` | off | Do not open the plot window; the chart is still saved |

Frame sampling matters because the dataset is much larger than `episodes * steps_per_episode`. `sequential` only ever visits the first frames in sorted order, which is fine for a quick check. `random` draws uniformly from the whole dataset and is what we used for the paper. `full-pass` raises the episode count so every frame is seen once, which is slow at full scale.

Each run writes `logs/run_<timestamp>.log` and a chart under `results/`, so earlier runs are never overwritten. The first lines of the log record the command used.

## Baselines

```bash
python baseline_comparison.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips" --episodes 20
```

This evaluates Random (uniform over the 17 valid actions), Cheapest (light, cut 0) and Static Medium (medium, cut 3) with the same simulator and frame sampling as training. It prints a table and writes no files. To add a trained policy, capture the agent built inside `run_dataset_training()` and pass it to `run_baseline_comparison(..., trained_agent=agent)` in the same session. Note that this policy samples from its masked action distribution with epsilon set to 0. It is not an argmax.

## Extra experiments

```bash
python reviewer1_experiments.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips"
```

This reports the scene class mix of random frames, evaluates every valid fixed action over 20 episodes, and trains several seeds before evaluating each frozen policy on the same 20 episodes. Results are saved to `reviewer1_results.json`. Useful flags are `--seeds`, `--episodes`, `--eval-episodes`, `--skip-sweep` and `--ablate-epsilon` (trains without epsilon-greedy exploration).

Results on the simulator, 20 evaluation episodes, seed 42:

| Policy | Accuracy | Latency (s) | Energy (J) | Drops/ep |
|---|---|---|---|---|
| Random | 83.6% | 1.518 | 94.16 | 3.65 |
| Cheapest | 72.0% | 0.155 | 9.79 | 0.00 |
| Static Medium | 85.0% | 1.335 | 80.68 | 6.05 |
| Best fixed (medium, cut 0) | 85.0% | 0.527 | 31.05 | 0.00 |
| Trained policy, frozen | 85.0% | 0.527 | 31.05 | 0.00 |

The trained policy matches the best fixed action in all five seeds we tried (42 to 46). The gap to Static Medium comes from the cut point, not from the variant, because both use the medium model.

## Limitations

- The cost model is a lookup table and is not validated on hardware.
- Accuracy is assigned from the variant and the scene class, not measured.
- The policy does not adapt to the input. It ends up on one action.
- Epsilon-greedy exploration in an on-policy A2C update biases the gradient. We did not correct for it.
- Training curves come from one seed.
