"""
Extra experiments: scene class mix, all fixed actions, frozen policy over seeds. Place this file next to
Edge.py and baseline_comparison.py and run, for example:

    python reviewer1_experiments.py --dataset-dir "TuSimpleDatasetarchive/TUSimple/train_set/clips"

It does NOT modify Edge.py. It runs, on the real TuSimple frames and with the
same simulator, bandwidth traces and evaluation harness as baseline_comparison.py:

  1. Scene-class distribution of random TuSimple frames (low / medium / high edge density).
  2. All 17 valid fixed actions, 20 evaluation episodes each (best-of-17 baseline).
  3. Random / Cheapest / Static Medium re-run (to check the Random energy value).
  4. For each of N seeds: train 200 episodes, then evaluate the FROZEN policy for 20
     episodes (sampled and greedy) under the identical harness, and report mean +/- std.
  5. Optional (--ablate-epsilon): the same, but trained without epsilon-greedy exploration.

Results are printed as markdown tables and saved to reviewer1_results.json.
"""
import argparse
import json
import random
from collections import Counter

import numpy as np
import torch

import Edge
from Edge import (EdgeSystemSimulator, ImageAnalyzer, FarsightedA2CAgent,
                  get_dataset_frame_paths, load_real_frame, generate_synthetic_frame,
                  run_dataset_training, set_seed)
from baseline_comparison import (RandomPolicy, CheapestPolicy, StaticMediumPolicy,
                                 TrainedRLPolicy, evaluate_policy)

VARIANTS = ['light', 'medium', 'heavy']


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
class FixedPolicy:
    def __init__(self, variant, cut):
        self.variant, self.cut = variant, cut
        self.name = f"{VARIANTS[variant]}, cut {cut}"

    def act(self, state):
        return (self.variant, self.cut)


class GreedyRLPolicy:
    """Frozen policy, always taking the most probable valid action."""
    name = "Our RL (greedy)"

    def __init__(self, agent):
        self.agent = agent
        self.counts = Counter()

    def act(self, state):
        with torch.no_grad():
            p, _ = self.agent.network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            p = p.squeeze(0) * self.agent.mask
            idx = int(torch.argmax(p).item())
        a = (idx // self.agent.num_cutpoints, idx % self.agent.num_cutpoints)
        self.counts[a] += 1
        return a


class CountingPolicy:
    """Wraps a policy and counts which (variant, cut) it picks during evaluation."""
    def __init__(self, inner):
        self.inner, self.name, self.counts = inner, inner.name, Counter()

    def act(self, state):
        a = self.inner.act(state)
        self.counts[tuple(a)] += 1
        return a


# ----------------------------------------------------------------------------
# Training with a recorded agent (no change to Edge.py)
# ----------------------------------------------------------------------------
_RECORDED = []
_FORCE_EPS = {'value': None}


class _RecordingAgent(FarsightedA2CAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _RECORDED.append(self)

    def get_action(self, state_vector, epsilon=0.1):
        if _FORCE_EPS['value'] is not None:
            epsilon = _FORCE_EPS['value']
        return super().get_action(state_vector, epsilon=epsilon)


def train_and_get_agent(seed, episodes, dataset_dir, force_epsilon=None):
    _RECORDED.clear()
    _FORCE_EPS['value'] = force_epsilon
    original = Edge.FarsightedA2CAgent
    Edge.FarsightedA2CAgent = _RecordingAgent
    try:
        results = run_dataset_training(num_episodes=episodes, seed=seed, dataset_dir=dataset_dir,
                                       frame_sampling='random', verbose=False)
    finally:
        Edge.FarsightedA2CAgent = original
        _FORCE_EPS['value'] = None
    return _RECORDED[-1], results


# ----------------------------------------------------------------------------
# 1. Scene-class distribution
# ----------------------------------------------------------------------------
def class_distribution(dataset_dir, n=1000, seed=42):
    set_seed(seed)
    analyzer = ImageAnalyzer()
    paths = get_dataset_frame_paths(dataset_dir)
    counts = Counter()
    for _ in range(n):
        frame = (load_real_frame(random.choice(paths)) if paths
                 else generate_synthetic_frame(random.choice(['low', 'medium', 'high'])))
        _, d = analyzer.analyze(frame)
        counts['low' if d < 0.04 else 'medium' if d < 0.12 else 'high'] += 1
    return {k: counts[k] / n for k in ('low', 'medium', 'high')}


def fmt(r):
    return (f"| {r['policy']} | {r['accuracy']:.1f}% | {r['latency']:.3f} | "
            f"{r['energy']:.2f} | {r['drops']:.2f} |")


HEADER = "| Policy | Accuracy | Latency (s) | Energy (J) | Drops/ep |\n|---|---|---|---|---|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default="TuSimpleDatasetarchive/TUSimple/train_set/clips")
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--eval-episodes', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--class-samples', type=int, default=1000)
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--ablate-epsilon', action='store_true',
                    help='also train each seed with epsilon=0 (pure policy sampling)')
    args = ap.parse_args()

    out = {}

    print("\n## 1. Scene-class distribution of random frames")
    dist = class_distribution(args.dataset_dir, args.class_samples)
    out['class_distribution'] = dist
    print({k: f"{v*100:.1f}%" for k, v in dist.items()})

    mask_agent = FarsightedA2CAgent()
    valid = [(int(i) // 7, int(i) % 7) for i in mask_agent.valid_action_indices]

    print("\n## 2/3. Fixed baselines (same harness as the paper's Table III)")
    base = [RandomPolicy(mask_agent), CheapestPolicy(), StaticMediumPolicy()]
    rows = [evaluate_policy(p, args.dataset_dir, args.eval_episodes, verbose=False) for p in base]
    out['baselines'] = rows
    print(HEADER)
    for r in rows:
        print(fmt(r))

    if not args.skip_sweep:
        print("\n## Best-of-17 sweep (all valid fixed actions)")
        sweep = [evaluate_policy(FixedPolicy(v, c), args.dataset_dir, args.eval_episodes,
                                 verbose=False) for v, c in valid]
        out['fixed_sweep'] = sweep
        print(HEADER)
        for r in sweep:
            print(fmt(r))

    print("\n## 4. Frozen-policy evaluation across seeds")
    variants = [('epsilon-greedy (as in paper)', None)]
    if args.ablate_epsilon:
        variants.append(('no epsilon (policy sampling only)', 0.0))

    out['seeds'] = {}
    for label, eps in variants:
        per_seed = {'sampled': [], 'greedy': [], 'final_actions': [], 'train_last20': []}
        for i in range(args.seeds):
            seed = args.base_seed + i
            agent, (rw, lat, en, dr, acc, fr) = train_and_get_agent(
                seed, args.episodes, args.dataset_dir, force_epsilon=eps)
            sampled = CountingPolicy(TrainedRLPolicy(agent))
            greedy = GreedyRLPolicy(agent)
            s = evaluate_policy(sampled, args.dataset_dir, args.eval_episodes, verbose=False)
            g = evaluate_policy(greedy, args.dataset_dir, args.eval_episodes, verbose=False)
            per_seed['sampled'].append(s)
            per_seed['greedy'].append(g)
            top = sampled.counts.most_common(1)[0]
            per_seed['final_actions'].append(
                {'action': f"{VARIANTS[top[0][0]]}, cut {top[0][1]}",
                 'share': top[1] / sum(sampled.counts.values())})
            per_seed['train_last20'] = per_seed['train_last20'] + [{
                'latency': float(np.mean(lat[-20:])), 'energy': float(np.mean(en[-20:])),
                'drops': float(np.mean(dr[-20:])), 'accuracy': float(np.mean(acc[-20:]) * 100)}]
            print(f"[{label}] seed {seed}: sampled {s['latency']:.3f}s/{s['energy']:.2f}J/"
                  f"{s['drops']:.2f} drops | greedy {g['latency']:.3f}s/{g['energy']:.2f}J/"
                  f"{g['drops']:.2f} drops | top action {per_seed['final_actions'][-1]}")
        out['seeds'][label] = per_seed

        print(f"\n### {label}: mean +/- std over {args.seeds} seeds, {args.eval_episodes} eval episodes")
        for key in ('sampled', 'greedy'):
            arr = {m: np.array([r[m] for r in per_seed[key]])
                   for m in ('accuracy', 'latency', 'energy', 'drops')}
            print(f"{key:8s} acc {arr['accuracy'].mean():.1f}+/-{arr['accuracy'].std():.1f}  "
                  f"lat {arr['latency'].mean():.3f}+/-{arr['latency'].std():.3f}  "
                  f"energy {arr['energy'].mean():.2f}+/-{arr['energy'].std():.2f}  "
                  f"drops {arr['drops'].mean():.2f}+/-{arr['drops'].std():.2f}")

    with open('reviewer1_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved reviewer1_results.json")


if __name__ == '__main__':
    main()
