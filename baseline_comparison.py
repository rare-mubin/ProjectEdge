"""
Baseline comparison harness for the advisor-requested Table (advisor point 4).

Implements three fixed, non-learning policies -- Random, Cheapest, Static
Medium -- evaluated under the *same* EdgeSystemSimulator cost model, queue
dynamics, and bandwidth trace generation used for the trained RL policy, so
all four rows in the resulting table are apples-to-apples.

IMPORTANT: this script reuses classes directly from Edge.py
(EdgeSystemSimulator, ImageAnalyzer, the frame loaders, FarsightedA2CAgent)
rather than reimplementing them, so there is exactly one source of truth for
the cost model. Put this file in the same directory as Edge.py.

WHAT THIS SCRIPT DOES NOT DO: it does not have access to your actual trained
200-episode Jetson policy, and it cannot download the real TuSimple dataset
in this environment. Running it here only proves the code is correct
(smoke-tested on synthetic frames) -- it does NOT produce numbers that
belong in the paper. See the bottom of this file / the accompanying message
for what to run on your end to get real numbers.
"""
import random
import numpy as np
import torch

from Edge import (
    EdgeSystemSimulator, ImageAnalyzer, FarsightedA2CAgent,
    get_dataset_frame_paths, load_real_frame, generate_synthetic_frame,
    set_seed, safe_mean,
)


# ==========================================
# FIXED (NON-LEARNING) BASELINE POLICIES
# ==========================================
class RandomPolicy:
    """Samples uniformly at random from the 17 valid actions every step."""
    name = "Random"

    def __init__(self, agent_for_mask):
        # Reuse the trained agent's precomputed valid-action list so the
        # random baseline respects the same action mask, rather than
        # sampling from all 21 nominal actions.
        self.valid_action_indices = agent_for_mask.valid_action_indices
        self.num_cutpoints = agent_for_mask.num_cutpoints

    def act(self, state):
        idx = self.valid_action_indices[
            torch.randint(len(self.valid_action_indices), (1,))
        ].item()
        return (idx // self.num_cutpoints, idx % self.num_cutpoints)


class CheapestPolicy:
    """Always the lightest model at its earliest valid cut point
    (light, cutpoint 0) -- maximal offloading, minimal local compute."""
    name = "Cheapest"

    def act(self, state):
        return (0, 0)  # (light, cutpoint 0)


class StaticMediumPolicy:
    """Always the medium model at a fixed middle cut point (medium, 3)."""
    name = "Static Medium"

    def act(self, state):
        return (1, 3)  # (medium, cutpoint 3)


class TrainedRLPolicy:
    """Wraps a trained FarsightedA2CAgent in pure-exploitation mode
    (epsilon=0) so it can be evaluated under the identical harness used
    for the fixed baselines below."""
    name = "Our RL"

    def __init__(self, agent):
        self.agent = agent

    def act(self, state):
        action, _ = self.agent.get_action(state, epsilon=0.0)
        return action


# ==========================================
# SHARED EVALUATION HARNESS
# ==========================================
def evaluate_policy(policy, dataset_dir, num_episodes=20, steps_per_episode=30,
                     seed=42, frame_sampling='random', verbose=True):
    """
    Runs `policy` (anything with an .act(state) -> (complexity, cutpoint)
    method) through the same simulator, frame source, and bandwidth-trace
    generation as run_dataset_training(), but with NO learning updates --
    this is pure evaluation. Returns per-episode-mean accuracy, latency,
    energy, and drop count, matching the columns in Table~II of the paper.

    frame_sampling defaults to 'random' to match the sampling method
    actually used for the reported 200-episode "Our RL" run (uniform random
    draw from the full dataset each step) -- using 'sequential' here would
    make the baselines see a systematically different slice of the dataset
    than the trained policy did, undermining the comparison.
    """
    set_seed(seed)
    sim = EdgeSystemSimulator()
    analyzer = ImageAnalyzer()
    frame_paths = get_dataset_frame_paths(dataset_dir)
    use_real_dataset = len(frame_paths) > 0

    ep_accs, ep_lats, ep_energies, ep_drops = [], [], [], []

    for ep in range(1, num_episodes + 1):
        sim.reset()
        bandwidth_trace = sim.generate_bandwidth_trace(steps_per_episode)
        accs, lats, energies = [], [], []
        drops = 0

        for t in range(steps_per_episode):
            if use_real_dataset:
                if frame_sampling == 'random':
                    frame_idx = random.randrange(len(frame_paths))
                else:
                    frame_idx = ((ep - 1) * steps_per_episode + t) % len(frame_paths)
                try:
                    frame = load_real_frame(frame_paths[frame_idx])
                except Exception:
                    frame = generate_synthetic_frame('medium')
            else:
                frame = generate_synthetic_frame(random.choice(['low', 'medium', 'high']))

            norm_entropy, edge_density = analyzer.analyze(frame)
            if edge_density < 0.04:
                complexity_type = 'low'
            elif edge_density < 0.12:
                complexity_type = 'medium'
            else:
                complexity_type = 'high'

            bandwidth = bandwidth_trace[t]
            state = [
                sim.local_queue_time, sim.offload_queue_time, bandwidth,
                sim.local_queue_time + sim.offload_queue_time,
                norm_entropy, edge_density, sim.remaining_battery,
            ]

            action = policy.act(state)
            delay, energy, acc, dropped = sim.step(action, t, complexity_type, bandwidth)

            if dropped:
                drops += 1
                lats.append(sim.timeout_threshold)
                accs.append(0.0)
            else:
                lats.append(delay)
                accs.append(acc)
            energies.append(energy)

        ep_accs.append(safe_mean(accs))
        ep_lats.append(np.mean(lats))
        ep_energies.append(np.sum(energies))
        ep_drops.append(drops)

    result = {
        'policy': policy.name,
        'accuracy': float(np.mean(ep_accs)) * 100,
        'latency': float(np.mean(ep_lats)),
        'energy': float(np.mean(ep_energies)),
        'drops': float(np.mean(ep_drops)),
    }
    if verbose:
        print(f"{result['policy']:<15} | Acc: {result['accuracy']:5.1f}% | "
              f"Latency: {result['latency']:.3f}s | Energy: {result['energy']:6.2f}J | "
              f"Drops/ep: {result['drops']:.2f}")
    return result


def run_baseline_comparison(dataset_dir, trained_agent=None, num_episodes=20,
                             steps_per_episode=30, seed=42, frame_sampling='random'):
    """
    Evaluates Random, Cheapest, Static Medium, and (if a trained agent is
    passed in) the trained RL policy, all under the identical harness.
    Prints a table matching the advisor-requested format.
    """
    dummy_agent_for_mask = FarsightedA2CAgent()  # only used for its action mask
    policies = [
        RandomPolicy(dummy_agent_for_mask),
        CheapestPolicy(),
        StaticMediumPolicy(),
    ]
    if trained_agent is not None:
        policies.append(TrainedRLPolicy(trained_agent))

    print("=" * 70)
    print(f"Baseline comparison ({num_episodes} eval episodes, "
          f"{steps_per_episode} steps/episode, seed={seed}, "
          f"frame_sampling={frame_sampling})")
    print("=" * 70)
    results = [evaluate_policy(p, dataset_dir, num_episodes, steps_per_episode,
                                seed, frame_sampling)
               for p in policies]

    print("\n| Policy | Accuracy | Latency (s) | Energy (J) | Drops/ep |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['policy']} | {r['accuracy']:.1f}% | {r['latency']:.3f} | "
              f"{r['energy']:.2f} | {r['drops']:.2f} |")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str,
                         default="TuSimpleDatasetarchive/TUSimple/train_set/clips",
                         help="Point this at your real TuSimple clips directory "
                              "(same path you passed to Edge.py) to get real numbers.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-sampling", type=str, default="random",
                         choices=["random", "sequential"],
                         help="Should match what you used for the 'Our RL' training "
                              "run (random, per run_20260902_110013.log) for a fair "
                              "comparison.")
    args = parser.parse_args()

    print("NOTE: this run has no access to your trained 200-episode policy, "
          "so it reports Random / Cheapest / Static Medium only. To add the "
          "'Our RL' row under the identical harness, either (a) import "
          "run_dataset_training from Edge.py, capture the trained `agent` it "
          "builds internally, and call run_baseline_comparison(dataset_dir, "
          "trained_agent=agent, frame_sampling='random') right after training "
          "in the same Python session, or (b) use the 'Our RL' row already "
          "available from Table~II (episode 200: 85.0% acc, 0.474s, 28.17J, "
          "0 drops) as a stand-in, since the model-choice distribution shows "
          "the policy had already converged to a fixed action by then.\n")

    run_baseline_comparison(args.dataset_dir, trained_agent=None,
                             num_episodes=args.episodes, seed=args.seed,
                             frame_sampling=args.frame_sampling)
