import rvo2
import numpy as np
from numpy.linalg import norm
from env import ObstacleEnv  # Import your environment setup
import argparse

class RVOBaseline:
    def __init__(self):
        self.env = ObstacleEnv()

    def run_baseline(self, num_episodes=100):
        success_rate = 0
        collision_rate = 0
        timeout_rate = 0
        rewards = []

        for episode in range(num_episodes):
            obs = self.env.reset(obstacle_num=10, layout="circle")
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                # Default RVO2 behavior: Follow preferred velocity directly
                velocities = [
                    self.env.sim.getAgentPrefVelocity(i)
                    for i in range(self.env.sim.getNumAgents())
                ]
                action = np.array(velocities[0])  # Robot's preferred velocity
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                steps += 1

            # Outcome tracking
            if info == "collision":
                collision_rate += 1
            elif info == "timeout":
                timeout_rate += 1
            else:
                success_rate += 1

            rewards.append(episode_reward)

            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, Outcome: {info}")

        # Calculate baseline metrics
        results = {
            "success_rate": success_rate / num_episodes,
            "collision_rate": collision_rate / num_episodes,
            "timeout_rate": timeout_rate / num_episodes,
            "average_reward": np.mean(rewards)
        }

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVO2 Baseline Evaluation")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()

    baseline = RVOBaseline()
    results = baseline.run_baseline(num_episodes=args.eval_episodes)

    print("\nBaseline Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value:.2f}")
