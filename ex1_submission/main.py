from env import BanditEnv
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from agent import EpsilonGreedy
from agent import UCB



def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()
    
    rewards = [[] for _ in range(k)]
    for arm in range(k):
        for _ in range(num_samples):
            reward = env.step(arm)
            rewards[arm].append(reward)

    # Plot the rewards
    plt.figure(figsize=(10,6))
    sns.violinplot(data=rewards)
    plt.title("Distribution of rewards for each arm")
    plt.xlabel("Arm")
    plt.ylabel("Reward")
    plt.show()

    # TODO
    pass


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    epsilons = [0, 0.01, 0.1]
    agents = [EpsilonGreedy(k, init=0, epsilon=eps) for eps in epsilons]

    avg_rewards = np.zeros((len(agents), steps))
    optimal_action_count = np.zeros((len(agents), steps))
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward, _ = env.step(action)
                agent.update(action, reward)

                avg_rewards[i][step] += reward
                if action == np.argmax(env.means):
                    optimal_action_count[i][step] += 1

    avg_rewards /= trials
    optimal_action_count = (optimal_action_count / trials) * 100  # convert to percentage
        # TODO For each trial, perform specified number of steps for each type of agent
    plt.figure(figsize=(10,6))
    
    for i, eps in enumerate(epsilons):
        plt.plot(avg_rewards[i], label=f'ε = {eps}')
    
    best_possible_reward = np.max(env.means)
    plt.plot([best_possible_reward]*steps, '--', label="Best possible reward")

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("ε-Greedy: Average Rewards over Steps")
    plt.show()

    plt.figure(figsize=(10,6))
    for i, eps in enumerate(epsilons):
        plt.plot(optimal_action_count[i], label=f'ε = {eps}')
    
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.title("ε-Greedy: % Optimal Action over Steps")
    plt.show()

    for i, eps in enumerate(epsilons):
        mean = avg_rewards[i]
        stderr = np.std(mean) / np.sqrt(trials)
        plt.plot(mean, label=f'ε = {eps}')
        plt.fill_between(range(steps), mean - 1.96*stderr, mean + 1.96*stderr, alpha=0.2)

    pass


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    agents = [
        EpsilonGreedy(k, init=0, epsilon=0),
        EpsilonGreedy(k, init=5, epsilon=0),
        EpsilonGreedy(k, init=0, epsilon=0.1),
        EpsilonGreedy(k, init=5, epsilon=0.1),
        UCB(k, init=0, c=2, step_size=0.1)
    ]

    avg_rewards = np.zeros((len(agents), steps))
    optimal_action_count = np.zeros((len(agents), steps))

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()

        # TODO For each trial, perform specified number of steps for each type of agent
        for step in range(steps):

            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward, _ = env.step(action)
                agent.update(action, reward)

                avg_rewards[i][step] += reward
                if action == np.argmax(env.means):
                    optimal_action_count[i][step] += 1

    avg_rewards /= trials
    optimal_action_count = (optimal_action_count / trials) * 100  

    labels = ["ε=0, Q1=0", "ε=0, Q1=5", "ε=0.1, Q1=0", "ε=0.1, Q1=5", "UCB, c=2"]
    
    # Plotting % Optimal Action
    plt.figure(figsize=(10,6))
    for i, label in enumerate(labels):
        plt.plot(optimal_action_count[i], label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.title("ε-Greedy & UCB: % Optimal Action over Steps")
    plt.show()
    
    # Plotting Average Rewards
    plt.figure(figsize=(10,6))
    best_possible_reward = np.max(env.means)
    for i, label in enumerate(labels):
        mean = avg_rewards[i]
        stderr = np.std(mean) / np.sqrt(trials)
        plt.plot(mean, label=label)
        plt.fill_between(range(steps), mean - 1.96*stderr, mean + 1.96*stderr, alpha=0.2)
    plt.plot([best_possible_reward]*steps, '--', label="Best possible reward")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("ε-Greedy & UCB: Average Rewards over Steps")
    plt.show()

    pass


def main():
    # TODO run code for all questions
    # q4(10, 1000)  # 10 arms, 1000 samples each       #uncomment this line for q4, comment while running q6,q7
    # q6(10, 2000, 2000)  #comment this line for q4, q7, uncomment while running q6
    q7(10, 2000, 1000)   #uncomment this line for q7, comment while running q6,q4

    pass


if __name__ == "__main__":
    main()
