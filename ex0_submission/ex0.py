import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import random
from typing import Tuple #for Q3


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    # TODO check if goal was reached
    # If in goal state, reset to start and give a reward of 1 as described
    goal_state = (10, 10)
    if state == goal_state:
        return reset(), 1.0

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action) as descibed below
    action_taken = action
    # Decide the actual action taken
    rand_num = np.random.rand()
    if rand_num <= 0.9:
        action_taken = action
    elif rand_num <= 0.95:
        # first perpendicular action
        action_taken = Action((action + 1) % 4)
    else:
        # second perpendicular action
        action_taken = Action((action + 3) % 4)

    # TODO calculate the next state and reward given state and action_taken (check those special cases)
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall

    dx, dy = actions_to_dxdy(action_taken)
    next_state = (state[0] + dx, state[1] + dy)

    # If the next state is a wall or out of bounds, stay in the current state
    if next_state in walls or next_state[0] < 0 or next_state[0] > 10 or next_state[1] < 0 or next_state[1] > 10:
        next_state = state

    # Give a reward if the agent reaches the goal
    reward = 1.0 if next_state == goal_state else 0.0

    # next_state = None
    # reward = None

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    print(f"Current state: {state}")
    print("Choose an action: [LEFT, DOWN, RIGHT, UP]")
    action_str = input().upper()
    
    action_mapping = {
        "LEFT": Action.LEFT,
        "DOWN": Action.DOWN,
        "RIGHT": Action.RIGHT,
        "UP": Action.UP,
    }
    
    return action_mapping.get(action_str, Action.LEFT)  # Default to LEFT if invalid input, asks the user for input
    pass


# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    all_rewards = []  # List to record cumulative rewards for each trial as a list

    for t in range(trials):
        state = reset()
        total_reward = 0  # Reset total reward for each trial
        trial_rewards = [0]  # Start with 0 reward
        # print(f"Trial {t+1}, Total Reward: {all_rewards}")      #uncomment this to view cummulative3 rewards

        for _ in range(steps):
            action = policy(state)
            next_state, reward = simulate(state, action)
            total_reward += reward
            trial_rewards.append(total_reward)
            state = next_state

        all_rewards.append(trial_rewards)

    return all_rewards


# Q3
def random_policy(state: Tuple[int, int]) -> Action:
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return random.choice(list(Action))    #takes a random action, explorartion
    pass


# Q4
def worse_policy(state: Tuple[int, int]) -> Action:
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return Action.UP        #takes an action always, greedy, no learning
    pass


# Q4
def better_policy(state: Tuple[int, int])-> Action:
    """A policy that is better than the random_policy
       A policy that prioritizes UP and RIGHT 80% of the time.
    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

        last_states (List[Tuple[int, int]]): The last few states the agent was in.
        max_repeat (int): The maximum number of times we allow the agent to be in the same state consecutively.

    Returns:
        action (Action)
    """
    # TODO     favours the direction towrds the goal to add balance in exploration and not get stuck
    

    favored_actions = [Action.UP, Action.RIGHT]
    if random.random() < 0.8:
        return random.choice(favored_actions)
    else:
        # Picking from other actions
        other_actions = [Action.LEFT, Action.DOWN]
        return random.choice(other_actions)

    pass


def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question, use this for plots
    policies = {
        "Random Policy": random_policy,
        "Better Policy": better_policy,
        "Worse Policy": worse_policy
    }

    all_avg_rewards = {}  # Dictionary to store average rewards of each policy

    for policy_name, policy in policies.items():
        print(f"Running agent with {policy_name}...")
        rewards = agent(steps=10**4, trials=10, policy=policy)

        # Plotting the results
        for trial_rewards in rewards:
            plt.plot(trial_rewards, linestyle='dotted', color='grey')

        average_rewards = np.mean(rewards, axis=0)
        all_avg_rewards[policy_name] = average_rewards
        plt.plot(average_rewards, label=policy_name, linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward over Time ({policy_name})')
        plt.legend()
        plt.show()

    # Fourth plot for all policies together
    for policy_name, average_rewards in all_avg_rewards.items():
        plt.plot(average_rewards, label=policy_name, linewidth=2)
        
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Time (All Policies)')
    plt.legend()
    plt.show()
    # pass


if __name__ == "__main__":
    # agent(steps=100, trials=1, policy=manual_policy)  #uncomment this to run manual policy for Q1, comment this while running Q2, Q3 and Q4
    


    main() #uncomment this to run Random, Better and Worse policy for Q2, Q3 and Q4, comment this while running Q1
