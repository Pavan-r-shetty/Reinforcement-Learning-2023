from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import poisson


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
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


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # Special states, next locations, and their rewards
        self.A = (0, 1)
        self.A_prime = (4, 1)
        self.A_reward = 10
        self.B = (0, 3)
        self.B_prime = (2, 3)
        self.B_reward = 5 
        # # TODO set the locations of A and B, the next locations, and their rewards
        # self.A = None
        # self.A_prime = None
        # self.A_reward = None
        # self.B = None
        # self.B_prime = None
        # self.B_reward = None

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """
        x, y = state

        # Special states
        if state == self.A:
            return self.A_prime, self.A_reward
        if state == self.B:
            return self.B_prime, self.B_reward

        dx, dy = actions_to_dxdy(action)
        new_x, new_y = x + dx, y + dy

        # If next state is within boundaries
        if (0 <= new_x < self.rows) and (0 <= new_y < self.cols):
            next_state = (new_x, new_y)
            reward = 0
        else:  # trying to move outside grid, stay and get -1 reward
            next_state = state
            reward = -1

        return next_state, reward
        # next_state = None
        # reward = None

        # # TODO Check if current state is A and B and return the next state and corresponding reward
        # # Else, check if the next step is within boundaries and return next state and reward
        # return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

      
        next_state, reward = self.transitions(state, action)
        return reward + gamma * V[next_state]
        # # TODO compute the expected return
        # ret = None

        # return ret


class JacksCarRental:
    def __init__(self, modified: bool = False) -> None:
        """JacksCarRental

        Args:
           modified (bool): False = original problem Q6a, True = modified problem for Q6b

        State: tuple of (# cars at location A, # cars at location B)

        Action (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the probability of ending the day with s_end \in [0,20] cars given that the location started with s_start \in [0, 20+5] cars.

        Args:
            loc_idx (int): the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs (np.ndarray): list of probabilities for all possible combination of s_start and s_end
            rewards (np.ndarray): average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)
        for start in range(probs.shape[0]):
            # TODO Calculate average rewards.
            # For all possible s_start, calculate the probability of renting k cars.
            # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
            avg_rent = 0.0
            for k in range(start+1):
                avg_rent += min(k, start) * self.rent[loc_idx].pmf(k)
            rewards[start] = self.rent_reward * avg_rent

            # TODO Calculate probabilities
            # Loop over every possible s_end
            for end in range(probs.shape[1]):
                prob = 0.0
                # Since s_start and s_end are specified,
                # you must rent a minimum of max(0, start-end)
                min_rent = max(0, start - end)

                # TODO Loop over all possible rent scenarios and compute probabilities
                # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
                for i in range(min_rent, start + 1):
                    prob_rent = self.rent[loc_idx].pmf(i)
                    prob_return = self.return_[loc_idx].pmf(end - start + i)
                    prob += prob_rent * prob_return

                probs[start, end] = prob

        return probs, rewards

    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        """A helper function to compute the cost of moving cars for a given (state, action)

        Note that you should compute costs differently if this is the modified problem.

        Args:
            state (Tuple[int,int]): state
            action (int): action
        """
        if self.modified:
            cost = abs(action) * self.move_cost
            if state[0] > self.overflow_cars:
                cost += self.overflow_cost
            if state[1] > self.overflow_cars:
                cost += self.overflow_cost
        else:
            cost = abs(action) * self.move_cost

        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        """Helper function to check if this action is valid for the given state

        Args:
            state:
            action:
        """
        if state[0] < action or state[1] < -(action):
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # TODO Calculate day rewards from renting
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        self.r[locA, locB, ia] = day_rewards_A[locA] + day_rewards_B[locB] - self._calculate_cost((locA, locB), action)


                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):

                                # TODO Calculate transition probabilities
                                # Use the probabilities computed from open_to_close
                                self.t[locA, locB, ia, locA_, locB_] = day_probs_A[locA, locA_] * day_probs_B[locB, locB_]

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        locA, locB = state

        if not self._valid_action(state, action):
            return float('-inf')  # If not a valid action, return negative infinity

        ia = self.action_space.index(action)

        ret = self.r[locA, locB, ia]  # Immediate reward for taking the action

        # Loop through all possible next states
        for locA_ in range(self.max_cars_end + 1):
            for locB_ in range(self.max_cars_end + 1):
                transition_prob = self.t[locA, locB, ia, locA_, locB_]
                ret += gamma * transition_prob * V[locA_, locB_]

        return ret

    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        # TODO
        probs = self.t[state[0], state[1], action, :, :]
        return probs

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (Action): action
        Returns:
            reward: float
        """
        # TODO
        return self.r[state[0], state[1], action]

def iterative_policy_evaluation(
    gridworld: Gridworld5x5, gamma: float = 0.9, theta: float = 1e-3
) -> np.ndarray:
    V = np.zeros((gridworld.rows, gridworld.cols))
    while True:
        delta = 0
        for x in range(gridworld.rows):
            for y in range(gridworld.cols):
                state = (x, y)
                v = V[state]

                # Loop over actions for equiprobable random policy
                new_value = 0
                for action in Action:
                    prob = 1.0 / gridworld.action_space
                    new_value += prob * gridworld.expected_return(V, state, action, gamma)

                V[state] = new_value
                delta = max(delta, abs(v - V[state]))

        # Convergence check
        if delta < theta:
            break

    return V

def value_iteration(gridworld, gamma=0.9, theta=1e-4):
    """
    Perform value iteration on the given Gridworld
    Args:
        gridworld: The environment, an instance of Gridworld5x5
        gamma (float): discount factor
        theta (float): threshold for convergence
    Returns:
        V (np.ndarray): optimal state-value function
        policy (np.ndarray): optimal policy
    """
    # Step 1: Initialize the state-value function
    V = np.zeros((gridworld.rows, gridworld.cols))
    
    while True:
        delta = 0
        # Step 2: Iteratively update the value function
        for state in gridworld.state_space:
            v = V[state]
            V[state] = max([gridworld.expected_return(V, state, action, gamma) for action in Action])
            delta = max(delta, abs(v - V[state]))
        # Check for convergence
        if delta < theta:
            break

    # Step 3: Retrieve the optimal policy
    policy = np.zeros((gridworld.rows, gridworld.cols), dtype=int)
    for state in gridworld.state_space:
        action_returns = [gridworld.expected_return(V, state, action, gamma) for action in Action]
        policy[state] = np.argmax(action_returns)

    return V, policy


# #5 a

# gridworld = Gridworld5x5()
# V = iterative_policy_evaluation(gridworld)
# print(V)

# #5 a

# #5 b

# gridworld = Gridworld5x5()
# optimal_V, optimal_policy = value_iteration(gridworld)

# print("Optimal Value Function:")
# print(optimal_V)
# print("\nOptimal Policy:")
# print(optimal_policy)

# #5 b

##5 c

def policy_evaluation(gridworld, policy, V, gamma=0.9, theta=0.0001):
    """Evaluate the state-value function for the given policy."""
    while True:
        delta = 0
        for state in gridworld.state_space:
            v = V[state]
            new_value = 0
            for action in Action:
                next_state, reward = gridworld.transitions(state, action)
                new_value += policy[state][action] * (reward + gamma * V[next_state])
            V[state] = new_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(gridworld, policy, V, gamma=0.9):
    """Update policy based on the current state-value function."""
    policy_stable = True
    for state in gridworld.state_space:
        old_action = np.argmax(policy[state])
        action_values = np.zeros(len(Action))
        for action in Action:
            next_state, reward = gridworld.transitions(state, action)
            action_values[action] = reward + gamma * V[next_state]
        best_action = np.argmax(action_values)
        policy[state] = np.eye(len(Action))[best_action]
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def policy_iteration(gridworld, gamma=0.9):
    """Policy Iteration algorithm."""
    V = {}
    policy = {}
    for state in gridworld.state_space:
        V[state] = 0
        policy[state] = np.ones(len(Action)) / len(Action)

    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(gridworld, policy, V, gamma)
        policy, policy_stable = policy_improvement(gridworld, policy, V, gamma)

    return V, policy

gridworld = Gridworld5x5()
V_star, pi_star = policy_iteration(gridworld)

# Print the optimal value function and policy
print("Optimal Value Function:")
for i in range(gridworld.rows):
    for j in range(gridworld.cols):
        print(f"{V_star[(i,j)]:.2f}", end="\t")
    print()

print("\nOptimal Policy:")
for i in range(gridworld.rows):
    for j in range(gridworld.cols):
        print(Action(np.argmax(pi_star[(i,j)])), end="\t")
    print()

##5 c










