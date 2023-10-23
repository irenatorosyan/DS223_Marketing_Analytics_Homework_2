"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *

import random
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

Bandit_Reward = [1, 2, 3, 4]
Bandit_Probabilities = [0.1, 0.2, 0.3, 0.4]
NumberOfTrials = 20000

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """
    A class for visualizing and exporting results of bandit algorithms.

    This class provides methods to plot and export cumulative rewards data
    obtained from Epsilon-Greedy and Thompson Sampling bandit algorithms.

    Attributes:
        None
    """
    def plot1(self, e_greedy_rewards, thompson_rewards):
        # Visualize the learning process for each algorithm (plot1())
        plt.figure(figsize=(12, 6))
        """
        Visualize the learning process for each algorithm.

        Args:
            e_greedy_rewards (list): List of cumulative rewards for Epsilon-Greedy algorithm.
            thompson_rewards (list): List of cumulative rewards for Thompson Sampling algorithm.

        Returns:
            None
        """
        # Linear Plot
        plt.subplot(1, 2, 1)
        plt.plot(e_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.title('Cumulative Rewards (Linear Scale)')
        plt.xlabel('Trials')
        plt.ylabel('Total Reward')
        plt.legend()

        # Log Plot
        plt.subplot(1, 2, 2)
        plt.plot(e_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, e_greedy_rewards, thompson_rewards):
        """
        Visualize cumulative rewards from Epsilon-Greedy and Thompson Sampling algorithms.

        Args:
            e_greedy_rewards (list): List of cumulative rewards for Epsilon-Greedy algorithm.
            thompson_rewards (list): List of cumulative rewards for Thompson Sampling algorithm.

        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        plt.plot(e_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.title('Cumulative Rewards')
        plt.xlabel('Trials')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.show()

    @staticmethod
    def csv(data, filename="result.csv"):
        """
        Export data to a CSV file.

        Args:
            data (list of tuples): A list of tuples containing Bandit information, reward, and algorithm used.
            filename (str): The name of the CSV file to be created. Default is "result.csv".

        Returns:
            None
        """
        df = pd.DataFrame(data, columns=["Bandit", "Reward", "Algorithm"])
        df.to_csv(filename, index=False)

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm implementation.

    This class represents the Epsilon-Greedy algorithm for solving the multi-armed bandit problem.
    It includes methods for pulling arms, updating the algorithm's state, conducting experiments,
    and reporting results.
    """
    def __init__(self, p):
        """
        Initialize the EpsilonGreedy bandit algorithm.

        Args:
            p (list): List of probabilities of bandit's rewards for each arm.
            n (int): Number of arms (bandits).
            k (numpy array): Number of times each arm was pulled.
            reward (numpy array): Sum of rewards for each arm.
            t (int): Time step.
        """
        self.p = p  # probabilities of bandit's reward
        self.n = len(p)
        self.k = np.zeros(self.n)  # number of times arm was pulled
        self.reward = np.zeros(self.n)  # sum of rewards for each arm
        self.t = 10

    def __repr__(self):
        """
        Return a string representation of the EpsilonGreedy instance.

        Returns:
            str: String representation of the instance.
        """
        return f"EpsilonGreedy({self.p})"

    def pull(self):
        """
        Choose an arm to pull using the Epsilon-Greedy strategy.

        Returns:
            int: Index of the chosen arm.
        """
        epsilon = 1 / self.t
        if random.random() < epsilon:
            chosen_bandit = random.choice(range(self.n))
        else:
            chosen_bandit = np.argmax(self.reward / (self.k + 1e-5))  # exploit
        return chosen_bandit

    def update(self, chosen_bandit):
        """
        Update the algorithm's state after pulling an arm.

        Args:
            chosen_bandit (int): Index of the chosen arm.

        Returns:
            int: Reward obtained from the chosen arm.
        """
        self.t += 1
        self.k[chosen_bandit] += 1
        reward = 1 if random.random() < self.p[chosen_bandit] else 0
        self.reward[chosen_bandit] += reward
        return reward

    def experiment(self):
        """
        Conduct an experiment to collect cumulative rewards.

        Returns:
            list: List of cumulative rewards obtained during the experiment.
        """
        cumulative_rewards = []
        total_reward = 0
        for _ in range(NumberOfTrials):
            chosen_bandit = self.pull()
            reward = self.update(chosen_bandit)
            total_reward += reward
            cumulative_rewards.append(total_reward)
        return cumulative_rewards

    def report(self):
        """
        Report the results of the EpsilonGreedy algorithm.

        Prints average reward and average regret.
        """
        avg_reward = np.sum(self.reward) / self.t
        optimal_reward = max(self.p) * self.t
        avg_regret = optimal_reward - np.sum(self.reward)
        # Saving to csv can be done using pandas
        print(f"Average Reward for EpsilonGreedy: {avg_reward}")
        print(f"Average Regret for EpsilonGreedy: {avg_regret}")

#--------------------------------------#


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit algorithm implementation.

    This class represents the Thompson Sampling algorithm for solving the multi-armed bandit problem.
    It includes methods for pulling arms, updating the algorithm's state, conducting experiments, and reporting results.
    """
    def __init__(self, p):
        """
        Initialize the ThompsonSampling bandit algorithm.

        Args:
            p (list): List of probabilities of bandit's rewards for each arm.
            n (int): Number of arms (bandits).
            alpha (numpy array): Alpha parameters for the Beta distribution.
            beta (numpy array): Beta parameters for the Beta distribution.
        """
        self.p = p
        self.n = len(p)
        self.alpha = np.ones(self.n)
        self.beta = np.ones(self.n)

    def __repr__(self):
        """
        Return a string representation of the ThompsonSampling instance.

        Returns:
            str: String representation of the instance.
        """
        return f"ThompsonSampling({self.p})"

    def pull(self):
        """
        Choose an arm to pull using the Thompson Sampling strategy.

        Returns:
            int: Index of the chosen arm.
        """
        sampled_probs = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        chosen_bandit = np.argmax(sampled_probs)
        return chosen_bandit

    def update(self, chosen_bandit):
        """
        Update the algorithm's state after pulling an arm.

        Args:
            chosen_bandit (int): Index of the chosen arm.

        Returns:
            int: Reward obtained from the chosen arm.
        """
        reward = 1 if random.random() < self.p[chosen_bandit] else 0
        if reward == 1:
            self.alpha[chosen_bandit] += 1
        else:
            self.beta[chosen_bandit] += 1
        return reward

    def experiment(self):
        """
        Conduct an experiment to collect cumulative rewards.

        Returns:
            list: List of cumulative rewards obtained during the experiment.
        """
        cumulative_rewards = []
        total_reward = 0
        for _ in range(NumberOfTrials):
            chosen_bandit = self.pull()
            reward = self.update(chosen_bandit)
            total_reward += reward
            cumulative_rewards.append(total_reward)
        return cumulative_rewards

    def report(self):
        """
        Report the results of the ThompsonSampling algorithm.

        Prints average reward and average regret.
        """
        total_reward = np.sum([a / (a + b) for a, b in zip(self.alpha, self.beta)])
        avg_reward = total_reward / (np.sum(self.alpha) + np.sum(self.beta) - 2 * len(self.alpha))
        optimal_reward = max(self.p) * (np.sum(self.alpha) + np.sum(self.beta) - 2 * len(self.alpha))
        avg_regret = optimal_reward - total_reward
        # Saving to CSV can be done using pandas
        print(f"Average Reward for ThompsonSampling: {avg_reward}")
        print(f"Average Regret for ThompsonSampling: {avg_regret}")

#--------------------------------------#


def comparison(probabilities):
    """
    Compare the Epsilon-Greedy and Thompson Sampling bandit algorithms.

    This function compares the performance of Epsilon-Greedy and Thompson Sampling bandit algorithms
    based on a given set of probabilities for bandit rewards. It conducts experiments, visualizes
    the results, reports performance metrics, and saves the results to a CSV file.

    Args:
        probabilities (list): List of probabilities of bandit's rewards for each arm.

    Returns:
        None
    """
    e_greedy = EpsilonGreedy(probabilities)
    thompson = ThompsonSampling(probabilities)

    e_greedy_rewards = e_greedy.experiment()
    thompson_rewards = thompson.experiment()

    # Visualization
    viz = Visualization()
    viz.plot1(e_greedy_rewards, thompson_rewards)
    viz.plot2(e_greedy_rewards, thompson_rewards)

    # Report
    e_greedy.report()
    thompson.report()

    # Save results to CSV
    results_data = [("EpsilonGreedy", e_greedy_rewards[-1], "Epsilon-Greedy"),
                    ("ThompsonSampling", thompson_rewards[-1], "Thompson Sampling")]
    viz.csv(results_data, filename="experiment_results.csv")


if __name__=='__main__':

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison(Bandit_Probabilities)
