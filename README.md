# DS223_Marketing_Analytics_Homework_2

# **A/B Testing with Epsilon Greedy and Thompson Sampling**

## **Introduction**

This project involves designing and implementing an A/B testing experiment using two different multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling.

## **Experiment Design**

### **Bandit Class**

A `Bandit` class serves as an abstract template for implementing bandit algorithms. It defines abstract methods that must be implemented by any specific bandit algorithm. The class has predefined abstract methods for initialization, pulling an arm, updating the algorithm's state, conducting experiments, and reporting results.

### **Experiment Parameters**

- `Bandit_Reward`: A list representing the reward values for the four bandits.
- `Bandit_Probabilities`: A list representing the probabilities of reward for each bandit.
- `NumberOfTrials`: The number of trials to conduct in the experiment (set to 20,000).

## **Bandit Algorithms**

### **Epsilon-Greedy**

- Implements the Epsilon-Greedy algorithm for multi-armed bandits.
- Decays epsilon by 1/t.
- Conducts experiments to collect cumulative rewards.
- Reports average reward and average regret.

### **Thompson Sampling**

- Implements the Thompson Sampling algorithm for multi-armed bandits.
- Designs with known precision.
- Conducts experiments to collect cumulative rewards.
- Reports average reward and average regret.

## **Visualization and Reporting**

The project includes a `Visualization` class for visualizing and exporting the results of bandit algorithms. It offers the following methods:

- `plot1(e_greedy_rewards, thompson_rewards)`: Visualizes the learning process for each algorithm, providing both linear and log-scale plots.
- `plot2(e_greedy_rewards, thompson_rewards)`: Visualizes cumulative rewards from Epsilon-Greedy and Thompson Sampling.
- `csv(data, filename="result.csv")`: Exports data to a CSV file, containing Bandit information, reward, and the algorithm used.

## **Experiment and Comparison**

The `comparison(probabilities)` function is the core of the project. It compares the performance of Epsilon-Greedy and Thompson Sampling algorithms using the provided probabilities for bandit rewards. It conducts experiments, visualizes the results, reports performance metrics (average reward and regret), and saves the results to a CSV file.

## **Usage**

To run the A/B testing experiment:

1. Define the `Bandit_Reward`, `Bandit_Probabilities`, and `NumberOfTrials`.
2. Create an instance of the `EpsilonGreedy` and `ThompsonSampling` classes with the given probabilities.
3. Call the `comparison(probabilities)` function to compare the two algorithms.

```python
if __name__=='__main__':
    comparison(Bandit_Probabilities)
```

By Irena Torosyan
