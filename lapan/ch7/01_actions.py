import numpy as np
import ptan

if __name__ == '__main__':
    # batch of two actions set
    q_vals = np.array([[1, 2, 3], [1, 0, -1]])
    # argmax - id of action with max value
    selector = ptan.actions.ArgmaxActionSelector()
    print(f"Batch of values {q_vals}")
    print(f"Argmax selector: {selector(q_vals)}")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0)
    print(f"Epsilon greedy selector, epsilon is 0: {selector(q_vals)}")
    selector.epsilon = 1 # means completly random selection
    print(f"Epsilon greedy selector, epsilon is 1: {selector(q_vals)}")

    selector = ptan.actions.ProbabilityActionSelector()
    for _ in range(10):
        # action set is normalized probability distribution
        acts = selector(np.array([
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ]))
        print(acts)