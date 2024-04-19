import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    hidden_states_list = [0, 1, 2]
    observations = [0, 1, 0, 1]

    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.4, 0.4],
                  [0.4, 0.1, 0.5]])
    B = np.array([[0.8, 0.2],
                  [0.1, 0.9],
                  [0.5, 0.5]])
    pi = np.array([0.5, 0.3, 0.2])

    results = list()

    index = 0
    for i in hidden_states_list:
        for j in hidden_states_list:
            for k in hidden_states_list:
                for l in hidden_states_list:
                    sequence = [i, j, k, l]

                    # Forward Algorithm
                    T = len(observations)
                    N = len(pi)
                    fwd = np.zeros((T, N))
                    fwd[0] = pi * B[:, observations[0]]
                    for t in range(1, T):
                        fwd[t] = (fwd[t - 1] @ A) * B[:, observations[t]]

                    total_prob_obs = np.sum(fwd[-1])

                    # Calculating details for a specific sequence
                    prior = pi[sequence[0]]
                    likelihood = B[sequence[0], observations[0]]
                    for t in range(1, len(sequence)):
                        prior *= A[sequence[t - 1], sequence[t]]
                        likelihood *= B[sequence[t], observations[t]]

                    posterior = (prior * likelihood) / total_prob_obs

                    sequence_str = str(i) + str(j) + str(k) + str(l)
                    results.append((sequence_str, prior, likelihood, posterior))
                    index += 1

    print(results)
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    print(results_sorted)

    #results[results[:, 3].argsort()]



