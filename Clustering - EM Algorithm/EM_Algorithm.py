from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# Generate the data according to the specification in the homework description

N = 500
x = np.random.rand(N)

pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([0.4, 0.3])

y = np.zeros_like(x)
for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y[i] = w0[k] * x[i] + b0[k] + np.random.randn() * sigma0[k]

# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.

# E-step: p(z_i=k | x_i, y_i ; 𝜽) = (π_k * φ(y_i; wx+b, σ^2)) / sigma from 1 to K (π_k' * φ(y_i; w'x+b', σ^2'))
#           then normalize, for example: p(z_i=1 | x_i=1, y_i=1) = p(z_i=1 | x_i=1, y_i=1) / (p(z_i=1 | x_i=1, y_i=1) + p(z_i=2 | x_i=1, y_i=1))
# M-Step: update 𝜽, including π, w, σ

pi = np.array([0.5, 0.5])
w = np.array([1.0, -1.0])
b = np.array([0.0, 0.0])
standard_deviation = np.array([np.std(y), np.std(y)])
r_0 = np.zeros(N)
r_1 = np.zeros(N)

iteration_count = 0
iteration_list = list()
marginal_log_likelihood_list = list()
while(True):
    iteration_count += 1

    # compute marginal log-likelihood
    marginal_log_likelihood = 0
    for i in range(N):
        marginal_log_likelihood_temp = 0
        for k in range(2):
            marginal_log_likelihood_temp += pi[k] * 1/(np.sqrt(2 * np.pi) * standard_deviation[k]) * np.exp(- ((y[i] - w[k] * x[i] - b[k])**2) / (2 * standard_deviation[k]**2))
        marginal_log_likelihood += np.log(marginal_log_likelihood_temp)
    #print(f"marginal_log_likelihood = {marginal_log_likelihood}")

    # stop when the log-likelihood increases by less than 0.0001
    if iteration_count > 1 and abs(marginal_log_likelihood - marginal_log_likelihood_list[-1]) < 0.0001:
        break

    iteration_list.append(iteration_count)
    marginal_log_likelihood_list.append(marginal_log_likelihood)

    # E-step: compute r_ik
    for i in range(N):
        denominator = 0
        for k in range(2):
            denominator += pi[k] * 1 / (np.sqrt(2 * np.pi) * standard_deviation[k]) * np.exp(-((y[i] - w[k] * x[i] - b[k])**2) / (2 * standard_deviation[k]**2))

        r_i_0 = (pi[0] * 1/(np.sqrt(2 * np.pi) * standard_deviation[0]) * np.exp(-((y[i] - w[0] * x[i] - b[0])**2)/(2 * standard_deviation[0]**2))) / denominator
        r_i_1 = (pi[1] * 1/(np.sqrt(2 * np.pi) * standard_deviation[1]) * np.exp(-((y[i] - w[1] * x[i] - b[1])**2)/(2 * standard_deviation[1]**2))) / denominator

        r_0[i] = r_i_0 / (r_i_0 + r_i_1)
        r_1[i] = r_i_1 / (r_i_0 + r_i_1)

    # M-Step: update 𝜽, including π, w, σ
    # update pi
    pi[0] = np.sum(r_0) / N
    pi[1] = np.sum(r_1) / N

    # update w
    # update b
    w_tilde_0 = np.zeros((2, 2))
    w_tilde_1 = np.zeros((2, 2))
    r_y_x_0 = np.zeros(2)
    r_y_x_1 = np.zeros(2)
    for i in range(N):
        w_tilde_temp = np.array([x[i], 1])
        out_product = np.outer(w_tilde_temp, w_tilde_temp)
        # compute w_0 and b_0
        w_tilde_0 += r_0[i] * out_product
        w_tilde_1 += r_1[i] * out_product

        r_y_x_0 += r_0[i] * y[i] * w_tilde_temp
        r_y_x_1 += r_1[i] * y[i] * w_tilde_temp

    for i in range(2):
        diagonal_value = w_tilde_0[i][i]
        w_tilde_0[i][i] = diagonal_value + 0.000000001

        diagonal_value = w_tilde_1[i][i]
        w_tilde_1[i][i] = diagonal_value + 0.000000001

    w_tilde_invert_0 = np.linalg.inv(w_tilde_0)
    w_tilde_invert_1 = np.linalg.inv(w_tilde_1)
    w_tilde_0 = np.matmul(w_tilde_invert_0, r_y_x_0)
    w_tilde_1 = np.matmul(w_tilde_invert_1, r_y_x_1)

    w[0] = w_tilde_0[0]
    b[0] = w_tilde_0[1]
    w[1] = w_tilde_1[0]
    b[1] = w_tilde_1[1]

    # update standard deviation
    numerator_0 = 0
    numerator_1 = 0
    for i in range(N):
        numerator_0 += r_0[i] * (y[i] - w[0] * x[i] - b[0])**2
        numerator_1 += r_1[i] * (y[i] - w[1] * x[i] - b[1])**2
    standard_deviation[0] = np.sqrt(numerator_0 / np.sum(r_0))
    standard_deviation[1] = np.sqrt(numerator_1 / np.sum(r_1))

print(f"iteration_count = {iteration_count}")
print(f"pi = {pi}")
print(f"w = {w}")
print(f"b = {b}")
print(f"standard deviation = {standard_deviation}")
#print(f"r = {(r_0, r_1)}")

# plot of iteration number vs marginal log-likelihood
plt.figure(figsize=(10, 5))
plt.plot(marginal_log_likelihood_list, c="b", marker="o")
plt.title("Iteration vs Log-Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.show()


# Here's the data plotted
plt.scatter(x, y, marker="x", alpha=0.5, label="Data")
plt.plot(np.sort(x), w[0] * np.sort(x) + b[0], label="Estimated Line 1", color="black")
plt.plot(np.sort(x), w[1] * np.sort(x) + b[1], label="Estimated Line 2", color="red")
plt.title('Data and Regression Lines')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()