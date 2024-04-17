from math import sqrt, exp, pi

# Given data
observations = [2, 3, 5, 10, 12]
x = 4
sigma_squared = 4
N = len(observations)

# Gaussian kernel density function
def gaussian_kernel_density(observations, x, sigma_squared):
    N = len(observations)
    kernel_sum = sum(exp(-(x - xi)**2 / (2 * sigma_squared)) for xi in observations)
    return (1 / (N * sqrt(2 * pi * sigma_squared))) * kernel_sum

# Calculate the kernel density estimate at x=4
kde = gaussian_kernel_density(observations, x, sigma_squared)
print(kde)

# Given variance
sigma = sqrt(sigma_squared)
normalizing_factor = 1 / (N * sqrt(2 * pi * sigma_squared))

# Define the exponential part for each observation
def exponential_part(obs, x, sigma):
    return exp(-(x - obs)**2 / (2 * sigma**2))

# Calculating the KDE for each answer
answers = {
    'A': normalizing_factor * exp(-53 / 4),
    'B': normalizing_factor * exp(-53 / 4) / 5,
    'C': normalizing_factor * (exponential_part(2, x, sigma) + 2 * exponential_part(3, x, sigma) +
                               exponential_part(5, x, sigma) + exponential_part(10, x, sigma) +
                               exponential_part(12, x, sigma)),
    'D': normalizing_factor * (exp(-1) + 2 * exp(-1/4) + exp(-9) + exp(-16)),
}

# Calculated KDE value
calculated_kde = kde

# Compare the calculated KDE to each answer
kde_comparison = {key: abs(calculated_kde - value) for key, value in answers.items()}
print(kde_comparison)


### SVAR er C) --> Men ikke sikker