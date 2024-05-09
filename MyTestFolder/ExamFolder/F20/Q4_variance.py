def calculate_correlation(covariance, variance_x3, variance_x4):
    """
    Calculate the correlation coefficient from covariance and variances.
    
    :param covariance: Covariance between two variables
    :param variance_x3: Variance of the first variable (x3)
    :param variance_x4: Variance of the second variable (x4)
    :return: Correlation coefficient
    """
    import math
    std_dev_x3 = math.sqrt(variance_x3)
    std_dev_x4 = math.sqrt(variance_x4)
    correlation_coefficient = covariance / (std_dev_x3 * std_dev_x4)
    return correlation_coefficient

# Example usage:
covariance = 9852
variance_x3 = 196
variance_x4 = 648025

correlation = calculate_correlation(covariance, variance_x3, variance_x4)
print(f"The correlation coefficient is: {correlation:.2f}")
