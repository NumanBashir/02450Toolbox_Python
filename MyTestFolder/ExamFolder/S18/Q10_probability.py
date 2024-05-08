def calculate_probability():
    # Given probabilities converted from percentages to decimals
    P_D_C = 0.00000271 / 100  # probability of dying by car
    P_D_B = 0.0000004 / 100   # probability of dying by bus
    P_D_P = 0.0000003 / 100   # probability of dying by plane
    P_C = 30 / 100            # probability of choosing car
    P_B = 10 / 100            # probability of choosing bus
    P_P = 60 / 100            # probability of choosing plane

    # Calculate the total probability of dying P(D)
    P_D = (P_D_P * P_P) + (P_D_B * P_B) + (P_D_C * P_C)

    # Calculate the probability of having chosen the plane given the person died P(P|D)
    P_P_D = (P_D_P * P_P) / P_D

    return P_P_D * 100  # convert to percentage for final answer

# Call the function and print the result
result = calculate_probability()
print(f"The probability of having chosen the plane given the person died is {result:.4f}%")
