import scipy.integrate as integrate

# TODO MAKE SURE TO CHANGE THE PIECEWISE FUNCTIONS AND LOOK AT THE P(X) VALUES FROM THE PLOT --> ALSO ADD MORE OR REMOVE LESS IF NEEDED

# Define the piecewise functions
def p1(x):
    return x * 0.6

def p2(x):
    return x * 1.0

def p3(x):
    return x * 1.6

# Integrate each piece
# TODO MAKE SURE TO CHANGE THE INTERVALS TO MATCH THE PIECEWISE FUNCTION
integral1, _ = integrate.quad(p1, 0, 0.2)
integral2, _ = integrate.quad(p2, 0.2, 0.6)
integral3, _ = integrate.quad(p3, 0.6, 0.9)

# Sum the integrals to get the expected value
expected_value = integral1 + integral2 + integral3

print(f"Expected value E[x]: {expected_value:.3f}")
