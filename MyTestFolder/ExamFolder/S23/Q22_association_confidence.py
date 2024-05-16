from fractions import Fraction
# Given values
support_X = 3 / 5
support_Y = 8 / 15
confidence_X_to_Y = 1 / 6

# Calculate the support of X ∪ Y using the confidence of X → Y
support_X_union_Y = confidence_X_to_Y * support_X

# Calculate the confidence of Y → X
confidence_Y_to_X = support_X_union_Y / support_Y
confidence_fraction = Fraction(confidence_Y_to_X).limit_denominator() 

print(f"Support of X: {support_X}")
print(f"Support of Y: {support_Y}")
print(f"Support of X ∪ Y: {support_X_union_Y}")
print(f"Confidence of X → Y: {confidence_X_to_Y}")
print(f"Confidence of Y → X: {confidence_Y_to_X}")
print(f"Confidence of Y → X as a fraction: {confidence_fraction}")
