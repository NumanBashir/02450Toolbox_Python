import numpy as np

def calculate_misclassified(error_rate, num_samples):
    return int(error_rate * num_samples)

def adaboost_weight_update(error_rate, num_samples, num_misclassified):
    # Beregn alpha_t
    alpha_t = 0.5 * np.log((1 - error_rate) / error_rate)
    
    # Initial vægt for hver observation
    initial_weight = 1 / num_samples
    
    # Vægt for forkert klassificerede observationer
    weight_misclassified = initial_weight * np.exp(alpha_t)
    
    # Vægt for korrekt klassificerede observationer
    weight_correct = initial_weight * np.exp(-alpha_t)
    
    # Samlet vægt for korrekt klassificerede observationer
    total_weight_correct = (num_samples - num_misclassified) * weight_correct
    
    # Samlet vægt for forkert klassificerede observationer
    total_weight_misclassified = num_misclassified * weight_misclassified
    
    # Total vægt
    total_weight = total_weight_correct + total_weight_misclassified
    
    # Normaliseret vægt for korrekt klassificerede observationer
    normalized_weight_correct = weight_correct / total_weight
    
    return normalized_weight_correct

# Eksempelværdier
error_rate = 1 / 16 #ændre
num_samples = 32 #ændre, dette er observationer i alt.

# Beregn antallet af forkert klassificerede observationer
num_misclassified = calculate_misclassified(error_rate, num_samples)

# Beregn den opdaterede vægt for korrekt klassificerede observationer
updated_weight_correct = adaboost_weight_update(error_rate, num_samples, num_misclassified)

print(f"Opdateret vægt for korrekt klassificerede observationer: {updated_weight_correct:.4f}")
