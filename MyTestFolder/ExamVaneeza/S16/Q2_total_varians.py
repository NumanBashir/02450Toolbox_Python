import numpy as np

#Denne script beregner totale varians af alle komponenter 
#Man kan bruge det til bagefter at tjekke udsagn igennem og beregne ud fra disse komponenter. 

def calculate_variance_percentages(singular_values):
    # Beregn den totale varians som summen af kvadraterne af singularværdierne
    total_variance = np.sum(np.square(singular_values))
    
    # Beregn variansprocenten for hver komponent
    variance_percentages = [(value**2 / total_variance) * 100 for value in singular_values]
    
    return variance_percentages

# Eksempel på matrix med singularværdierne
singular_values = np.array([149, 118, 53, 42, 3])

# Beregn variansprocenterne
variance_percentages = calculate_variance_percentages(singular_values)

# Udskriv resultaterne
for i, percentage in enumerate(variance_percentages, 1):
    print(f"Komponent {i}: {percentage:.2f}% varians")
