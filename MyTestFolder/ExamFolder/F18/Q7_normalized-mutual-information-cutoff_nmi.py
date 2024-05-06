from sklearn.metrics.cluster import normalized_mutual_info_score # type: ignore
import numpy as np # type: ignore

# TODO: CHECK WHICH OBSERVATIONS ARE BEING CLUSTERED BASED ON THE CUTOFF --> THEIR CORRESPONDING OBSERVATION WILL BE ASSIGNED 0, 1 or 2.
# Example cluster assignments based on the dendrogram cutoff
your_clusters = [0, 2, 0, 1, 0, 1, 0, 0, 0, 0]  # Each index corresponds to O1 through O10, respectively

# TODO: LOOK AT THE TABLE FOR WHICH CLASSES EACH OBSERVATION BELONGS TO ACCORDING TO THE COLOR
# Ground truth from Table 2 Look at the split of the colors
ground_truth = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]  # Corresponding to O1 to O10, using your provided labels

# Calculate Normalized Mutual Information
# Ensuring we're using 'arithmetic' average method, which is typical but check documentation for your specific version/environment
nmi = normalized_mutual_info_score(ground_truth, your_clusters, average_method='arithmetic')

print("Normalized Mutual Information (NMI):", nmi)
