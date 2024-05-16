import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_covariance_matrix(mu, Sigma):
    """
    Analyserer kovariansmatricen og bestemmer om der er positiv eller negativ kovarians,
    samt spredningen langs PCA1 og PCA2.

    Args:
    mu: Middelværdier (mean vector)
    Sigma: Kovariansmatrice (covariance matrix)
    """
    # Udtræk varians og kovarians værdier
    var_PCA1 = Sigma[0, 0]
    var_PCA2 = Sigma[1, 1]
    cov_PCA1_PCA2 = Sigma[0, 1]
    
    # Bestem kovarians
    if cov_PCA1_PCA2 > 0:
        covariance_type = "Positive Covariance"
    elif cov_PCA1_PCA2 < 0:
        covariance_type = "Negative Covariance"
    else:
        covariance_type = "No Covariance"
    
    # Bestem spredning
    if var_PCA1 > var_PCA2:
        spread_direction = "More spread in PCA1 direction"
    elif var_PCA1 < var_PCA2:
        spread_direction = "More spread in PCA2 direction"
    else:
        spread_direction = "Equal spread in both directions"
    
    # Print resultater
    print(f"Mean (μ): {mu}")
    print(f"Covariance Matrix (Σ):\n{Sigma}")
    print(f"Covariance Type: {covariance_type}")
    print(f"Spread Direction: {spread_direction}")

def plot_covariance(mu, Sigma, title=''):
    """
    Plotter konturerne af den Gaussian fordeling givet middelværdi og kovariansmatrice.

    Args:
    mu: Middelværdier (mean vector)
    Sigma: Kovariansmatrice (covariance matrix)
    title: Titel for plottet (default: '')
    """
    # Generer datapunkter
    x, y = np.random.multivariate_normal(mu, Sigma, 5000).T

    # Plot konturer
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x, y=y, fill=True, cmap="viridis", bw_adjust=0.5)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter(mu[0], mu[1], color='red')  # Plot middelværdien
    plt.show()

# Eksempler på kovariansmatricer og middelværdier
    
    #Man kan afprøve de forskellige statements ved skifte alle ud :
mu1 = np.array([-1.129, 0.417])
Sigma1 = np.array([[1.307, 0.557], [0.557, 0.575]])

mu2 = np.array([-1.298, -1.261])
Sigma2 = np.array([[1.458, -1.982], [-1.982, 2.771]])

mu3 = np.array([1.957, 0.091])
Sigma3 = np.array([[0.248, -0.128], [-0.128, 0.101]])

# Analyser og plot hver kluster
analyze_covariance_matrix(mu1, Sigma1)
plot_covariance(mu1, Sigma1, title='Cluster 1')

analyze_covariance_matrix(mu2, Sigma2)
plot_covariance(mu2, Sigma2, title='Cluster 2')

analyze_covariance_matrix(mu3, Sigma3)
plot_covariance(mu3, Sigma3, title='Cluster 3')
