import numpy as np
import ot
from sklearn.decomposition import PCA
def estimate_tv_distance_hist(samples1, samples2, bins=10, n_components = 5):
    samples1, samples2 = np.asarray(samples1), np.asarray(samples2)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    samples1_reduced = pca.fit_transform(samples1)
    samples2_reduced = pca.transform(samples2)
    
    # Define histogram range
    hist_range = [[min(np.min(samples1_reduced[:, i]), np.min(samples2_reduced[:, i])), 
                   max(np.max(samples1_reduced[:, i]), np.max(samples2_reduced[:, i]))] 
                  for i in range(n_components)]
    
    # Compute histograms
    hist1, edges = np.histogramdd(samples1_reduced, bins=bins, range=hist_range, density=True)
    hist2, _ = np.histogramdd(samples2_reduced, bins=bins, range=hist_range, density=True)
    
    # Compute TV distance
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
    return tv_distance

def OptimalTV(sample1, sample2):
    a, b = np.ones((len(sample1),)) / len(sample1), np.ones((len(sample2),)) / len(sample2)
    W = ot.emd2(a, b, ot.dist(sample1, sample2))
    return W


