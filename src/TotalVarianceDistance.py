import numpy as np
import ot

def estimate_tv_distance_hist(samples1, samples2, bins=50):
    samples1, samples2 = np.asarray(samples1), np.asarray(samples2)
    n_dims = samples1.shape[1]
    if n_dims == 1:
        hist1, edges = np.histogram(samples1, bins=bins, density=True)
        hist2, _ = np.histogram(samples2, bins=edges, density=True)
    else:
        hist_range = [[min(np.min(samples1[:, i]), np.min(samples2[:, i])), 
                       max(np.max(samples1[:, i]), np.max(samples2[:, i]))] 
                      for i in range(n_dims)]
        hist1, edges = np.histogramdd(samples1, bins=bins, range=hist_range, density=True)
        hist2, _ = np.histogramdd(samples2, bins=bins, range=hist_range, density=True)
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
    return tv_distance

def OptimalTV(sample1, sample2):
    a, b = np.ones((len(sample1),)) / len(sample1), np.ones((len(sample2),)) / len(sample2)
    W = ot.emd2(a, b, ot.dist(samples1, samples2))
    return W

samples1 = np.random.normal(0, 1, (1000, 2))
samples2 = np.random.normal(1, 1, (1500, 2))

W = OptimalTV(samples1, samples2)
print(f"Wasserstein Distance: {W:.4f}")
tv = estimate_tv_distance_hist(samples1, samples2, bins=50)
print(f"TV Distance: {tv:.4f}")
