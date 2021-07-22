# gap.py
# (c) 2013 Mikael Vejdemo-Johansson for the code structure
# (c) 2021 Maria Korosteleva for PyTorch adaptation and the rest of the changes
# BSD License
#
# PyTorch-based function to compute the gap statistic and standard error for evaluating k-means clustering.
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423  (http://web.stanford.edu/~hastie/Papers/gap.pdf)


import torch 
from kmeans_pytorch import kmeans  # https://github.com/subhadarship/kmeans_pytorch


def gaps(data, nrefs=20, max_k=10, extra_sencitivity_threshold=0.0, logs=False):
    """
        Find the optimal number of clusters in n x m dataset in data (presented as torch.Tensor) based on
        Gap statistic

        * State the number *nrefs* of reference distributions for automatic generation with a
        uniformed distribution within the bounding box of data.
        * Give the list of k-values for which you want to compute the statistic in ks.
        * 'extra_sencitivity_threshold' reduce sencitivity for improvement of adding more classes 
            (creates additional bias towards smaller number of classes)

        Returns: 
            optimal class_numbers, 
            how much 1-class scenario is worse then the optimal, 
            labels for optimal class, 
            cluster_centers for optimal class

        Devices note: all computations are performed on the device where the *data* is located. 
        Both CPU and GPU are supported.
    """
    shape = data.shape

    max_k = min(max_k, len(data))  # cannot be more then number of datapoints

    # reference distributions
    rands = _reference_distributions(data, nrefs)
    if rands is None:  # degenerate case, no need for further processing
        labels, cluster_center = _single_cluster_kmeans(data)
        return 1, 0.0, labels, [cluster_center]

    # Optimal class number
    gaps = [None] * (max_k + 1)  # list(s allow for None values
    std_errors = [None] * (max_k + 1)
    labels_per_k = []
    ccs_per_k = []
    for k in range(1, max_k + 1):   
        # Step 1 -- clustering
        # on the data
        if k == 1:
            labels, cluster_centers = _single_cluster_kmeans(data)
        else:
            labels, cluster_centers = kmeans(X=data, num_clusters=k, device=data.device, tqdm_flag=False)
            labels, cluster_centers = labels.to(data.device), cluster_centers.to(data.device)
        labels_per_k.append(labels)  # save labels to return
        ccs_per_k.append(cluster_centers)

        disp = sum([torch.dist(data[m], cluster_centers[labels[m]]) for m in range(shape[0])])

        # on the reference distributions
        refdisps = torch.zeros(nrefs, device=data.device)
        for j in range(nrefs):
            if k == 1:
                labels, cluster_centers = _single_cluster_kmeans(rands[j])
            else:
                labels, cluster_centers = kmeans(X=rands[j], num_clusters=k, device=rands.device, tqdm_flag=False)
                labels, cluster_centers = labels.to(rands.device), cluster_centers.to(rands.device)

            refdisps[j] = sum([torch.dist(rands[j, m], cluster_centers[labels[m]]) for m in range(shape[0])])

        # Step 2 -- gaps
        # flipped mean & log https://gist.github.com/michiexile/5635273#gistcomment-2324237
        reflogs = torch.log(refdisps)
        refmean = torch.mean(reflogs)
        gaps[k] = refmean - torch.log(disp)

        # Step 3 -- standard errors
        std_errors[k] = torch.sqrt(torch.mean((reflogs - refmean) ** 2) * (1 + 1. / nrefs))
        std_errors[k] += extra_sencitivity_threshold  # with some adjustment

        # Check optimality criteria 
        if k > 1 and gaps[k - 1] >= gaps[k] - std_errors[k]:
            # optimal class found!
            return k - 1, max(-1 * (gaps[1] - (gaps[k - 1] - std_errors[k - 1])), 0.), labels_per_k[-2], ccs_per_k[-2]

    if logs:
        print('Gaps::Warning::Optimal K not found, returning the last one, for gaps {}'.format(gaps))
    return max_k, max(-1 * (gaps[1] - (gaps[max_k] - std_errors[max_k])), 0.), labels_per_k[-1], ccs_per_k[-1]


def optimal_clusters(data, max_k=10, sencitivity_threshold=0.1, logs=False):
    """
        Find the optimal number of clusters in n x m dataset in data (presented as torch.Tensor) based on
        cluster compactness. Optimal K is the minimum K wich wich cluster member are `sencitivity_threshold`
        away from the center on average

        * Give the list of k-values for which you want to compute the statistic in ks.
        * 'sencitivity_threshold' -- Target compactness

        Returns: 
            optimal class_numbers, 
            optimal compactness, 
            labels for optimal class, 
            cluster_centers for optimal class

        Devices note: all computations are performed on the device where the *data* is located. 
        Both CPU and GPU are supported.
    """
    shape = data.shape

    max_k = min(max_k, len(data))  # cannot be more then number of datapoints

    # Optimal class number
    labels_per_k = []
    ccs_per_k = []
    disp = 0.0
    for k in range(1, max_k + 1):   
        # Step 1 -- clustering
        # on the data
        if k == 1:
            labels, cluster_centers = _single_cluster_kmeans(data)
        else:
            labels, cluster_centers = kmeans(X=data, num_clusters=k, device=data.device, tqdm_flag=False)
            labels, cluster_centers = labels.to(data.device), cluster_centers.to(data.device)
        labels_per_k.append(labels)  # save labels to return
        ccs_per_k.append(cluster_centers)

        # average distance to cluster centers
        # TODO with log?
        disp = torch.mean([torch.dist(data[m], cluster_centers[labels[m]]) for m in range(shape[0])])

        if logs:
            print(f'Distance to cluster centers {disp} for k={k}')

        if disp <= sencitivity_threshold:
            # optimal class found! Clustering is compact enough
            return k, disp, labels_per_k[-1], ccs_per_k[-1]

    if logs:
        print('Optimal_clusters::Warning::Optimal K not found, returning the last one, for disp {}'.format(disp))
    return max_k, disp, labels_per_k[-1], ccs_per_k[-1]



def _single_cluster_kmeans(data):
    """
        Evaluate cluster center and give the set of labels in KMeans format 
        when the requested K=1
    """
    cluster_center = data.mean(dim=0).unsqueeze(0)
    labels = torch.zeros(data.shape[0], dtype=torch.int, device=data.device)

    return labels, cluster_center


def _reference_distributions(data, nrefs):
    """
        Generate (random) reference distributions for given dataset
    """

    # refs
    tops, _ = data.max(dim=0)
    bots, _ = data.min(dim=0)
    dists = tops - bots 

    if torch.allclose(tops, bots, atol=0.01):  # degenerate case, no need for further processing
        return None

    # uniform distribution
    rands = torch.rand((nrefs, data.shape[0], data.shape[1]), device=data.device)
    rands = rands * dists + bots

    return rands
