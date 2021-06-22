# gap.py
# (c) 2013 Mikael Vejdemo-Johansson  
# BSD License
#
# SciPy function to compute the gap statistic for evaluating k-means clustering.
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
# Available here: https://gist.github.com/michiexile/5635273

# Maria Korosteleva: Re-written for PyTorch allowing GPU-only operations
# PyTorch & GPU friendliness 
import torch 
from kmeans_pytorch import kmeans  # https://github.com/subhadarship/kmeans_pytorch


def gap_torch(data, refs=None, nrefs=20, ks=range(1, 11)):
    """
        Compute the Gap statistic for an n x m dataset in data (presented as torch.Tensor).

        Either give a precomputed set of reference distributions in *refs* as an (nrefs, n, m) torch tensor,
        or state the number *nrefs* of reference distributions for automatic generation with a
        uniformed distribution within the bounding box of data.

        Give the list of k-values for which you want to compute the statistic in ks.

        Devices note: all computations are performed on the device where the *data* is located. 
        Both CPU and GPU are supported.
    """
    shape = data.shape
    if refs is None:
        tops, _ = data.max(dim=0)
        bots, _ = data.min(dim=0)
        dists = tops - bots 

        # uniform distribution
        rands = torch.rand((nrefs, shape[0], shape[1]), device=data.device)
        rands = rands * dists + bots

    else:
        rands = refs

    gaps = [] * len(ks)  # lists allow for None values
    zero = torch.zeros(1, device=data.device)
    labels_per_k = []
    for (i, k) in enumerate(ks):   
        # on the data
        if k == 1:
            labels, cluster_centers = _single_cluster_kmeans(data)
        else:
            labels, cluster_centers = kmeans(X=data, num_clusters=k, device=data.device, tqdm_flag=False)
            labels, cluster_centers = labels.to(data.device), cluster_centers.to(data.device)
        labels_per_k.append(labels)  # save labels to return

        disp = sum([torch.dist(data[m], cluster_centers[labels[m]]) for m in range(shape[0])])

        if torch.isclose(disp, zero):   # degenerate case
            gaps[i] = None
            continue

        # on the reference distributions
        refdisps = torch.zeros(nrefs, device=data.device)
        for j in range(nrefs):
            if k == 1:
                labels, cluster_centers = _single_cluster_kmeans(rands[j])
            else:
                labels, cluster_centers = kmeans(X=rands[j], num_clusters=k, device=rands.device, tqdm_flag=False)
                labels, cluster_centers = labels.to(rands.device), cluster_centers.to(rands.device)

            refdisps[j] = sum([torch.dist(rands[j, m], cluster_centers[labels[m]]) for m in range(shape[0])])

            # TODO any sort of warning in degenerate cases here?

        # gap statistic
        if torch.isclose(disp, zero) or torch.allclose(refdisps, zero):  # degenerate cases
            gaps[i] = None
        else:
            # flipped mean & log https://gist.github.com/michiexile/5635273#gistcomment-2324237
            gaps[i] = torch.mean(torch.log(refdisps)) - torch.log(disp)

    return gaps, labels_per_k[-1] if len(labels_per_k) else None


def _single_cluster_kmeans(data):
    """
        Evaluate cluster center and give the set of labels in KMeans format 
        when the requested K=1
    """
    cluster_center = data.mean(dim=0).unsqueeze(0)
    labels = torch.zeros(data.shape[0], dtype=torch.int, device=data.device)

    return labels, cluster_center
