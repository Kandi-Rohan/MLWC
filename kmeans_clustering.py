import numpy as np


def compute_ncc(hk, hk_prime):
    """
    Computes Normalized Channel Correlation (NCC) between two channel vectors.
    hk, hk_prime: Complex channel vectors of shape (M, 1) or (M,)
    """
    hk = hk.flatten()
    hk_prime = hk_prime.flatten()
    
    numerator = np.abs(np.vdot(hk, hk_prime))
    denominator = np.linalg.norm(hk) * np.linalg.norm(hk_prime)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def k_means_plus_plus_cu_grouping(H_CUs, num_groups):
    """
    Stage 1: K-means++ algorithm for CU grouping based on NCC.
    
    Args:
        H_CUs: Complex channel matrix of all CUs. Shape (K, M) where K is number of CUs.
        num_groups (int): I (number of groups)
        
    Returns:
        clusters: A list of lists, where each sublist contains the indices of CUs in that group.
        centers: A list of central CU indices for each group.
    """
    K = H_CUs.shape[0]
    if num_groups > K:
        raise ValueError("Number of groups cannot exceed number of CUs.")
        
    # Precompute pairwise NCC
    ncc_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                ncc_matrix[i, j] = compute_ncc(H_CUs[i], H_CUs[j])
            else:
                ncc_matrix[i, j] = 1.0
                
    # Step 3: Calculate pairwise NCC of all CUs and select the pair of CUs
    # with the smallest NCC value as the centers of the first and second groups.
    if num_groups >= 2:
        # Find the pair with smallest NCC
        min_ncc = np.inf
        center_1, center_2 = -1, -1
        for i in range(K):
            for j in range(i+1, K):
                if ncc_matrix[i, j] < min_ncc:
                    min_ncc = ncc_matrix[i, j]
                    center_1 = i
                    center_2 = j
        centers = [center_1, center_2]
    else:
        centers = [0]
        
    # Then, select the CU with the smallest sum of NCC with all center CUs 
    # from the remaining CUs as the center CU of the m-th group.
    remaining_cus = set(range(K)) - set(centers)
    while len(centers) < num_groups:
        min_sum_ncc = np.inf
        next_center = -1
        for cu in remaining_cus:
            sum_ncc = sum([ncc_matrix[cu, c] for c in centers])
            if sum_ncc < min_sum_ncc:
                min_sum_ncc = sum_ncc
                next_center = cu
        centers.append(next_center)
        remaining_cus.remove(next_center)
        
    # After obtaining the central CU set, each non-central CU k is assigned
    # to the group of the central CU with the highest correlation
    clusters = {c: [c] for c in centers}
    for cu in remaining_cus:
        best_center = -1
        max_corr = -1
        for c in centers:
            if ncc_matrix[cu, c] > max_corr:
                max_corr = ncc_matrix[cu, c]
                best_center = c
        clusters[best_center].append(cu)
        
    return list(clusters.values()), centers

if __name__ == "__main__":
    # Quick test
    K = 6
    M = 36
    H = np.random.randn(K, M) + 1j * np.random.randn(K, M)
    num_groups = 3
    clusters, centers = k_means_plus_plus_cu_grouping(H, num_groups)
    print("Centers:", centers)
    print("Clusters:", clusters)
