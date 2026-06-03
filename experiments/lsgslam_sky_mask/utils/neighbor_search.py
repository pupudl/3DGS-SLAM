import torch

try:
    import faiss
    import faiss.contrib.torch_utils
except ImportError:
    faiss = None


def torch_3d_knn(pts, num_knn, method="l2"):
    if faiss is None:
        if method != "l2":
            raise NotImplementedError("Fallback KNN only supports L2 distance.")
        distances = torch.cdist(pts, pts)
        distances, indices = torch.topk(distances, k=num_knn, dim=1, largest=False)
        return distances, indices

    # Initialize FAISS index
    if method == "l2":
        index = faiss.IndexFlatL2(pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    # Convert FAISS index to GPU
    if pts.get_device() != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add points to index and compute distances
    index.add(pts)
    distances, indices = index.search(pts, num_knn)
    return distances, indices


def torch_knn_search(query_pts, ref_pts, num_knn=1, method="l2"):
    if faiss is None:
        if method != "l2":
            raise NotImplementedError("Fallback KNN search only supports L2 distance.")
        ref_chunk_size = 4096
        best_distances = None
        best_indices = None
        for ref_start in range(0, ref_pts.shape[0], ref_chunk_size):
            ref_end = min(ref_start + ref_chunk_size, ref_pts.shape[0])
            ref_chunk = ref_pts[ref_start:ref_end]
            chunk_distances = torch.cdist(query_pts, ref_chunk)
            chunk_distances, chunk_indices = torch.topk(
                chunk_distances,
                k=min(num_knn, ref_chunk.shape[0]),
                dim=1,
                largest=False,
            )
            chunk_indices = chunk_indices + ref_start
            if best_distances is None:
                best_distances = chunk_distances
                best_indices = chunk_indices
            else:
                merged_distances = torch.cat((best_distances, chunk_distances), dim=1)
                merged_indices = torch.cat((best_indices, chunk_indices), dim=1)
                best_distances, best_order = torch.topk(
                    merged_distances, k=num_knn, dim=1, largest=False
                )
                best_indices = torch.gather(merged_indices, 1, best_order)
        return best_distances, best_indices

    # Initialize FAISS index on the reference points, then query with arbitrary points.
    if method == "l2":
        index = faiss.IndexFlatL2(ref_pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(ref_pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    if ref_pts.get_device() != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(ref_pts.contiguous())
    distances, indices = index.search(query_pts.contiguous(), num_knn)
    return distances, indices
    

def calculate_neighbors(params, variables, time_idx, num_knn=20):
    if time_idx is None:
        pts = params['means3D'].detach()
    else:
        pts = params['means3D'][:, :, time_idx].detach()
    neighbor_dist, neighbor_indices = torch_3d_knn(pts.contiguous(), num_knn)
    neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
    variables["neighbor_indices"] = neighbor_indices.long().contiguous()
    variables["neighbor_weight"] = neighbor_weight.float().contiguous()
    variables["neighbor_dist"] = neighbor_dist.float().contiguous()
    return variables
