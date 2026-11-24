# 从 CrystalFlow 的 CSPNet 抽取的按需建图逻辑，供 CSPNet 与 MatformerCSPNet 共用。
# 仅把原来的实例方法改成显式参数函数，计算逻辑保持不变。

import torch
from torch_geometric.utils import dense_to_sparse


def select_symmetric_edges(tensor, mask, reorder_idx, inverse_neg):
    """选出单向边，拼接反向边，再按重排索引整理。"""
    tensor_directed = tensor[mask]
    sign = 1 - 2 * inverse_neg
    tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
    return tensor_cat[reorder_idx]


def reorder_symmetric_edges(edge_index, cell_offsets, neighbors, edge_vector):
    """
    把边整理成“每个晶体、每条无向边及其反向边连续排列”的形式。
    只保留 i<j 的单向边，再补上反向边，便于后续按晶体聚合。
    """
    # 只有 kNN 建图会用到，按需导入以避免 fc 路径依赖 data_utils/pymatgen
    from matformer_csp.common.data_utils import repeat_blocks

    mask_sep_atoms = edge_index[0] < edge_index[1]
    cell_earlier = (
        (cell_offsets[:, 0] < 0)
        | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
        | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
    )
    mask_same_atoms = edge_index[0] == edge_index[1]
    mask_same_atoms &= cell_earlier
    mask = mask_sep_atoms | mask_same_atoms

    edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)
    edge_index_cat = torch.cat(
        [edge_index_new, torch.stack([edge_index_new[1], edge_index_new[0]], dim=0)],
        dim=1,
    )

    batch_edge = torch.repeat_interleave(
        torch.arange(neighbors.size(0), device=edge_index.device), neighbors,
    )
    batch_edge = batch_edge[mask]
    neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

    edge_reorder_idx = repeat_blocks(
        neighbors_new // 2,
        repeats=2,
        continuous_indexing=True,
        repeat_inc=edge_index_new.size(1),
    )

    edge_index_new = edge_index_cat[:, edge_reorder_idx]
    cell_offsets_new = select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
    edge_vector_new = select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

    return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new


def gen_edges(edge_style, cutoff, max_neighbors, num_atoms, frac_coords, lattices, node2graph):
    """
    根据当前晶格与分数坐标现场建边。

    返回:
        edge_index: (2, E)，edge_index[0] 为邻居，edge_index[1] 为中心原子
        frac_diff:  (E, 3)，从中心原子指向邻居的分数坐标差（含周期像）
    """
    if edge_style == "fc":
        # 每个晶体内部全连接
        lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

    elif edge_style in ("knn", "knn_cart"):
        from matformer_csp.common.data_utils import radius_graph_pbc

        lattice_nodes = lattices[node2graph]
        cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords,
            None,
            None,
            num_atoms,
            cutoff,
            max_neighbors,
            device=num_atoms.device,
            lattices=lattices,
        )

        j_index, i_index = edge_index
        distance_vectors = frac_coords[j_index] - frac_coords[i_index]
        distance_vectors = distance_vectors + to_jimages.float()

        edge_index_new, _, _, edge_vector_new = reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, distance_vectors
        )
        return edge_index_new, -edge_vector_new

    elif edge_style == "knn_frac":
        from matformer_csp.common.data_utils import get_max_neighbors_mask

        # 在分数坐标空间做 kNN，不需要真实的笛卡尔距离
        num_atoms_per_image = num_atoms
        num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()
        index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

        index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
        num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)

        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
        index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
        atom_count_sqr = torch.arange(num_atom_pairs, device=num_atoms.device) - index_sqr_offset

        index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + index_offset_expand
        index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand

        frac_pos1 = torch.index_select(frac_coords, 0, index1)
        frac_pos2 = torch.index_select(frac_coords, 0, index2)
        frac_dist_mic_sqr = torch.sum(((frac_pos1 - frac_pos2 - 0.5) % 1 - 0.5) ** 2, dim=1).view(-1)

        mask_num_neighbors, num_bonds = get_max_neighbors_mask(
            natoms=num_atoms,
            index=index1,
            atom_distance=frac_dist_mic_sqr,
            max_num_neighbors_threshold=max_neighbors,
        )
        if not torch.all(mask_num_neighbors):
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)

        edge_index = torch.stack((index2, index1))
        j_index, i_index = edge_index
        edge_vector = (frac_coords[j_index] - frac_coords[i_index]) % 1
        return edge_index, edge_vector

    else:
        raise ValueError(f"Unknown edge style: {edge_style}")
