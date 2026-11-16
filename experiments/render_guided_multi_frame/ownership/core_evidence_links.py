"""Bounded geometry paths between independently supported motion-core points.

These links connect reliable candidate endpoints for core certification. They
do not turn the intermediate surface into foreground. Every path is verified
in ONE source camera, inside the endpoints' local image rectangle, using the
ordinary depth/distance edge gates and a bounded sum of 3D edge lengths.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_bipartite_matching
from scipy.spatial import cKDTree

from ownership.object_connectivity_constraints import _snapshot_edges


def _unique_source_bindings(snapshot):
    """Keep the full-source ambiguity decision before clipping a local ROI."""
    uv = np.asarray(snapshot['payload']['uv'], float)
    xyz = np.asarray(snapshot['payload']['xyz'], float)
    valid = np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    xy = np.zeros_like(uv, np.int64)
    xy[valid] = np.rint(uv[valid]).astype(np.int64)
    valid &= (xy >= 0).all(1)
    result = np.zeros(len(uv), bool)
    if not valid.any():
        return result
    observations = snapshot.get('render_observations') or {}
    if 'source' in observations:
        height, width = np.asarray(observations['source']).shape[:2]
    else:
        width, height = xy[valid].max(axis=0)+1
    valid &= (xy[:, 0] < width) & (xy[:, 1] < height)
    rows = np.flatnonzero(valid)
    pixels = xy[rows, 1]*width+xy[rows, 0]
    _, inverse, counts = np.unique(pixels, return_inverse=True, return_counts=True)
    result[rows] = counts[inverse] == 1
    return result


def _independent_tracks(rows, track_sets):
    tracks = sorted({track for row in rows for track in track_sets.get(int(row), ())})
    if not tracks:
        return 0
    columns = {track: c for c, track in enumerate(tracks)}
    left, right = [], []
    for r, row in enumerate(rows):
        for track in track_sets.get(int(row), ()):
            left.append(r)
            right.append(columns[track])
    graph = sparse.csr_matrix((np.ones(len(left), bool), (left, right)),
                              shape=(len(rows), len(tracks)))
    return int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type='column') >= 0))


def build_core_evidence_links(window, candidate_by_object, options=None, height_keep=None):
    """Return per-object endpoint links; input candidates must already be reliable.

    ``candidate_by_object`` contains indices into sorted ``window._active_gids``.
    The caller remains responsible for repeated motion evidence and its height
    gate. A rescued cluster additionally needs three distinct candidate/track
    pairs with positive measured motion evidence. One track cannot certify
    several endpoints. Local links are never inserted into other objects' graphs.
    """
    options = {} if options is None else dict(options.get('connectivity', options))
    settings = dict(options.get('core_evidence_links', {}))
    pixel_limit = float(settings.get('max_pixel_distance', 64.))
    padding = float(settings.get('bbox_padding_px', 4.))
    path_limit = float(settings.get('max_path_length_m', 4.))
    minimum_points = int(settings.get('min_points', options.get('core_min_points', 3)))
    minimum_tracks = int(settings.get('min_independent_tracks', 3))
    if (pixel_limit <= 0 or padding < 0 or path_limit <= 0 or
            minimum_points < 3 or minimum_tracks < 3):
        raise ValueError('Core evidence links require positive bounds and at least three independent points/tracks')
    gids = np.asarray(window._active_gids, np.int64)
    n = len(gids)
    if gids.ndim != 1 or np.any(gids[1:] <= gids[:-1]):
        raise ValueError('Core evidence links require sorted unique active Gaussian IDs')
    height_keep = np.ones(n, bool) if height_keep is None else np.asarray(height_keep, bool)
    if height_keep.shape != (n,):
        raise ValueError('Core evidence height gate must align with active Gaussian IDs')
    background = int(options.get('background_object_id', 0))
    snapshots = []
    for pair, snapshot in sorted(window._active.items()):
        local_gids = np.asarray(snapshot['gaussian_ids'], np.int64)
        rows = np.searchsorted(gids, local_gids)
        if np.any(rows >= n) or not np.array_equal(gids[rows], local_gids):
            raise ValueError('Core evidence snapshots must align with active Gaussian IDs')
        snapshots.append((pair, snapshot, rows))
    edges_by_object, groups_by_object, object_reports = {}, {}, {}
    binding_cache = {}
    for oid_value, raw in sorted(candidate_by_object.items()):
        oid = int(oid_value)
        candidates = np.unique(np.asarray(raw, np.int64))
        if candidates.ndim != 1 or np.any(candidates < 0) or np.any(candidates >= n):
            raise ValueError('Core evidence candidates must be valid active row indices')
        candidates = candidates[height_keep[candidates]]
        report = dict(candidate_count=int(len(candidates)), accepted_pairs=0,
                      tested_paths=0, qualified_groups=[], endpoint_links=0,
                      bounded_pairs=[], local_groups=[],
                      intermediate_rows_assigned=False, path_scope='single_source_camera_local_bbox')
        edges_by_object[oid] = np.empty((0, 2), np.int64)
        groups_by_object[oid] = []
        object_reports[str(oid)] = report
        if oid == background or len(candidates) < minimum_points:
            report['action'] = 'insufficient_reliable_candidates'
            continue
        candidate_gids = gids[candidates]
        visible, track_sets = [], {}
        for pair, snapshot, global_rows in snapshots:
            local_objects = np.asarray(snapshot['object_ids'])
            if oid not in local_objects or background not in local_objects:
                continue
            local = np.flatnonzero(np.isin(snapshot['gaussian_ids'], candidate_gids))
            if not len(local):
                continue
            b = int(np.flatnonzero(local_objects == background)[0])
            c = int(np.flatnonzero(local_objects == oid)[0])
            costs = np.asarray(snapshot['costs'])
            weights = np.asarray(snapshot['weights'])
            if weights.ndim == 1:
                reliability = weights[local]
            else:
                reliability = np.minimum(weights[local, b], weights[local, c])
            tracks = np.asarray(snapshot.get('track_ids', np.full(len(global_rows), -1)))
            good = (np.isfinite(costs[local, b]) & np.isfinite(costs[local, c]) &
                    (costs[local, c] < float(options.get('core_cost_max', .5))) &
                    (costs[local, b]-costs[local, c] >= float(options.get('core_advantage', .3))) &
                    (reliability >= float(options.get('core_min_weight', .15))))
            for row in local[good & (tracks[local] >= 0)]:
                track_sets.setdefault(int(global_rows[row]), set()).add(int(tracks[row]))
            if len(local) >= 2:
                visible.append((pair, snapshot, global_rows, local))

        verified_edges, accepted_edges, groups = [], [], []
        for pair, snapshot, global_rows, local in visible:
            payload = snapshot['payload']
            uv = np.asarray(payload['uv'], float)
            xyz = np.asarray(payload['xyz'], float)
            finite = np.isfinite(uv[local]).all(1) & np.isfinite(xyz[local]).all(1)
            local = local[finite]
            if len(local) < 2:
                continue
            if id(snapshot) not in binding_cache:
                binding_cache[id(snapshot)] = _unique_source_bindings(snapshot)
            unique_binding = binding_cache[id(snapshot)]
            pairs = sorted(cKDTree(uv[local]).query_pairs(pixel_limit))
            local_edges, path_details = [], []
            for a, b in pairs:
                left, right = int(local[a]), int(local[b])
                if np.linalg.norm(xyz[left]-xyz[right]) > path_limit:
                    continue
                lower = np.minimum(uv[left], uv[right])-padding
                upper = np.maximum(uv[left], uv[right])+padding
                roi = np.flatnonzero((uv[:, 0] >= lower[0]) & (uv[:, 0] <= upper[0]) &
                                     (uv[:, 1] >= lower[1]) & (uv[:, 1] <= upper[1]) &
                                     height_keep[global_rows] & unique_binding)
                positions = np.searchsorted(roi, [left, right])
                if np.any(positions >= len(roi)) or not np.array_equal(roi[positions], [left, right]):
                    continue
                cropped = dict(payload={key: np.asarray(payload[key])[roi]
                                        for key in ('uv', 'xyz', 'color')},
                               render_observations=snapshot.get('render_observations'))
                edges, _, _ = _snapshot_edges(cropped, np.arange(len(roi)), options)
                report['tested_paths'] += 1
                if not len(edges):
                    continue
                lengths = np.linalg.norm(xyz[roi[edges[:, 0]]]-xyz[roi[edges[:, 1]]], axis=1)
                graph = sparse.csr_matrix((np.r_[np.maximum(lengths, 1e-8), np.maximum(lengths, 1e-8)],
                                          (np.r_[edges[:, 0], edges[:, 1]],
                                           np.r_[edges[:, 1], edges[:, 0]])),
                                         shape=(len(roi), len(roi)))
                distance = dijkstra(graph, directed=False, indices=int(positions[0]),
                                    limit=path_limit)[int(positions[1])]
                if not np.isfinite(distance):
                    continue
                local_edges.append((a, b))
                path_details.append(dict(first=int(pair[0]), second=int(pair[1]),
                                         gids=gids[global_rows[[left, right]]].astype(int).tolist(),
                                         pixel_distance=float(np.linalg.norm(uv[left]-uv[right])),
                                         path_length_m=float(distance), bbox=[lower.tolist(), upper.tolist()]))
            if not local_edges:
                continue
            report['bounded_pairs'].extend(path_details)
            local_edges = np.asarray(local_edges, np.int64)
            verified_edges.extend(np.sort(global_rows[local[local_edges]], axis=1).tolist())
            graph = sparse.csr_matrix((np.ones(2*len(local_edges)),
                                      (np.r_[local_edges[:, 0], local_edges[:, 1]],
                                       np.r_[local_edges[:, 1], local_edges[:, 0]])),
                                     shape=(len(local), len(local)))
            _, components = connected_components(graph, directed=False)
            for component in np.unique(components):
                selected = np.flatnonzero(components == component)
                cluster = global_rows[local[selected]]
                independent = _independent_tracks(cluster, track_sets)
                report['local_groups'].append(dict(
                    first=int(pair[0]), gids=gids[cluster].astype(int).tolist(),
                    independent_tracks=independent,
                    positive_track_ids={str(int(gids[row])): sorted(track_sets.get(int(row), ()))
                                        for row in cluster}))
        if verified_edges:
            # Each edge was verified in one source camera. Stable GIDs may
            # combine such evidence across cameras, just like the ordinary
            # window graph; coordinates themselves are never concatenated.
            verified_edges = np.unique(np.asarray(verified_edges, np.int64), axis=0)
            indexed = np.searchsorted(candidates, verified_edges)
            graph = sparse.csr_matrix((np.ones(2*len(indexed)),
                                      (np.r_[indexed[:, 0], indexed[:, 1]],
                                       np.r_[indexed[:, 1], indexed[:, 0]])),
                                     shape=(len(candidates), len(candidates)))
            _, components = connected_components(graph, directed=False)
            for component in np.unique(components):
                cluster = candidates[components == component]
                independent = _independent_tracks(cluster, track_sets)
                if len(cluster) < minimum_points or independent < minimum_tracks:
                    continue
                # A common source view must witness the WHOLE group within
                # the local pixel bound. A long chain of individually short
                # links cannot gradually sweep in a scene-wide background.
                witness = None
                for pair, snapshot, global_rows, local in visible:
                    selected = local[np.isin(global_rows[local], cluster)]
                    if len(selected) != len(cluster):
                        continue
                    positions = np.asarray(snapshot['payload']['uv'])[selected]
                    diameter = float(np.linalg.norm(positions[:, None]-positions[None], axis=2).max())
                    if np.isfinite(diameter) and diameter <= pixel_limit:
                        witness = (pair, diameter)
                        break
                if witness is None:
                    continue
                keep = np.isin(verified_edges[:, 0], cluster) & np.isin(verified_edges[:, 1], cluster)
                retained = verified_edges[keep]
                accepted_edges.extend(retained.tolist())
                groups.append(cluster.copy())
                retained_details = [detail for detail in report['bounded_pairs']
                                    if np.all(np.isin(detail['gids'], gids[cluster]))]
                pair, diameter = witness
                report['qualified_groups'].append(dict(
                    witness_first=int(pair[0]), witness_second=int(pair[1]),
                    gids=gids[cluster].astype(int).tolist(), independent_tracks=independent,
                    pixel_diameter=diameter, paths=retained_details))
                report['accepted_pairs'] += len(retained)
        if accepted_edges:
            edges_by_object[oid] = np.unique(np.asarray(accepted_edges, np.int64), axis=0)
            groups_by_object[oid] = groups
            report.update(action='bounded_core_endpoint_links', endpoint_links=len(edges_by_object[oid]))
        else:
            report['action'] = 'no_independent_bounded_core_group'
    return dict(edges_by_object=edges_by_object, groups_by_object=groups_by_object,
                report=dict(max_pixel_distance=pixel_limit, bbox_padding_px=padding,
                            max_path_length_m=path_limit, min_independent_tracks=minimum_tracks,
                            objects=object_reports, intermediate_rows_assigned=False))
