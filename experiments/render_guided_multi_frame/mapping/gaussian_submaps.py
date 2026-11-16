"""Fixed image-count submaps with one shared boundary image.

Only the current submap owns resident map geometry. Archived submaps keep their
own local map IDs and share the global camera/world and object canonical frames.
Crossing a boundary does not reset frontend tracks or object lifecycles.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from mapping.gaussian_map_backend import GaussianMapBackend


def partition_submaps(frame_ids, frames=50, overlap_frames=1):
    """Return strict image windows; the final window can be shorter.

    An exact boundary never creates a trailing singleton. ``frames`` counts
    sampled images, independently of their numeric frame-ID spacing.
    """
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 2:
        raise ValueError('Submap frames must be an integer of at least 2')
    if overlap_frames != 1 or isinstance(overlap_frames, bool):
        raise ValueError('Fixed submaps require exactly one overlapping image')
    raw = np.asarray(frame_ids)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('Submap frame IDs must be a one-dimensional integer sequence')
    if raw.size and (np.any(raw < 0) or np.any(np.diff(raw) <= 0)):
        raise ValueError('Submap frame IDs must be nonnegative and strictly increasing')
    ids = list(map(int, raw))
    result = []
    start = 0
    while start < len(ids):
        end = min(start + int(frames), len(ids))
        result.append(dict(submap_id=len(result), start_index=start, end_index=end-1,
                           frame_ids=ids[start:end], image_count=end-start,
                           partial=end-start < frames))
        if end == len(ids):
            break
        start = end-1
    return result


def _json_copy(value):
    return json.loads(json.dumps(value))


class SubmapGaussianMapBackend:
    """Delegate consecutive pairs to one fixed-size map at a time.

    Per-submap point capacity is raised to a conservative bound for all source
    insertions and topology additions, so capacity never shortens a submap.
    The only geometry passed across a boundary is a bounded set of currently
    tracked sparse anchors; images and archived dense map blocks are released.
    """

    def __init__(self, cfg, output_root, frame_ids, backend_factory=None):
        self.cfg = copy.deepcopy(cfg)
        self.output = Path(output_root) / 'gaussian_map'
        self.directory = self.output / 'submaps'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / 'manifest.json'
        self.options = dict(cfg.get('gaussian_map', {}).get('submaps', {}))
        self.frames_per_submap = self.options.get('frames', 50)
        self.overlap_frames = self.options.get('overlap_frames', 1)
        self.ranges = partition_submaps(frame_ids, self.frames_per_submap, self.overlap_frames)
        self.frames_per_submap = int(self.frames_per_submap)
        self.overlap_frames = int(self.overlap_frames)
        self.frame_ids = [int(frame) for frame in frame_ids]
        if len(self.frame_ids) < 2:
            raise ValueError('A submap sequence needs at least two processed images')
        self.max_boundary_landmarks = int(self.options.get('max_boundary_landmarks', 10000))
        if self.max_boundary_landmarks < 0:
            raise ValueError('max_boundary_landmarks must be nonnegative')
        self._factory = GaussianMapBackend if backend_factory is None else backend_factory
        self._backend = None
        self._boundary = None
        self._current_index = 0
        self._pair_index = 0
        self._entries = []
        self._known_objects = {}
        self._last_report = {}
        self._finalized = False
        self._write_manifest()

    @property
    def backend(self):
        """Current resident backend, or None after a submap was archived."""
        return self._backend

    @property
    def store(self):
        if self._backend is None:
            raise RuntimeError('No submap map geometry is currently resident')
        return self._backend.store

    @property
    def frames_dir(self):
        if self._backend is None:
            raise RuntimeError('No submap is currently open')
        return self._backend.frames_dir

    def _relative(self, path):
        return Path(path).relative_to(self.output.parent).as_posix()

    def _effective_cfg(self):
        cfg = copy.deepcopy(self.cfg)
        section = cfg.setdefault('gaussian_map', {})
        refinement = section.setdefault('refinement', {})
        insertion = int(section.get('max_new_gaussians_per_frame', 20000))
        topology = int(refinement.get('max_new_per_update', 1024)) if refinement.get('enabled', False) else 0
        required = self.max_boundary_landmarks + (self.frames_per_submap-1)*(insertion+topology)
        requested = int(refinement.get('max_total_gaussians', 0))
        refinement['max_total_gaussians'] = max(3000000, requested, required)
        return cfg

    def _open(self):
        specification = self.ranges[self._current_index]
        path = self.directory / f'submap_{self._current_index:06d}'
        cfg = self._effective_cfg()
        self._backend = self._factory(cfg, self.output.parent, map_output=path)
        if self._boundary is not None:
            self._restore_boundary(self._backend, self._boundary)
            self._boundary = None
        entry = dict(specification, state='open', directory=self._relative(path),
                     checkpoint=self._relative(path / 'map' / 'manifest.json'),
                     snapshots=self._relative(path / 'frames'),
                     trajectory=self._relative(path / 'trajectory.jsonl'),
                     world_from_submap=np.eye(4).tolist(),
                     coordinate_system='background_world;dynamic_object_canonical',
                     map_id_namespace=f'submap_{self._current_index:06d}',
                     max_total_gaussians=cfg['gaussian_map']['refinement']['max_total_gaussians'],
                     processed_pairs=0, live_gaussians=0, allocated_gaussians=0)
        self._entries.append(entry)
        self._write_manifest()

    def _capture_boundary(self, backend):
        tracks = dict(backend._track_landmarks)
        live = {track: value for track, value in tracks.items()
                if backend.store.is_live(np.asarray([value[1]], np.int64))[0]}
        # The feature frontend bounds this dictionary further after each pair.
        # A separate cap prevents an unusually dense feature set retaining a
        # whole previous submap through its sparse boundary context.
        selected_ids = sorted({mid for _, mid in live.values()})[:self.max_boundary_landmarks]
        chosen = set(selected_ids)
        live = {track: value for track, value in live.items() if value[1] in chosen}
        batch = backend.store.select(np.asarray(selected_ids, np.int64))
        self._known_objects.update({int(oid): dict(metadata)
                                    for oid, metadata in backend.store.to_dict()['objects'].items()})
        return dict(camera_to_world=np.array(backend._camera_to_world, copy=True),
                    pose_frame=backend._pose_frame,
                    object_to_world={int(oid): np.array(pose, copy=True)
                                     for oid, pose in backend._object_to_world.items()},
                    tracks=live, batch=batch,
                    observation_links={int(oid): tuple(mid for mid in mids if mid in chosen)
                                       for oid, mids in backend.ownership.links.items()
                                       if any(mid in chosen for mid in mids)},
                    observation_frames=dict(backend.ownership.last_frames),
                    source_submap_id=self._current_index)

    def _restore_boundary(self, backend, boundary):
        backend._camera_to_world = boundary['camera_to_world'].copy()
        backend._pose_frame = int(boundary['pose_frame'])
        backend._object_to_world = {oid: pose.copy() for oid, pose in boundary['object_to_world'].items()}
        for oid, metadata in self._known_objects.items():
            backend.store.set_status(oid, metadata['status'], metadata.get('last_frame'))
        batch = boundary['batch']
        remap = {}
        for oid in np.unique(batch['object_ids']):
            positions = np.flatnonzero(batch['object_ids'] == oid)
            selected = {name: value[positions].copy() for name, value in batch.items()
                        if name != 'optimizer_state'}
            selected['optimizer_state'] = dict(
                map_ids=selected['map_ids'].copy(), params={
                    name: {key: value[positions].copy() for key, value in state.items()}
                    for name, state in batch['optimizer_state']['params'].items()})
            old_ids = selected['map_ids'].copy()
            new_ids = backend.store.import_rows(selected, int(oid))
            remap.update(zip(map(int, old_ids), map(int, new_ids)))
        backend._track_landmarks = {track: (oid, remap[mid])
                                    for track, (oid, mid) in boundary['tracks'].items() if mid in remap}
        backend.ownership.links = {oid: tuple(remap[mid] for mid in mids if mid in remap)
                                   for oid, mids in boundary['observation_links'].items()}
        backend.ownership.last_frames = {oid: boundary['observation_frames'][oid]
                                         for oid in backend.ownership.links}
        # Confirmation counters refer to local update numbers. Restart them;
        # measured labels and their remapped links remain available for voting.
        backend.ownership.pending.clear()
        provenance = dict(source_submap_id=boundary['source_submap_id'],
                          destination_submap_id=self._current_index,
                          frame=int(boundary['pose_frame']),
                          source_map_ids=list(remap), destination_map_ids=list(remap.values()),
                          geometry='same_global_owner_canonical_coordinates',
                          tracked_anchor_count=len(remap))
        with (backend.output / 'boundary_seed.json').open('w') as stream:
            json.dump(provenance, stream, indent=2)
            stream.write('\n')

    def retain_tracks(self, track_ids):
        if self._backend is not None:
            self._backend.retain_tracks(track_ids)
        if self._boundary is not None:
            keep = set(map(int, track_ids))
            self._boundary['tracks'] = {track: value for track, value in self._boundary['tracks'].items()
                                         if track in keep}
            ids = {mid for _, mid in self._boundary['tracks'].values()}
            batch = self._boundary['batch']
            positions = np.flatnonzero(np.isin(batch['map_ids'], list(ids)))
            self._boundary['batch'] = {
                name: value[positions].copy() for name, value in batch.items() if name != 'optimizer_state'}
            self._boundary['batch']['optimizer_state'] = dict(
                map_ids=batch['map_ids'][positions].copy(), params={
                    name: {key: value[positions].copy() for key, value in state.items()}
                    for name, state in batch['optimizer_state']['params'].items()})
            self._boundary['observation_links'] = {
                oid: tuple(mid for mid in mids if mid in ids)
                for oid, mids in self._boundary['observation_links'].items() if any(mid in ids for mid in mids)}

    def _close(self, registry, *, carry):
        backend = self._backend
        if carry:
            self._boundary = self._capture_boundary(backend)
        result = backend.finalize(registry)
        entry = self._entries[-1]
        entry.update(state='closed', processed_pairs=backend._updates,
                     live_gaussians=len(backend.store), allocated_gaussians=backend.store.next_id,
                     object_ids=backend.store.object_ids,
                     object_status_at_close={str(oid): backend.store.get_status(oid)
                                             for oid in backend.store.object_ids})
        # No backend or store object is retained for an archived submap.
        self._backend = None
        self._write_manifest()
        return result

    def process(self, first, second, *args, **kwargs):
        if self._finalized:
            raise RuntimeError('Cannot append to a finalized submap sequence')
        if self._pair_index >= len(self.frame_ids)-1:
            raise ValueError('Received more map pairs than the configured frame sequence')
        expected = self.frame_ids[self._pair_index:self._pair_index+2]
        if [int(first), int(second)] != expected:
            raise ValueError(f'Submap pairs must follow the configured frame sequence: expected {expected}')
        if self._backend is None:
            if self._entries:
                self._current_index += 1
            self._open()
        camera0 = (np.eye(4) if self._backend._camera_to_world is None else
                   self._backend._camera_to_world.copy())
        local = self._backend.process(first, second, *args, **kwargs)
        # Registry is the eighth positional backend argument after the pair,
        # or an explicitly named argument for callers using keyword arguments.
        registry = kwargs.get('registry', args[7] if len(args) > 7 else None)
        if registry is None:
            raise TypeError('Submap mapping requires the shared object registry')
        camera1 = self._backend._camera_to_world
        trajectory = dict(first=int(first), second=int(second), submap_id=self._current_index,
                          camera_to_world_first=camera0.tolist(), camera_to_world_second=camera1.tolist())
        for label, frame, camera in (('first', first, camera0), ('second', second, camera1)):
            trajectory['object_to_world_'+label] = {
                str(oid): (camera @ pose).tolist()
                for oid, pose in self._backend._views[int(frame)]['object_to_camera'].items()}
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(trajectory)+'\n')
        entry = self._entries[-1]
        entry.update(processed_pairs=self._backend._updates, live_gaussians=len(self._backend.store),
                     allocated_gaussians=self._backend.store.next_id)
        self._pair_index += 1
        reached_boundary = self._pair_index == self.ranges[self._current_index]['end_index']
        if reached_boundary:
            local = self._close(registry, carry=self._pair_index < len(self.frame_ids)-1)
        self._last_report = dict(local, updates=self._pair_index, submap_updates=local.get('updates', 0),
                                 submaps_enabled=True, submap_id=self._current_index,
                                 submap_image_count=self.ranges[self._current_index]['image_count'],
                                 submap_frames=self.ranges[self._current_index]['frame_ids'],
                                 submap_partial=self.ranges[self._current_index]['partial'],
                                 submap_closed=reached_boundary,
                                 submap_manifest=self._relative(self.manifest_path),
                                 checkpoint=entry['checkpoint'], snapshots=entry['snapshots'],
                                 archived_submaps=sum(item['state'] == 'closed' for item in self._entries),
                                 resident_submaps=int(self._backend is not None),
                                 global_stored_gaussians=sum(item['live_gaussians'] for item in self._entries),
                                 global_allocated_gaussians=sum(item['allocated_gaussians'] for item in self._entries),
                                 global_map_id='(submap_id, local_map_id)',
                                 boundary_landmarks=(len(self._boundary['batch']['map_ids'])
                                                     if self._boundary is not None else 0))
        self._write_manifest()
        return self.to_dict()

    def _write_manifest(self):
        objects = {str(oid): _json_copy(metadata) for oid, metadata in self._known_objects.items()}
        manifest = dict(format='fixed_gaussian_submaps', format_version=1,
                        frames_per_submap=self.frames_per_submap, overlap_frames=self.overlap_frames,
                        requested_frame_ids=self.frame_ids, planned_submaps=len(self.ranges),
                        processed_pairs=self._pair_index, finalized=self._finalized,
                        coordinate_system='background_world;dynamic_object_canonical',
                        world_from_submap=np.eye(4).tolist(),
                        global_map_id='(submap_id, local_map_id)',
                        global_object_ids=True, object_lifecycle_resets_at_boundary=False,
                        submaps=self._entries, objects=objects,
                        resident_submaps=int(self._backend is not None),
                        global_stored_gaussians=sum(item['live_gaussians'] for item in self._entries))
        temporary = self.manifest_path.with_suffix('.tmp.json')
        with temporary.open('w') as stream:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
        temporary.replace(self.manifest_path)

    def finalize(self, registry):
        if self._pair_index != len(self.frame_ids)-1:
            raise RuntimeError('Cannot finalize before all configured submap frame pairs were processed')
        if self._backend is not None:
            self._close(registry, carry=False)
        self._boundary = None
        self._known_objects.update({int(oid): dict(record) for oid, record in registry.objects.items()})
        self._finalized = True
        self._last_report.update(finalized=True, resident_submaps=0, boundary_landmarks=0,
                                 submap_manifest=self._relative(self.manifest_path))
        self._write_manifest()
        return self.to_dict()

    def to_dict(self):
        return _json_copy(self._last_report)


GaussianSubmapBackend = SubmapGaussianMapBackend
