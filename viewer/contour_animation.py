# Contour animation replay and update methods extracted from contour_mesh.py
# Mixin class providing animation replay, update, state save/restore, and persistence.

import numpy as np
import copy


class ContourAnimationMixin:
    """
    Mixin class providing contour animation methods for MeshLoader.
    Handles replay/update for each pipeline step, animation helpers,
    visibility/mode helpers, level selection logic, and persistence.

    Merged into the final class via ContourMeshMixin inheritance.
    All self.* references resolve at runtime on the combined class.
    """

    # ── Animation State Init ────────────────────────────────────────

    def _init_animation_properties(self):
        """Initialize all animation-related properties. Called from _init_contour_properties."""
        # Contour animation replay state
        self._contour_anim_active = False
        self._contour_anim_progress = 0.0
        self._contour_anim_total = 0
        self._contour_anim_original_indices = []
        self._contour_replayed = False
        self._find_contours_count = 0

        # Fill gaps animation replay state
        self._fill_gaps_inserted_indices = []  # Indices of contours inserted by refine_contours
        self._fill_gaps_anim_active = False
        self._fill_gaps_anim_progress = 0.0
        self._fill_gaps_anim_step = 0  # Current gap being highlighted
        self._fill_gaps_replayed = False

        # Find transitions animation replay state
        self._transitions_inserted_indices = []
        self._transitions_anim_active = False
        self._transitions_anim_progress = 0.0
        self._transitions_anim_step = 0
        self._transitions_replayed = False

        # Smoothing animation replay state
        self._smooth_anim_active = False
        self._smooth_anim_progress = 0.0
        self._smooth_bp_before = None  # Snapshot before smoothing
        self._smooth_bp_after = None   # Snapshot after smoothing
        self._smooth_replayed = False

        # Stream smooth animation replay state (post-cut)
        self._stream_smooth_anim_active = False
        self._stream_smooth_anim_progress = 0.0
        self._stream_smooth_bp_before = None   # [stream][level] snapshot
        self._stream_smooth_bp_after = None
        self._stream_smooth_swing_data = None
        self._stream_smooth_twist_data = None
        self._stream_smooth_num_levels = 0
        self._stream_smooth_replayed = False

        # Level select shrink animation state
        self._level_select_anim_active = False
        self._level_select_anim_progress = 0.0
        self._level_select_anim_scales = None   # {(stream_i, level_j): float 0-1}
        self._level_select_anim_unselected = None  # set of (stream_i, level_j)
        self._level_select_anim_num_levels = 0
        self._level_select_replayed = False
        self._level_select_anim_pending_resume = False

        # Build fibers replay state
        self._fiber_anim_waypoints = None      # Deep copy of waypoints for replay
        self._fiber_anim_stream_endpoints = None  # saved _stream_endpoints during defer
        self._fiber_anim_active = False
        self._fiber_anim_progress = 0.0
        self._fiber_anim_level_progress = None   # float: growth progress in levels (0..num_levels-1)
        self._build_fibers_replayed = False

        # BP color override during smooth animations {(i,j): (r,g,b,a)}
        self._smooth_anim_bp_colors = None

        # Cut animation replay state
        self._cut_anim_active = False
        self._cut_anim_progress = 0.0
        self._cut_color_before = None   # [stream_idx][level_idx] = RGB array
        self._cut_color_after = None    # [stream_idx][level_idx] = RGB array
        self._cut_bp_before = None      # [stream_idx][level_idx] = BP dict snapshot
        self._cut_bp_after = None       # [stream_idx][level_idx] = BP dict snapshot
        self._cut_anim_contour_colors = None  # Override for draw_contours (None = use normal)
        self._cut_replayed = False
        self._cut_num_levels_before = 0

        # Animation highlight for newly revealed contours (multiple simultaneous)
        self._anim_highlight_fades = {}   # {contour_idx: fade_value} 1.0→0.0

    # ── Visibility / Mode Helpers ───────────────────────────────────

    def _is_stream_mode(self):
        """Check if contours are in stream mode [stream][level] (post-cut)."""
        return (self.draw_contour_stream is not None
                and len(self.draw_contour_stream) > 0
                and isinstance(self.draw_contour_stream[0], list))

    def _set_level_visible(self, level_idx, visible):
        """Set visibility for a level index, works in both level and stream mode."""
        if self._is_stream_mode():
            for s in range(len(self.draw_contour_stream)):
                if level_idx < len(self.draw_contour_stream[s]):
                    self.draw_contour_stream[s][level_idx] = visible
        else:
            if level_idx < len(self.draw_contour_stream):
                self.draw_contour_stream[level_idx] = visible

    def _hide_all_levels(self):
        """Hide all contour levels, works in both modes."""
        if self._is_stream_mode():
            for s in range(len(self.draw_contour_stream)):
                for j in range(len(self.draw_contour_stream[s])):
                    self.draw_contour_stream[s][j] = False
        else:
            for i in range(len(self.draw_contour_stream)):
                self.draw_contour_stream[i] = False

    def _num_levels(self):
        """Get the number of contour levels (works in both modes)."""
        if self._is_stream_mode():
            return len(self.draw_contour_stream[0]) if len(self.draw_contour_stream) > 0 else 0
        return len(self.draw_contour_stream) if self.draw_contour_stream else 0

    # ── Contour Animation (Replay + Update) ─────────────────────────

    def replay_contour_animation(self):
        """Start replaying the contour reveal animation from origin to insertion.
        Each contour appears one by one with a highlight pulse and BP grow effect."""
        if self.contours is None or len(self.contours) == 0:
            return
        # Switch to level mode if in stream mode (e.g. after cut replay)
        if hasattr(self, '_precut_contours') and self._precut_contours is not None:
            self.contours = [list(c) for c in self._precut_contours]
            self.bounding_planes = [[copy.deepcopy(bp) for bp in level]
                                    for level in self._precut_bounding_planes]
            if hasattr(self, '_precut_draw_contour_stream') and self._precut_draw_contour_stream is not None:
                self.draw_contour_stream = list(self._precut_draw_contour_stream)
        # _precut_bounding_planes is post-smooth; revert to pre-smooth BPs
        if getattr(self, '_smooth_bp_before_level', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before_level)
        elif getattr(self, '_smooth_bp_before', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before)
        total = self._num_levels()
        # Build set of indices inserted by later steps to skip during contour animation
        skip_indices = set(getattr(self, '_fill_gaps_inserted_indices', []))
        skip_indices.update(getattr(self, '_transitions_inserted_indices', []))
        self._contour_anim_original_indices = [i for i in range(total) if i not in skip_indices]
        self._contour_anim_total = len(self._contour_anim_original_indices)
        # Hide all contours
        self._hide_all_levels()
        # BP scale per level: 0.0 = point, 1.0 = full size
        self._contour_anim_bp_scale = {}
        self._anim_highlight_fades = {}
        self._contour_anim_progress = 0.0
        self._contour_anim_active = True
        self.is_draw_contours = True
        self.is_draw_bounding_box = True

    def update_contour_animation(self, dt):
        """Advance contour reveal animation with highlight and BP grow.
        Contours are revealed staggered within reveal_dur. Each highlight pulse
        lasts highlight_dur independently, so multiple can overlap."""
        if not self._contour_anim_active:
            return False

        orig_indices = getattr(self, '_contour_anim_original_indices', None)
        total = self._contour_anim_total
        if total == 0 or orig_indices is None:
            self._contour_anim_active = False
            return False

        reveal_dur = 2.0     # all contours appear within this time
        highlight_dur = 0.5  # each highlight pulse lasts this long
        time_per_contour = reveal_dur / max(total, 1)
        self._contour_anim_progress += dt
        progress = self._contour_anim_progress

        # +1 so first contour appears immediately
        revealed = min(int(progress / time_per_contour) + 1, total)

        # Reveal contours and update BP scale
        for k in range(revealed):
            idx = orig_indices[k]
            self._set_level_visible(idx, True)
            # BP scale: grows over highlight_dur from reveal time
            slot_start = k * time_per_contour
            bp_t = min(max((progress - slot_start) / highlight_dur, 0.0), 1.0)
            # Ease out: fast start, gentle finish
            bp_t = 1.0 - (1.0 - bp_t) * (1.0 - bp_t)
            self._contour_anim_bp_scale[idx] = bp_t

        # Update highlight fades for all revealed contours
        fades = {}
        for k in range(revealed):
            idx = orig_indices[k]
            slot_start = k * time_per_contour
            elapsed = progress - slot_start
            fade = max(0.0, 1.0 - elapsed / highlight_dur)
            if fade > 0.01:
                fades[idx] = fade
        self._anim_highlight_fades = fades

        # Done when last highlight has faded
        last_reveal_time = (total - 1) * time_per_contour
        if progress >= last_reveal_time + highlight_dur:
            self._contour_anim_active = False
            self._contour_replayed = True
            self._anim_highlight_fades = {}
            self._contour_anim_bp_scale = {}
            for idx in orig_indices:
                self._set_level_visible(idx, True)
            return False

        return True

    # ── Fill Gaps Animation ─────────────────────────────────────────

    def replay_fill_gaps_animation(self):
        """Start replaying the fill gaps animation, revealing inserted contours one by one."""
        if not self._fill_gaps_inserted_indices or len(self._fill_gaps_inserted_indices) == 0:
            # No gaps were filled - mark as replayed immediately
            self._fill_gaps_replayed = True
            return
        # Switch to level mode if in stream mode
        if hasattr(self, '_precut_contours') and self._precut_contours is not None:
            self.contours = [list(c) for c in self._precut_contours]
            self.bounding_planes = [[copy.deepcopy(bp) for bp in level]
                                    for level in self._precut_bounding_planes]
            if hasattr(self, '_precut_draw_contour_stream') and self._precut_draw_contour_stream is not None:
                self.draw_contour_stream = list(self._precut_draw_contour_stream)
        # _precut_bounding_planes is post-smooth; revert to pre-smooth BPs
        if getattr(self, '_smooth_bp_before_level', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before_level)
        elif getattr(self, '_smooth_bp_before', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before)
        # Hide only the gap-filled contours, leave everything else as-is
        for idx in self._fill_gaps_inserted_indices:
            self._set_level_visible(idx, False)
        self._fill_gaps_anim_step = 0
        self._fill_gaps_anim_progress = 0.0
        self._fill_gaps_anim_active = True
        self.is_draw_contours = True

    def update_fill_gaps_animation(self, dt):
        """Advance fill gaps animation. Reveals inserted contours one by one with highlight."""
        if not self._fill_gaps_anim_active:
            return False

        indices = self._fill_gaps_inserted_indices
        if len(indices) == 0:
            self._fill_gaps_anim_active = False
            self._fill_gaps_replayed = True
            self._anim_highlight_fades = {}
            return False

        time_per_gap = 0.5
        self._fill_gaps_anim_progress += dt

        # +1 so first contour appears immediately at t=0
        revealed = min(int(self._fill_gaps_anim_progress / time_per_gap) + 1, len(indices))

        for i in range(revealed):
            self._set_level_visible(indices[i], True)

        self._fill_gaps_anim_step = revealed - 1

        # Highlight the most recently revealed contour
        current_slot = revealed - 1
        idx = indices[current_slot]
        time_in_slot = self._fill_gaps_anim_progress - current_slot * time_per_gap
        fade = max(0.0, 1.0 - time_in_slot / time_per_gap)
        self._anim_highlight_fades = {idx: fade} if fade > 0.01 else {}

        # Done when last contour's highlight has faded
        if self._fill_gaps_anim_progress >= len(indices) * time_per_gap:
            self._fill_gaps_anim_active = False
            self._fill_gaps_replayed = True
            self._anim_highlight_fades = {}
            for idx in indices:
                self._set_level_visible(idx, True)
            return False

        return True

    # ── Transitions Animation ───────────────────────────────────────

    def replay_transitions_animation(self):
        """Start replaying the transitions animation, revealing inserted contours one by one."""
        if not self._transitions_inserted_indices or len(self._transitions_inserted_indices) == 0:
            self._transitions_replayed = True
            return
        # Switch to level mode if in stream mode
        if hasattr(self, '_precut_contours') and self._precut_contours is not None:
            self.contours = [list(c) for c in self._precut_contours]
            self.bounding_planes = [[copy.deepcopy(bp) for bp in level]
                                    for level in self._precut_bounding_planes]
            if hasattr(self, '_precut_draw_contour_stream') and self._precut_draw_contour_stream is not None:
                self.draw_contour_stream = list(self._precut_draw_contour_stream)
        # _precut_bounding_planes is post-smooth; revert to pre-smooth BPs
        if getattr(self, '_smooth_bp_before_level', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before_level)
        elif getattr(self, '_smooth_bp_before', None) is not None:
            self._apply_bp_snapshot(self._smooth_bp_before)
        # _precut_draw_contour_stream may have fill-gap contours hidden (deferred).
        # Transitions starts after fill-gaps, so fill-gap contours must be visible.
        for idx in getattr(self, '_fill_gaps_inserted_indices', []):
            self._set_level_visible(idx, True)
        # Hide the transition-inserted ones (to reveal them during animation)
        for idx in self._transitions_inserted_indices:
            self._set_level_visible(idx, False)
        self._transitions_anim_step = 0
        self._transitions_anim_progress = 0.0
        self._transitions_anim_active = True
        self.is_draw_contours = True

    def update_transitions_animation(self, dt):
        """Advance transitions animation. Reveals inserted contours one by one with highlight."""
        if not self._transitions_anim_active:
            return False

        indices = self._transitions_inserted_indices
        if len(indices) == 0:
            self._transitions_anim_active = False
            self._transitions_replayed = True
            self._anim_highlight_fades = {}
            return False

        time_per_item = 0.5
        self._transitions_anim_progress += dt

        # +1 so first contour appears immediately at t=0
        revealed = min(int(self._transitions_anim_progress / time_per_item) + 1, len(indices))

        for i in range(revealed):
            self._set_level_visible(indices[i], True)

        self._transitions_anim_step = revealed - 1

        # Highlight the most recently revealed contour
        current_slot = revealed - 1
        idx = indices[current_slot]
        time_in_slot = self._transitions_anim_progress - current_slot * time_per_item
        fade = max(0.0, 1.0 - time_in_slot / time_per_item)
        self._anim_highlight_fades = {idx: fade} if fade > 0.01 else {}

        # Done when last contour's highlight has faded
        if self._transitions_anim_progress >= len(indices) * time_per_item:
            self._transitions_anim_active = False
            self._transitions_replayed = True
            self._anim_highlight_fades = {}
            for idx in indices:
                self._set_level_visible(idx, True)
            return False

        return True

    # ── Animation Helpers ───────────────────────────────────────────

    def _apply_bp_snapshot(self, snapshot):
        """Apply a bounding plane snapshot back to live data."""
        for i, level in enumerate(snapshot):
            if i >= len(self.bounding_planes):
                break
            for j, snap in enumerate(level):
                if j >= len(self.bounding_planes[i]):
                    break
                bp = self.bounding_planes[i][j]
                bp['mean'] = snap['mean'].copy()
                bp['basis_x'] = snap['basis_x'].copy()
                bp['basis_y'] = snap['basis_y'].copy()
                bp['basis_z'] = snap['basis_z'].copy()
                if snap['bounding_plane'] is not None:
                    bp['bounding_plane'] = snap['bounding_plane'].copy()
                bp['square_like'] = snap.get('square_like', False)

    def _save_replay_state(self):
        """Save current display state so replay can restore it on completion.
        Each replay shows only its step's transformation, then restores."""
        state = {
            'contours': self.contours,
            'bounding_planes': [[copy.deepcopy(bp) for bp in group] for group in self.bounding_planes],
            'draw_contour_stream': [list(dcs) if isinstance(dcs, list) else dcs
                                    for dcs in self.draw_contour_stream] if self.draw_contour_stream else None,
        }
        # Track alias relationships
        has_stream = hasattr(self, 'stream_contours') and self.stream_contours is not None
        contours_aliased = has_stream and self.contours is self.stream_contours
        bps_aliased = has_stream and self.bounding_planes is self.stream_bounding_planes
        state['contours_aliased'] = contours_aliased
        state['bps_aliased'] = bps_aliased
        if has_stream and not bps_aliased:
            state['stream_bounding_planes'] = [[copy.deepcopy(bp) for bp in s]
                                                for s in self.stream_bounding_planes]
        if has_stream and not contours_aliased:
            state['stream_contours'] = self.stream_contours
        return state

    def _restore_replay_state(self, state):
        """Restore display state saved by _save_replay_state."""
        if state is None:
            return
        self.contours = state['contours']
        self.bounding_planes = state['bounding_planes']
        if state.get('draw_contour_stream') is not None:
            self.draw_contour_stream = state['draw_contour_stream']
        # Restore stream aliases
        if state.get('contours_aliased'):
            self.stream_contours = self.contours
        elif 'stream_contours' in state:
            self.stream_contours = state['stream_contours']
        if state.get('bps_aliased'):
            self.stream_bounding_planes = self.bounding_planes
        elif 'stream_bounding_planes' in state:
            self.stream_bounding_planes = state['stream_bounding_planes']

    def _smooth_wave_t(self, overall_t, level_idx, num_levels, overlap=0.4):
        """Compute per-level eased t for wave animation from origin to insertion."""
        if num_levels <= 1:
            level_t = overall_t
        else:
            level_start = (1.0 - overlap) * (level_idx / (num_levels - 1))
            level_t = np.clip((overall_t - level_start) / overlap, 0.0, 1.0)
        return level_t * level_t * (3.0 - 2.0 * level_t)  # ease in-out

    @staticmethod
    def _rodrigues(v, axis, angle):
        """Rotate vector v around axis by angle using Rodrigues' formula."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)

    def _cut_wave_t(self, overall_t, level_idx, num_levels, overlap=0.4):
        """Compute per-level eased t for wave animation."""
        if num_levels <= 1:
            level_t = overall_t
        else:
            level_start = (1.0 - overlap) * (level_idx / (num_levels - 1))
            level_t = np.clip((overall_t - level_start) / overlap, 0.0, 1.0)
        return level_t * level_t * (3.0 - 2.0 * level_t)  # smoothstep

    def _remap_smooth_data_to_stream_mode(self, num_streams, num_levels):
        """Remap smooth animation snapshots from [level][contour] to [stream][level].

        After cut, bounding_planes change from [level][contour] to [stream][level].
        The smooth snapshots were captured in level mode, so we remap them using
        stream_groups to match the new stream layout.
        """
        smooth_bp_before = getattr(self, '_smooth_bp_before', None)
        smooth_bp_after = getattr(self, '_smooth_bp_after', None)
        smooth_swing = getattr(self, '_smooth_swing_data', None)
        smooth_twist = getattr(self, '_smooth_twist_data', None)

        if smooth_bp_before is None or smooth_bp_after is None:
            return

        if not hasattr(self, 'stream_groups') or self.stream_groups is None:
            return

        def _remap(data):
            """Remap [level][contour] data to [stream][level]."""
            remapped = [[] for _ in range(num_streams)]
            for level_i in range(num_levels):
                for stream_i in range(num_streams):
                    # Find which group this stream belongs to at this level
                    group_idx = 0
                    if level_i < len(self.stream_groups):
                        for gi, group in enumerate(self.stream_groups[level_i]):
                            if stream_i in group:
                                group_idx = gi
                                break
                    # Get the data for this original contour
                    if level_i < len(data) and group_idx < len(data[level_i]):
                        remapped[stream_i].append(copy.deepcopy(data[level_i][group_idx]))
                    elif level_i < len(data) and len(data[level_i]) > 0:
                        remapped[stream_i].append(copy.deepcopy(data[level_i][0]))
                    else:
                        remapped[stream_i].append(None)
            return remapped

        self._smooth_bp_before = _remap(smooth_bp_before)
        self._smooth_bp_after = _remap(smooth_bp_after)
        if smooth_swing is not None:
            self._smooth_swing_data = _remap(smooth_swing)
        if smooth_twist is not None:
            self._smooth_twist_data = _remap(smooth_twist)
        self._smooth_num_levels = num_levels

    def _apply_stream_bp_snapshot(self, snapshot):
        """Apply a BP snapshot to stream_bounding_planes[stream][level]."""
        if not hasattr(self, 'stream_bounding_planes') or self.stream_bounding_planes is None:
            return
        for i, stream_bps in enumerate(snapshot):
            if i >= len(self.stream_bounding_planes):
                break
            for j, snap in enumerate(stream_bps):
                if snap is None:
                    continue
                if j >= len(self.stream_bounding_planes[i]):
                    break
                bp = self.stream_bounding_planes[i][j]
                for key in ('mean', 'basis_x', 'basis_y', 'basis_z', 'bounding_plane'):
                    if key in snap and snap[key] is not None:
                        bp[key] = np.array(snap[key]).copy()
                if 'square_like' in snap:
                    bp['square_like'] = snap['square_like']

    # ── Smooth Animation (Replay + Update) ──────────────────────────

    def replay_smooth_animation(self):
        """Start replaying the smoothing animation.
        Phase 1: fade muscle transparency from current to 0.5
        Phase 2: interpolate axes/bounding planes"""
        if self._smooth_bp_before is None or self._smooth_bp_after is None:
            self._smooth_replayed = True
            return
        # Switch to level mode if we're in stream mode (e.g. after cut replay)
        if hasattr(self, '_precut_contours') and self._precut_contours is not None:
            self.contours = [list(c) for c in self._precut_contours]
            self.bounding_planes = [[copy.deepcopy(bp) for bp in level]
                                    for level in self._precut_bounding_planes]
            if hasattr(self, '_precut_draw_contour_stream') and self._precut_draw_contour_stream is not None:
                self.draw_contour_stream = list(self._precut_draw_contour_stream)
        # _precut_draw_contour_stream may have fill-gap/transition contours hidden (deferred).
        # Smooth starts after all pre-cut steps, so all contours must be visible.
        for idx in getattr(self, '_fill_gaps_inserted_indices', []):
            self._set_level_visible(idx, True)
        for idx in getattr(self, '_transitions_inserted_indices', []):
            self._set_level_visible(idx, True)
        # Reset to initial state
        self._apply_bp_snapshot(self._smooth_bp_before)
        self._smooth_anim_bp_colors = None
        self._smooth_anim_progress = 0.0
        self._smooth_anim_active = True
        self._smooth_anim_orig_transparency = getattr(self, 'transparency', 1.0)

    def update_smooth_animation(self, dt):
        """Advance smoothing animation in four phases.
        Phase 1 (0-0.5s):   fade transparency to 0.5
        Phase 2a (0.5-2.0s): rotate axes to align z-axis
        Phase 2b (2.0-3.5s): twist around z to align x/y
        Phase 2c (3.5-4.5s): interpolate bounding plane corners"""
        if not self._smooth_anim_active:
            return False

        bp_before = self._smooth_bp_before
        bp_after = self._smooth_bp_after
        swing_data = getattr(self, '_smooth_swing_data', None)
        twist_data = getattr(self, '_smooth_twist_data', None)
        if bp_before is None or bp_after is None:
            self._smooth_anim_active = False
            self._smooth_replayed = True
            return False

        self._smooth_anim_progress += dt
        progress = self._smooth_anim_progress

        # Phase durations
        p1_dur = 0.5    # transparency fade (fast)
        p2a_dur = 1.5   # z-axis swing
        p2b_dur = 1.5   # x/y twist around z
        p2c_dur = 1.0   # bounding plane corners
        p1_end = p1_dur
        p2a_end = p1_end + p2a_dur
        p2b_end = p2a_end + p2b_dur
        p2c_end = p2b_end + p2c_dur
        total_duration = p2c_end

        num_levels = getattr(self, '_smooth_num_levels', len(bp_before))
        orig_alpha = getattr(self, '_smooth_anim_orig_transparency', 1.0)
        target_alpha = 0.5

        # --- Phase 1: fade transparency ---
        if progress < p1_end:
            fade_t = progress / p1_dur
            fade_t = fade_t * fade_t * (3.0 - 2.0 * fade_t)
            new_alpha = orig_alpha + (target_alpha - orig_alpha) * fade_t
            self.transparency = new_alpha
            if self.vertex_colors is not None and self.is_draw_scalar_field:
                self.vertex_colors[:, 3] = new_alpha
            return True

        # Ensure transparency at target
        self.transparency = target_alpha
        if self.vertex_colors is not None and self.is_draw_scalar_field:
            self.vertex_colors[:, 3] = target_alpha

        # Compute sub-phase overall progress (0..1 within each sub-phase)
        swing_overall = np.clip((progress - p1_end) / p2a_dur, 0.0, 1.0)
        twist_overall = np.clip((progress - p2a_end) / p2b_dur, 0.0, 1.0) if progress > p2a_end else 0.0
        bp_overall = np.clip((progress - p2b_end) / p2c_dur, 0.0, 1.0) if progress > p2b_end else 0.0

        rod = self._rodrigues

        for i in range(min(len(bp_before), len(self.bounding_planes))):
            swing_t = self._smooth_wave_t(swing_overall, i, num_levels)
            twist_t = self._smooth_wave_t(twist_overall, i, num_levels)
            bp_t = self._smooth_wave_t(bp_overall, i, num_levels)

            for j in range(min(len(bp_before[i]), len(self.bounding_planes[i]))):
                if j >= len(bp_after[i]):
                    continue
                bpl = self.bounding_planes[i][j]
                before = bp_before[i][j]
                after = bp_after[i][j]

                z_b = before['basis_z']
                x_b = before['basis_x']
                y_b = before['basis_y']
                z_a = after['basis_z']

                has_swing = (swing_data is not None and i < len(swing_data) and
                             j < len(swing_data[i]) and swing_data[i][j] is not None)
                has_twist = (twist_data is not None and i < len(twist_data) and
                             j < len(twist_data[i]) and twist_data[i][j] is not None)

                # Phase 2a: swing all axes to align z with target
                if has_swing:
                    swing_axis, swing_angle = swing_data[i][j]
                    if swing_angle > 1e-10 and swing_t > 0:
                        a = swing_angle * swing_t
                        z = rod(z_b, swing_axis, a)
                        x = rod(x_b, swing_axis, a)
                        y = rod(y_b, swing_axis, a)
                    else:
                        z = z_b.copy()
                        x = x_b.copy()
                        y = y_b.copy()
                else:
                    z = z_b.copy()
                    x = x_b.copy()
                    y = y_b.copy()

                # Phase 2b: twist x,y around z_after (swing is complete at this point)
                if has_twist and abs(twist_data[i][j]) > 1e-10 and twist_t > 0:
                    ta = twist_data[i][j] * twist_t
                    x = rod(x, z_a, ta)
                    y = rod(y, z_a, ta)

                bpl['basis_x'] = x
                bpl['basis_y'] = y
                bpl['basis_z'] = z

                # Phase 2c: bounding plane corners + mean
                bpl['mean'] = (1 - bp_t) * before['mean'] + bp_t * after['mean']
                if before['bounding_plane'] is not None and after['bounding_plane'] is not None:
                    bpl['bounding_plane'] = (1 - bp_t) * before['bounding_plane'] + bp_t * after['bounding_plane']

                # Color lerp when square_like changes
                sq_before = before.get('square_like', False)
                sq_after = after.get('square_like', False)
                if sq_before != sq_after:
                    color_b = (1, 0, 0, 1) if sq_before else (0, 0, 0, 1)
                    color_a = (1, 0, 0, 1) if sq_after else (0, 0, 0, 1)
                    lerped = tuple((1 - bp_t) * b + bp_t * a for b, a in zip(color_b, color_a))
                    if self._smooth_anim_bp_colors is None:
                        self._smooth_anim_bp_colors = {}
                    self._smooth_anim_bp_colors[(i, j)] = lerped
                elif self._smooth_anim_bp_colors is not None and (i, j) in self._smooth_anim_bp_colors:
                    del self._smooth_anim_bp_colors[(i, j)]

        if progress >= total_duration:
            self._smooth_anim_active = False
            self._smooth_replayed = True
            self._smooth_anim_bp_colors = None
            self._apply_bp_snapshot(bp_after)
            return False

        return True

    # ── Stream Smooth Animation (post-cut) ──────────────────────────

    def replay_stream_smooth_animation(self):
        """Start replaying the stream smooth animation."""
        if self._stream_smooth_bp_before is None or self._stream_smooth_bp_after is None:
            self._stream_smooth_replayed = True
            return
        # Ensure we're in stream mode
        if hasattr(self, 'stream_contours') and self.stream_contours is not None:
            self.contours = self.stream_contours
            self.bounding_planes = self.stream_bounding_planes
            num_streams = len(self.stream_contours)
            num_levels = len(self.stream_contours[0]) if num_streams > 0 else 0
            self.draw_contour_stream = [[True] * num_levels for _ in range(num_streams)]
        self._apply_bp_snapshot(self._stream_smooth_bp_before)
        self._smooth_anim_bp_colors = None
        self._stream_smooth_anim_progress = 0.0
        self._stream_smooth_anim_active = True

    def update_stream_smooth_animation(self, dt):
        """Advance stream smooth animation in three phases (no transparency fade).
        Streams are staggered: each stream starts its 3-phase animation slightly
        after the previous, with overlap.
        Per-stream phases:
          Phase 1 (1.5s): rotate axes to align z-axis (swing)
          Phase 2 (1.5s): twist around z to align x/y
          Phase 3 (1.0s): interpolate bounding plane corners"""
        if not self._stream_smooth_anim_active:
            return False

        bp_before = self._stream_smooth_bp_before
        bp_after = self._stream_smooth_bp_after
        swing_data = self._stream_smooth_swing_data
        twist_data = self._stream_smooth_twist_data
        if bp_before is None or bp_after is None:
            self._stream_smooth_anim_active = False
            self._stream_smooth_replayed = True
            return False

        self._stream_smooth_anim_progress += dt
        progress = self._stream_smooth_anim_progress

        # Per-stream phase durations
        p1_dur = 1.5   # z-axis swing
        p2_dur = 1.5   # x/y twist around z
        p3_dur = 1.0   # bounding plane corners
        per_stream_dur = p1_dur + p2_dur + p3_dur  # 4.0s per stream

        num_levels = self._stream_smooth_num_levels
        num_streams = min(len(bp_before), len(self.bounding_planes))

        # Stagger: each stream starts 0.5s after the previous
        stream_stagger = 0.5
        total_duration = per_stream_dur + stream_stagger * max(num_streams - 1, 0)

        rod = self._rodrigues

        for i in range(num_streams):
            # Per-stream progress offset by stagger
            sp = progress - stream_stagger * i
            if sp < 0:
                continue  # This stream hasn't started yet

            # Sub-phase progress for this stream
            swing_overall = np.clip(sp / p1_dur, 0.0, 1.0)
            twist_overall = np.clip((sp - p1_dur) / p2_dur, 0.0, 1.0) if sp > p1_dur else 0.0
            bp_overall = np.clip((sp - p1_dur - p2_dur) / p3_dur, 0.0, 1.0) if sp > p1_dur + p2_dur else 0.0

            for j in range(min(len(bp_before[i]), len(self.bounding_planes[i]))):
                if j >= len(bp_after[i]):
                    continue

                swing_t = self._smooth_wave_t(swing_overall, j, num_levels)
                twist_t = self._smooth_wave_t(twist_overall, j, num_levels)
                bp_t = self._smooth_wave_t(bp_overall, j, num_levels)

                bpl = self.bounding_planes[i][j]
                before = bp_before[i][j]
                after = bp_after[i][j]

                z_b = before['basis_z']
                x_b = before['basis_x']
                y_b = before['basis_y']
                z_a = after['basis_z']

                has_swing = (swing_data is not None and i < len(swing_data) and
                             j < len(swing_data[i]) and swing_data[i][j] is not None)
                has_twist = (twist_data is not None and i < len(twist_data) and
                             j < len(twist_data[i]) and twist_data[i][j] is not None)

                # Phase 1: swing all axes to align z with target
                if has_swing:
                    swing_axis, swing_angle = swing_data[i][j]
                    if swing_angle > 1e-10 and swing_t > 0:
                        a = swing_angle * swing_t
                        z = rod(z_b, swing_axis, a)
                        x = rod(x_b, swing_axis, a)
                        y = rod(y_b, swing_axis, a)
                    else:
                        z = z_b.copy()
                        x = x_b.copy()
                        y = y_b.copy()
                else:
                    z = z_b.copy()
                    x = x_b.copy()
                    y = y_b.copy()

                # Phase 2: twist x,y around z_after
                if has_twist and abs(twist_data[i][j]) > 1e-10 and twist_t > 0:
                    ta = twist_data[i][j] * twist_t
                    x = rod(x, z_a, ta)
                    y = rod(y, z_a, ta)

                bpl['basis_x'] = x
                bpl['basis_y'] = y
                bpl['basis_z'] = z

                # Phase 3: bounding plane corners + mean
                bpl['mean'] = (1 - bp_t) * before['mean'] + bp_t * after['mean']
                if before['bounding_plane'] is not None and after['bounding_plane'] is not None:
                    bpl['bounding_plane'] = (1 - bp_t) * before['bounding_plane'] + bp_t * after['bounding_plane']

                # Color lerp when square_like changes
                sq_before = before.get('square_like', False)
                sq_after = after.get('square_like', False)
                if sq_before != sq_after:
                    color_b = (1, 0, 0, 1) if sq_before else (0, 0, 0, 1)
                    color_a = (1, 0, 0, 1) if sq_after else (0, 0, 0, 1)
                    lerped = tuple((1 - bp_t) * b + bp_t * a for b, a in zip(color_b, color_a))
                    if self._smooth_anim_bp_colors is None:
                        self._smooth_anim_bp_colors = {}
                    self._smooth_anim_bp_colors[(i, j)] = lerped
                elif self._smooth_anim_bp_colors is not None and (i, j) in self._smooth_anim_bp_colors:
                    del self._smooth_anim_bp_colors[(i, j)]

        if progress >= total_duration:
            self._stream_smooth_anim_active = False
            self._stream_smooth_replayed = True
            self._smooth_anim_bp_colors = None
            self._apply_bp_snapshot(bp_after)
            return False

        return True

    # ── Cut Animation (Replay + Update) ─────────────────────────────

    def replay_cut_animation(self):
        """Start replaying the cut animation from the beginning."""
        if self._cut_color_before is None or self._cut_color_after is None:
            self._cut_replayed = True
            return
        # Swap to stream-mode (post-cut) contour geometry
        if hasattr(self, 'stream_contours') and self.stream_contours is not None:
            self.contours = self.stream_contours
            self.bounding_planes = self.stream_bounding_planes
            num_streams = len(self.stream_contours)
            num_levels = len(self.stream_contours[0]) if num_streams > 0 else 0
            self.draw_contour_stream = [[True] * num_levels for _ in range(num_streams)]
        # Reset to pre-cut colors
        self._cut_anim_contour_colors = [[c.copy() for c in stream] for stream in self._cut_color_before]
        # Reset BPs to pre-cut state (includes smooth result axes)
        if self._cut_bp_before is not None:
            self._apply_stream_bp_snapshot(self._cut_bp_before)
        self._contour_anim_bp_scale = {}
        self._cut_bp_switched = False
        self._smooth_anim_bp_colors = None
        self._cut_anim_orig_transparency = self.transparency
        self._cut_anim_progress = 0.0
        self._cut_anim_active = True

    def update_cut_animation(self, dt):
        """Advance cut animation.

        Phase 0 (0-0.5s): Fade transparency to 0.5 (if not already).
        Phase 1 (0-2s): Color wave — lerp per-contour color from before to after.
        Phase 2a (2-3s): BP shrink — cut-level BPs scale from 1→0 (wave).
        Phase 2b (3-4s): BP grow — cut-level BPs scale from 0→1 (wave).
        """
        if not self._cut_anim_active:
            return False

        color_before = self._cut_color_before
        color_after = self._cut_color_after
        bp_before = self._cut_bp_before
        bp_after = self._cut_bp_after
        if color_before is None or color_after is None:
            self._cut_anim_active = False
            self._cut_replayed = True
            return False

        self._cut_anim_progress += dt

        num_streams = len(color_before)
        num_levels = len(color_before[0]) if num_streams > 0 else 0

        # Phase durations
        color_dur = 2.0
        bp_shrink_dur = 1.0
        bp_grow_dur = 1.0
        has_bp = getattr(self, '_cut_has_bp_change', False)
        bp_dur = (bp_shrink_dur + bp_grow_dur) if has_bp else 0.0
        total_duration = color_dur + bp_dur
        changed_levels = getattr(self, '_cut_bp_changed_levels', set())

        progress = self._cut_anim_progress

        # Phase 1: Color wave
        if progress < color_dur:
            overall_t = progress / color_dur
        else:
            overall_t = 1.0

        for i in range(num_streams):
            for j in range(num_levels):
                t = self._cut_wave_t(overall_t, j, num_levels)
                self._cut_anim_contour_colors[i][j] = (1 - t) * color_before[i][j] + t * color_after[i][j]

        # Phase 2: BP shrink-then-grow (only at levels where cutting occurred)
        if has_bp and bp_before is not None and bp_after is not None:
            bp_start = color_dur
            bp_mid = bp_start + bp_shrink_dur

            if progress > bp_start and progress < bp_mid:
                # Phase 2a: Shrink cut-level BPs to their means
                shrink_t = (progress - bp_start) / bp_shrink_dur
                for j in changed_levels:
                    wave_t = self._cut_wave_t(shrink_t, j, num_levels)
                    self._contour_anim_bp_scale[j] = 1.0 - wave_t

            elif progress >= bp_mid:
                # Transition: switch from pre-cut to post-cut BP positions (once)
                if not getattr(self, '_cut_bp_switched', False):
                    self._apply_stream_bp_snapshot(bp_after)
                    self._cut_bp_switched = True

                # Phase 2b: Grow cut-level BPs from their means
                grow_t = min((progress - bp_mid) / bp_grow_dur, 1.0)
                for j in changed_levels:
                    wave_t = self._cut_wave_t(grow_t, j, num_levels)
                    self._contour_anim_bp_scale[j] = wave_t

            # Animate square_like color transitions over Phase 2
            sq_changed = getattr(self, '_cut_sq_changed', set())
            if sq_changed:
                phase2_t = max(0.0, min((progress - bp_start) / (bp_shrink_dur + bp_grow_dur), 1.0))
                for (i, j) in sq_changed:
                    bb = bp_before[i][j]
                    ba = bp_after[i][j]
                    if bb is None or ba is None:
                        continue
                    color_b = (1, 0, 0, 1) if bb.get('square_like', False) else (0, 0, 0, 1)
                    color_a = (1, 0, 0, 1) if ba.get('square_like', False) else (0, 0, 0, 1)
                    bp_t = self._cut_wave_t(phase2_t, j, num_levels)
                    lerped = tuple((1 - bp_t) * b + bp_t * a for b, a in zip(color_b, color_a))
                    if self._smooth_anim_bp_colors is None:
                        self._smooth_anim_bp_colors = {}
                    self._smooth_anim_bp_colors[(i, j)] = lerped

        # Done?
        if progress >= total_duration:
            self._cut_anim_active = False
            self._cut_replayed = True
            self._cut_anim_contour_colors = None
            self._contour_anim_bp_scale = {}
            self._cut_bp_switched = False
            self._smooth_anim_bp_colors = None
            if has_bp:
                self._apply_stream_bp_snapshot(bp_after)
            return False

        return True

    # ── Level Selection Logic ───────────────────────────────────────

    def _save_level_select_post_state(self):
        """Save post-selection state for replay restoration (avoids destructive re-apply).
        Saves the filtered contours/bps — stream_contours stays unfiltered.
        Deep copies BPs to avoid corruption by other replays.
        Applies post-stream-smooth axes if smooth was deferred."""
        saved_bps = [[copy.deepcopy(bp) for bp in level] for level in self.bounding_planes]

        # Apply post-stream-smooth axes if smooth was deferred
        ss_after = getattr(self, '_stream_smooth_bp_after', None)
        if ss_after is not None and not getattr(self, '_stream_smooth_replayed', False):
            sel_levels = getattr(self, 'stream_selected_levels', None)
            for i in range(len(saved_bps)):
                if i >= len(ss_after):
                    continue
                for k in range(len(saved_bps[i])):
                    if sel_levels is not None and i < len(sel_levels) and k < len(sel_levels[i]):
                        orig_j = sel_levels[i][k]
                    else:
                        orig_j = k
                    if orig_j < len(ss_after[i]):
                        snap = ss_after[i][orig_j]
                        bp = saved_bps[i][k]
                        bp['mean'] = snap['mean'].copy()
                        bp['basis_x'] = snap['basis_x'].copy()
                        bp['basis_y'] = snap['basis_y'].copy()
                        bp['basis_z'] = snap['basis_z'].copy()
                        if snap['bounding_plane'] is not None:
                            bp['bounding_plane'] = snap['bounding_plane'].copy()

        self._level_select_anim_post = {
            'contours': [list(sc) for sc in self.contours],
            'bounding_planes': saved_bps,
            'draw_contour_stream': [list(dcs) for dcs in self.draw_contour_stream],
        }

    def _restore_level_select_post_state(self):
        """Restore post-selection state from saved snapshot (for replay completion).
        Only restores contours/bps/dcs — does NOT touch stream_contours."""
        post = self._level_select_anim_post
        self.contours = [list(sc) for sc in post['contours']]
        self.bounding_planes = [list(bp) for bp in post['bounding_planes']]
        self.draw_contour_stream = [list(dcs) for dcs in post['draw_contour_stream']]

    def _start_level_select_animation(self, defer=False):
        """Start shrink animation for unselected levels before applying selection.

        Args:
            defer: If True, compute animation data but apply selection immediately
                   without animating (for later replay).
        """
        if not hasattr(self, '_level_select_checkboxes') or self._level_select_checkboxes is None:
            return

        max_stream_count = self.max_stream_count
        orig = self._level_select_original

        # Identify unselected (stream, level) pairs
        unselected = set()
        for stream_i in range(max_stream_count):
            for level_i, checked in enumerate(self._level_select_checkboxes[stream_i]):
                if not checked:
                    unselected.add((stream_i, level_i))

        # Store animation data for replay (even when deferring)
        num_levels = len(orig['stream_contours'][0]) if max_stream_count > 0 else 0
        self._level_select_anim_unselected = unselected
        self._level_select_anim_num_levels = num_levels
        # Save pre-selection data for replay (deep copy BPs to avoid corruption by other replays)
        # Use post-stream-smooth axes if smooth was deferred (stream_bounding_planes still has pre-smooth)
        anim_bps = [[copy.deepcopy(bp) for bp in sbs]
                     for sbs in orig['stream_bounding_planes']]
        ss_after = getattr(self, '_stream_smooth_bp_after', None)
        if ss_after is not None and not getattr(self, '_stream_smooth_replayed', False):
            for i, stream_snaps in enumerate(ss_after):
                if i < len(anim_bps):
                    for j, snap in enumerate(stream_snaps):
                        if j < len(anim_bps[i]):
                            anim_bps[i][j]['mean'] = snap['mean'].copy()
                            anim_bps[i][j]['basis_x'] = snap['basis_x'].copy()
                            anim_bps[i][j]['basis_y'] = snap['basis_y'].copy()
                            anim_bps[i][j]['basis_z'] = snap['basis_z'].copy()
                            if snap['bounding_plane'] is not None:
                                anim_bps[i][j]['bounding_plane'] = snap['bounding_plane'].copy()
        self._level_select_anim_original = {
            'stream_contours': [list(sc) for sc in orig['stream_contours']],
            'stream_bounding_planes': anim_bps,
            'checkboxes': [list(cb) for cb in self._level_select_checkboxes],
        }

        # If nothing unselected, just apply directly
        if not unselected:
            self._level_select_window_open = False
            self._apply_level_selection()
            self._save_level_select_post_state()
            self._level_select_anim_pending_resume = True
            return

        if defer:
            # Deferred: apply selection immediately, replay animation later
            self._level_select_window_open = False
            self._apply_level_selection()
            self._save_level_select_post_state()
            self._level_select_anim_pending_resume = True
            self._level_select_replayed = False
            print(f"Level select animation deferred: {len(unselected)} unselected contours")
            return

        # Restore ALL levels visible (from original) so we can animate the shrink
        self.contours = [list(sc) for sc in orig['stream_contours']]
        self.bounding_planes = [list(bp) for bp in orig['stream_bounding_planes']]
        self.draw_contour_stream = [[True] * len(orig['stream_contours'][s]) for s in range(max_stream_count)]
        # Also update the stream aliases so draw_bounding_box sees them
        self.stream_contours = self.contours
        self.stream_bounding_planes = self.bounding_planes

        # Close selection window but keep checkboxes/original alive for _apply_level_selection
        self._level_select_window_open = False

        # Initialize animation (first run, not replay)
        self._level_select_anim_is_replay = False
        self._level_select_anim_scales = {key: 1.0 for key in unselected}
        self._level_select_anim_progress = 0.0
        self._level_select_anim_active = True

        print(f"Level select animation started: {len(unselected)} unselected contours, {num_levels} levels")

    def update_level_select_animation(self, dt):
        """Advance level select shrink animation: wave from origin to insertion."""
        if not self._level_select_anim_active:
            return False

        duration = 2.0
        self._level_select_anim_progress += dt / duration
        overall_t = min(self._level_select_anim_progress, 1.0)

        num_levels = self._level_select_anim_num_levels
        unselected = self._level_select_anim_unselected

        # Update scales for each unselected contour using wave
        for (s, l) in unselected:
            wave_t = self._smooth_wave_t(overall_t, l, num_levels)
            self._level_select_anim_scales[(s, l)] = 1.0 - wave_t  # shrink 1→0

        # Check completion
        if overall_t >= 1.0:
            self._level_select_anim_active = False
            self._level_select_anim_scales = None
            self._level_select_anim_unselected = None
            if getattr(self, '_level_select_anim_is_replay', False):
                # Replay: restore to post-selection state (after this step)
                self._restore_level_select_post_state()
                self._level_select_replayed = True
            else:
                # First run: apply selection and save post state
                self._apply_level_selection()
                self._save_level_select_post_state()
                self._level_select_anim_pending_resume = True
            print("Level select animation complete")

        return True

    def replay_level_select_animation(self):
        """Replay the level select shrink animation from saved data."""
        anim_orig = getattr(self, '_level_select_anim_original', None)
        if anim_orig is None:
            print("No level select animation data to replay")
            return

        unselected = set()
        checkboxes = anim_orig['checkboxes']
        max_stream_count = len(checkboxes)
        for stream_i in range(max_stream_count):
            for level_i, checked in enumerate(checkboxes[stream_i]):
                if not checked:
                    unselected.add((stream_i, level_i))

        if not unselected:
            self._level_select_replayed = True
            return

        # Restore all levels visible from saved pre-selection state (deep copies to avoid corruption)
        self.contours = [list(sc) for sc in anim_orig['stream_contours']]
        self.bounding_planes = [[copy.deepcopy(bp) for bp in sbs]
                                for sbs in anim_orig['stream_bounding_planes']]
        self.draw_contour_stream = [[True] * len(anim_orig['stream_contours'][s]) for s in range(max_stream_count)]
        self.stream_contours = self.contours
        self.stream_bounding_planes = self.bounding_planes

        # Start animation (replay mode — restores saved state on completion)
        num_levels = len(anim_orig['stream_contours'][0]) if max_stream_count > 0 else 0
        self._level_select_anim_is_replay = True
        self._level_select_anim_unselected = unselected
        self._level_select_anim_num_levels = num_levels
        self._level_select_anim_scales = {key: 1.0 for key in unselected}
        self._level_select_anim_progress = 0.0
        self._level_select_anim_active = True

        print(f"Replaying level select animation: {len(unselected)} unselected contours")

    def _apply_level_selection(self):
        """Apply the current checkbox selection to stream data (called by Finish Select)."""
        if not hasattr(self, '_level_select_checkboxes') or self._level_select_checkboxes is None:
            return

        max_stream_count = self.max_stream_count

        # Build selected levels from checkboxes
        stream_selected = []
        for stream_i in range(max_stream_count):
            selected = [i for i, checked in enumerate(self._level_select_checkboxes[stream_i]) if checked]
            stream_selected.append(selected)

        self.stream_selected_levels = stream_selected
        print(f"\nApplying level selection:")
        for s in range(max_stream_count):
            print(f"  Stream {s}: {self.stream_selected_levels[s]}")

        # Get original data
        orig = self._level_select_original
        orig_stream_contours = orig['stream_contours']
        orig_stream_bounding_planes = orig['stream_bounding_planes']
        orig_stream_groups = orig['stream_groups']

        # Apply selection to stream_contours and stream_bounding_planes
        new_stream_contours = [[] for _ in range(max_stream_count)]
        new_stream_bounding_planes = [[] for _ in range(max_stream_count)]
        new_stream_groups = []

        # Build new stream_groups (only for selected levels)
        all_selected = set()
        for s in range(max_stream_count):
            all_selected.update(self.stream_selected_levels[s])
        all_selected_sorted = sorted(all_selected)

        for level_i in all_selected_sorted:
            if level_i < len(orig_stream_groups):
                new_stream_groups.append(orig_stream_groups[level_i])

        for stream_i in range(max_stream_count):
            for level_i in self.stream_selected_levels[stream_i]:
                if level_i < len(orig_stream_contours[stream_i]):
                    new_stream_contours[stream_i].append(orig_stream_contours[stream_i][level_i])
                    new_stream_bounding_planes[stream_i].append(orig_stream_bounding_planes[stream_i][level_i])

        # Store filtered results separately — stream_contours stays unfiltered for replays
        self._selected_stream_contours = new_stream_contours
        self._selected_stream_bounding_planes = new_stream_bounding_planes
        self._selected_stream_groups = new_stream_groups

        # Update visualization from selected data
        self.contours = self._selected_stream_contours
        self.bounding_planes = self._selected_stream_bounding_planes
        self.draw_contour_stream = [[True] * len(self._selected_stream_contours[s]) for s in range(max_stream_count)]

        level_counts = [len(self._selected_stream_contours[s]) for s in range(max_stream_count)]
        print(f"Levels updated: {level_counts} per stream")

        # Close window and clean up
        self._level_select_window_open = False
        self._level_select_checkboxes = None
        self._level_select_original = None

    def _undo_level_selection(self):
        """Restore original state (all levels selected)."""
        if not hasattr(self, '_level_select_original') or self._level_select_original is None:
            return

        max_stream_count = self.max_stream_count

        # Reset checkboxes to all True (select all levels)
        self._level_select_checkboxes = []
        for stream_i in range(max_stream_count):
            num_levels_stream = len(self._level_select_original['stream_contours'][stream_i])
            self._level_select_checkboxes.append([True] * num_levels_stream)

        # Update visualization to show all levels
        self._update_level_select_visualization()
        print("Level selection reset to all levels selected")

    def _update_level_select_visualization(self):
        """Update visualization based on current checkbox state."""
        if not hasattr(self, '_level_select_checkboxes') or self._level_select_checkboxes is None:
            return
        if not hasattr(self, '_level_select_original') or self._level_select_original is None:
            return

        max_stream_count = self.max_stream_count
        orig = self._level_select_original

        # Temporarily update contours/bounding_planes for visualization
        # (without actually deleting unselected levels)
        temp_contours = [[] for _ in range(max_stream_count)]
        temp_bps = [[] for _ in range(max_stream_count)]

        for stream_i in range(max_stream_count):
            for level_i, checked in enumerate(self._level_select_checkboxes[stream_i]):
                if checked and level_i < len(orig['stream_contours'][stream_i]):
                    temp_contours[stream_i].append(orig['stream_contours'][stream_i][level_i])
                    temp_bps[stream_i].append(orig['stream_bounding_planes'][stream_i][level_i])

        self.contours = temp_contours
        self.bounding_planes = temp_bps
        self.draw_contour_stream = [[True] * len(temp_contours[s]) for s in range(max_stream_count)]

    # ── Fiber Animation ─────────────────────────────────────────────

    def _save_fiber_anim_data(self):
        """Save waypoints for replay."""
        if self.waypoints is None:
            return
        # Deep copy waypoints
        self._fiber_anim_waypoints = [
            [[np.array(wp).copy() for wp in level] for level in stream]
            for stream in self.waypoints
        ]

    def replay_fiber_animation(self):
        """Animate build fibers: mesh fades from 0.5 to 0.0, then fibers shown."""
        if self._fiber_anim_waypoints is None:
            self._build_fibers_replayed = True
            return

        # Restore waypoints from saved data
        self.waypoints = [
            [[np.array(wp).copy() for wp in level] for level in stream]
            for stream in self._fiber_anim_waypoints
        ]

        # Set contours/BPs to selected data (same as what build_fibers used)
        src = getattr(self, '_selected_stream_contours', None) or \
              (self.stream_contours if hasattr(self, 'stream_contours') and self.stream_contours is not None else None)
        src_bps = getattr(self, '_selected_stream_bounding_planes', None) or \
                  (self.stream_bounding_planes if hasattr(self, 'stream_bounding_planes') and self.stream_bounding_planes is not None else None)
        if src is not None:
            self.contours = src
            self.bounding_planes = src_bps
            self.draw_contour_stream = [[True] * len(src[s]) for s in range(len(src))]

        # Restore _stream_endpoints (cleared during defer)
        saved_ep = getattr(self, '_fiber_anim_stream_endpoints', None)
        if saved_ep is not None:
            self._stream_endpoints = saved_ep

        # Compute num_levels for growth phase
        num_levels = max(len(s) for s in self.waypoints) if self.waypoints else 1

        # Start state: mesh visible at 0.5 transparency, fibers hidden (grow from level 0)
        self.is_draw = True
        self.transparency = 0.5
        self.is_draw_fiber_architecture = True
        self._fiber_anim_progress = 0.0
        self._fiber_anim_level_progress = 0.0
        self._fiber_anim_num_levels = num_levels
        self._fiber_anim_active = True

    def update_fiber_animation(self, dt):
        """Phase 1: mesh fades 0.5→0.0. Phase 2: lines grow from origin to insertion."""
        if not self._fiber_anim_active:
            return False

        self._fiber_anim_progress += dt
        fade_dur = 1.0
        grow_dur = 2.0
        total_dur = fade_dur + grow_dur
        num_levels = getattr(self, '_fiber_anim_num_levels', 1)

        if self._fiber_anim_progress >= total_dur:
            # Done: mesh off, fibers fully drawn
            self.transparency = 0.0
            self.is_draw = False
            self.is_draw_fiber_architecture = True
            self._fiber_anim_active = False
            self._fiber_anim_level_progress = None
            self._build_fibers_replayed = True
            return False

        if self._fiber_anim_progress < fade_dur:
            # Phase 1: mesh transparency fade
            t = self._fiber_anim_progress / fade_dur
            t = t * t * (3.0 - 2.0 * t)
            self.transparency = 0.5 * (1.0 - t)
            if self.vertex_colors is not None and self.is_draw_scalar_field:
                self.vertex_colors[:, 3] = self.transparency
            self._fiber_anim_level_progress = 0.0
        else:
            # Phase 2: lines grow level by level
            self.transparency = 0.0
            self.is_draw = False
            if self.vertex_colors is not None and self.is_draw_scalar_field:
                self.vertex_colors[:, 3] = 0.0
            grow_t = (self._fiber_anim_progress - fade_dur) / grow_dur
            self._fiber_anim_level_progress = grow_t * max(num_levels - 1, 1)

        return True

    # ── Persistence ─────────────────────────────────────────────────

    def save_animation_state(self, filepath):
        """Save all processed state + animation replay data to a pickle file."""
        import pickle

        state = {}

        # Scalar field
        state['scalar_field'] = self.scalar_field
        state['vertex_colors'] = self.vertex_colors
        state['is_draw_scalar_field'] = self.is_draw_scalar_field
        state['_scalar_anim_target_colors'] = getattr(self, '_scalar_anim_target_colors', None)
        state['_scalar_anim_normalized_u'] = getattr(self, '_scalar_anim_normalized_u', None)

        # Contours and bounding planes (the final processed state)
        state['contours'] = self.contours
        state['bounding_planes'] = self.bounding_planes
        state['draw_contour_stream'] = getattr(self, 'draw_contour_stream', None)
        state['is_draw_contours'] = self.is_draw_contours
        state['is_draw_bounding_box'] = self.is_draw_bounding_box

        # Contour animation data
        state['_find_contours_count'] = getattr(self, '_find_contours_count', 0)

        # Fill gaps animation data
        state['_fill_gaps_inserted_indices'] = getattr(self, '_fill_gaps_inserted_indices', [])

        # Transitions animation data
        state['_transitions_inserted_indices'] = getattr(self, '_transitions_inserted_indices', [])

        # Smooth animation data (level-mode and stream-mode)
        state['_smooth_bp_before_level'] = getattr(self, '_smooth_bp_before_level', None)
        state['_smooth_bp_after_level'] = getattr(self, '_smooth_bp_after_level', None)
        state['_smooth_swing_data_level'] = getattr(self, '_smooth_swing_data_level', None)
        state['_smooth_twist_data_level'] = getattr(self, '_smooth_twist_data_level', None)
        state['_smooth_num_levels_level'] = getattr(self, '_smooth_num_levels_level', 0)
        state['_smooth_bp_before_stream'] = getattr(self, '_smooth_bp_before_stream', None)
        state['_smooth_bp_after_stream'] = getattr(self, '_smooth_bp_after_stream', None)
        state['_smooth_swing_data_stream'] = getattr(self, '_smooth_swing_data_stream', None)
        state['_smooth_twist_data_stream'] = getattr(self, '_smooth_twist_data_stream', None)
        state['_smooth_num_levels_stream'] = getattr(self, '_smooth_num_levels_stream', 0)
        # Also save current smooth data (for backward compat with files without level/stream split)
        state['_smooth_bp_before'] = getattr(self, '_smooth_bp_before', None)
        state['_smooth_bp_after'] = getattr(self, '_smooth_bp_after', None)
        state['_smooth_swing_data'] = getattr(self, '_smooth_swing_data', None)
        state['_smooth_twist_data'] = getattr(self, '_smooth_twist_data', None)
        state['_smooth_num_levels'] = getattr(self, '_smooth_num_levels', 0)

        # Pre-cut data (level-mode contours and BPs for deferred replay)
        state['_precut_contours'] = getattr(self, '_precut_contours', None)
        state['_precut_bounding_planes'] = getattr(self, '_precut_bounding_planes', None)
        state['_precut_draw_contour_stream'] = getattr(self, '_precut_draw_contour_stream', None)

        # Stream data (post-cut)
        state['stream_contours'] = getattr(self, 'stream_contours', None)
        state['stream_bounding_planes'] = getattr(self, 'stream_bounding_planes', None)
        state['stream_groups'] = getattr(self, 'stream_groups', None)

        # Cut animation data
        state['_cut_color_before'] = getattr(self, '_cut_color_before', None)
        state['_cut_color_after'] = getattr(self, '_cut_color_after', None)
        state['_cut_bp_before'] = getattr(self, '_cut_bp_before', None)
        state['_cut_bp_after'] = getattr(self, '_cut_bp_after', None)
        state['_cut_has_bp_change'] = getattr(self, '_cut_has_bp_change', False)
        state['_cut_bp_changed_levels'] = getattr(self, '_cut_bp_changed_levels', set())
        state['_cut_num_levels_before'] = getattr(self, '_cut_num_levels_before', 0)

        # Stream smooth animation data (post-cut)
        state['_stream_smooth_bp_before'] = getattr(self, '_stream_smooth_bp_before', None)
        state['_stream_smooth_bp_after'] = getattr(self, '_stream_smooth_bp_after', None)
        state['_stream_smooth_swing_data'] = getattr(self, '_stream_smooth_swing_data', None)
        state['_stream_smooth_twist_data'] = getattr(self, '_stream_smooth_twist_data', None)
        state['_stream_smooth_num_levels'] = getattr(self, '_stream_smooth_num_levels', 0)

        # Transparency
        state['transparency'] = getattr(self, 'transparency', 1.0)

        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Animation state saved to {filepath}")

    def load_animation_state(self, filepath):
        """Load processed state + animation replay data from a pickle file."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Scalar field
        self.scalar_field = state.get('scalar_field')
        self.vertex_colors = state.get('vertex_colors')
        self.is_draw_scalar_field = state.get('is_draw_scalar_field', False)
        self._scalar_anim_target_colors = state.get('_scalar_anim_target_colors')
        self._scalar_anim_normalized_u = state.get('_scalar_anim_normalized_u')

        # Contours and bounding planes
        self.contours = state.get('contours')
        self.bounding_planes = state.get('bounding_planes', [])
        self.draw_contour_stream = state.get('draw_contour_stream')
        self.is_draw_contours = state.get('is_draw_contours', False)
        self.is_draw_bounding_box = state.get('is_draw_bounding_box', False)

        # Contour animation data
        self._find_contours_count = state.get('_find_contours_count', 0)

        # Fill gaps animation data
        self._fill_gaps_inserted_indices = state.get('_fill_gaps_inserted_indices', [])

        # Transitions animation data
        self._transitions_inserted_indices = state.get('_transitions_inserted_indices', [])

        # Smooth animation data (level-mode and stream-mode)
        self._smooth_bp_before_level = state.get('_smooth_bp_before_level')
        self._smooth_bp_after_level = state.get('_smooth_bp_after_level')
        self._smooth_swing_data_level = state.get('_smooth_swing_data_level')
        self._smooth_twist_data_level = state.get('_smooth_twist_data_level')
        self._smooth_num_levels_level = state.get('_smooth_num_levels_level', 0)
        self._smooth_bp_before_stream = state.get('_smooth_bp_before_stream')
        self._smooth_bp_after_stream = state.get('_smooth_bp_after_stream')
        self._smooth_swing_data_stream = state.get('_smooth_swing_data_stream')
        self._smooth_twist_data_stream = state.get('_smooth_twist_data_stream')
        self._smooth_num_levels_stream = state.get('_smooth_num_levels_stream', 0)
        # Default smooth data (backward compat or current mode)
        self._smooth_bp_before = state.get('_smooth_bp_before')
        self._smooth_bp_after = state.get('_smooth_bp_after')
        self._smooth_swing_data = state.get('_smooth_swing_data')
        self._smooth_twist_data = state.get('_smooth_twist_data')
        self._smooth_num_levels = state.get('_smooth_num_levels', 0)

        # Pre-cut data (level-mode contours and BPs)
        self._precut_contours = state.get('_precut_contours')
        self._precut_bounding_planes = state.get('_precut_bounding_planes')
        self._precut_draw_contour_stream = state.get('_precut_draw_contour_stream')

        # Stream data (post-cut)
        self.stream_contours = state.get('stream_contours')
        self.stream_bounding_planes = state.get('stream_bounding_planes')
        if 'stream_groups' in state and state['stream_groups'] is not None:
            self.stream_groups = state['stream_groups']

        # Cut animation data
        self._cut_color_before = state.get('_cut_color_before')
        self._cut_color_after = state.get('_cut_color_after')
        self._cut_bp_before = state.get('_cut_bp_before')
        self._cut_bp_after = state.get('_cut_bp_after')
        # Recompute changed levels from stream_groups (levels where contours were split)
        bp_changed_levels = set()
        if hasattr(self, 'stream_groups') and self.stream_groups is not None:
            for level_i, groups in enumerate(self.stream_groups):
                for group in groups:
                    if len(group) > 1:
                        bp_changed_levels.add(level_i)
                        break
        self._cut_bp_changed_levels = bp_changed_levels
        self._cut_has_bp_change = len(bp_changed_levels) > 0
        self._cut_num_levels_before = state.get('_cut_num_levels_before', 0)

        # Stream smooth animation data (post-cut)
        self._stream_smooth_bp_before = state.get('_stream_smooth_bp_before')
        self._stream_smooth_bp_after = state.get('_stream_smooth_bp_after')
        self._stream_smooth_swing_data = state.get('_stream_smooth_swing_data')
        self._stream_smooth_twist_data = state.get('_stream_smooth_twist_data')
        self._stream_smooth_num_levels = state.get('_stream_smooth_num_levels', 0)

        # Transparency — save the processed value but reset to 1.0 for deferred replay
        # (smooth animation will fade it down during its replay)
        self._smooth_anim_orig_transparency = state.get('transparency', 1.0)
        self.transparency = 1.0
        # Reset alpha to 1.0 in all color arrays (they may have 0.5 baked in from smooth)
        if self.vertex_colors is not None:
            self.vertex_colors[:, 3] = 1.0
        if self._scalar_anim_target_colors is not None:
            self._scalar_anim_target_colors[:, 3] = 1.0

        # Reset all animation playback states (ready for replay)
        self._scalar_anim_active = False
        self._scalar_anim_progress = 0.0
        self._scalar_replayed = False
        self._contour_anim_active = False
        self._contour_anim_progress = 0.0
        self._contour_anim_original_indices = []
        self._contour_replayed = False
        self._fill_gaps_anim_active = False
        self._fill_gaps_anim_progress = 0.0
        self._fill_gaps_anim_step = 0
        self._fill_gaps_replayed = False
        self._transitions_anim_active = False
        self._transitions_anim_progress = 0.0
        self._transitions_anim_step = 0
        self._transitions_replayed = False
        self._smooth_anim_active = False
        self._smooth_anim_progress = 0.0
        self._smooth_replayed = False
        self._cut_anim_active = False
        self._cut_anim_progress = 0.0
        self._cut_replayed = False
        self._cut_anim_contour_colors = None
        self._stream_smooth_anim_active = False
        self._stream_smooth_anim_progress = 0.0
        self._stream_smooth_replayed = False

        # ── Restore deferred visual state so everything looks pre-processed ──

        # 1. Scalar: reset to default muscle color, hide scalar field
        if self._scalar_anim_target_colors is not None:
            self.is_draw_scalar_field = False
            n = len(self._scalar_anim_target_colors)
            default_color = getattr(self, 'color', [0.8, 0.8, 0.8])
            alpha = getattr(self, 'transparency', 1.0)
            self.vertex_colors = np.tile(
                np.array([default_color[0], default_color[1], default_color[2], alpha], dtype=np.float32),
                (n, 1)
            )

        # 2. Contours: restore pre-cut level-mode data if available
        if self._precut_contours is not None and self._cut_color_before is not None:
            # Restore level-mode contours and BPs for pre-cut animations
            self.contours = self._precut_contours
            self.bounding_planes = [[copy.deepcopy(bp) for bp in level] for level in self._precut_bounding_planes] if self._precut_bounding_planes else self.bounding_planes
            if self._precut_draw_contour_stream is not None:
                self.draw_contour_stream = list(self._precut_draw_contour_stream)
            # Use level-mode smooth data for pre-cut replay
            if self._smooth_bp_before_level is not None:
                self._smooth_bp_before = self._smooth_bp_before_level
                self._smooth_bp_after = self._smooth_bp_after_level
                self._smooth_swing_data = self._smooth_swing_data_level
                self._smooth_twist_data = self._smooth_twist_data_level
                self._smooth_num_levels = self._smooth_num_levels_level

        if self.contours is not None and len(self.contours) > 0:
            self.is_draw_contours = False
            self.is_draw_bounding_box = False
            # Set up visibility array matching current mode
            if (self.draw_contour_stream is not None
                    and len(self.draw_contour_stream) > 0
                    and isinstance(self.draw_contour_stream[0], list)):
                num_streams = len(self.draw_contour_stream)
                num_levels = len(self.draw_contour_stream[0])
                self.draw_contour_stream = [[False] * num_levels for _ in range(num_streams)]
            else:
                self.draw_contour_stream = [False] * len(self.contours)

        # 3. Smooth: apply pre-smooth BPs
        if self._smooth_bp_before is not None:
            self._apply_bp_snapshot(self._smooth_bp_before)

        # 4. Cut: handle deferred state based on mode
        if self._precut_contours is not None:
            # New: level mode — no color override needed (normal coloring is correct)
            self._cut_anim_contour_colors = None
        else:
            # Backward compat: stream mode — use color override for pre-cut appearance
            if self._cut_color_before is not None:
                self._cut_anim_contour_colors = [[c.copy() for c in stream] for stream in self._cut_color_before]
            if self._cut_bp_before is not None and getattr(self, '_cut_has_bp_change', False):
                if not hasattr(self, 'stream_bounding_planes') or self.stream_bounding_planes is None:
                    self.stream_bounding_planes = self.bounding_planes
                self._apply_stream_bp_snapshot(self._cut_bp_before)

        print(f"Animation state loaded from {filepath}")
