"""Zygote mesh UI — extracted from viewer.py.

Three free functions that take the viewer instance (v) as first argument.
"""
import imgui
import numpy as np
import os
import traceback

# UI dimension constants (mirrored from viewer.py to avoid circular imports)
wide_button_width = 308
button_width = 150
MIN_EYE_DISTANCE = 0.5


def draw_zygote_ui(v):
    """Top-level Zygote tree node with scrollable child region."""
    if imgui.tree_node("Zygote", imgui.TREE_NODE_DEFAULT_OPEN):
        # Scrollable child region for Zygote menu
        imgui.begin_child("ZygoteScroll", width=0, height=500, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        draw_zygote_muscle_ui(v)
        draw_zygote_skeleton_ui(v)
        imgui.end_child()  # End ZygoteScroll
        imgui.tree_pop()


def draw_zygote_muscle_ui(v):
    """Muscle section inside the Zygote tree node."""
    if imgui.tree_node("Muscle", imgui.TREE_NODE_DEFAULT_OPEN):
        changed, v.is_draw_zygote_muscle = imgui.checkbox("Draw", v.is_draw_zygote_muscle)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw = v.is_draw_zygote_muscle
        changed, v.is_draw_zygote_muscle_open_edges = imgui.checkbox("Draw Open Edges", v.is_draw_zygote_muscle_open_edges)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw_open_edges = v.is_draw_zygote_muscle_open_edges

        _, v.is_draw_one_zygote_muscle = imgui.checkbox("Draw One Muscle", v.is_draw_one_zygote_muscle)
        changed, v.zygote_muscle_color = imgui.color_edit3("Color", *v.zygote_muscle_color)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.color = v.zygote_muscle_color

        changed, v.zygote_muscle_transparency = imgui.slider_float("Transparency", v.zygote_muscle_transparency, 0.0, 1.0)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.transparency = v.zygote_muscle_transparency
                if obj.vertex_colors is not None:
                    obj.vertex_colors[:, 3] = obj.transparency

        # Muscle Add/Remove UI
        if imgui.tree_node("Add/Remove Muscles"):
            imgui.text("Available:")
            # Calculate total available count
            total_available = sum(len(muscles) for muscles in v.available_muscle_by_category.values())

            if total_available > 0:
                # Draw category-based listbox using child region
                imgui.begin_child("##available_muscles_child", width=0, height=150, border=True)

                for category, muscles in v.available_muscle_by_category.items():
                    if len(muscles) == 0:
                        continue

                    # Category header with expand/collapse
                    is_expanded = v.available_category_expanded.get(category, False)
                    arrow = "v" if is_expanded else ">"
                    category_label = f"{arrow} {category} ({len(muscles)})"

                    # Make category clickable
                    if imgui.selectable(category_label, False)[0]:
                        v.available_category_expanded[category] = not is_expanded

                    # Show muscles if expanded
                    if is_expanded:
                        for name, path in muscles:
                            # Indent muscle names
                            imgui.indent(15)
                            is_selected = (v.available_selected_muscle == name)
                            clicked, _ = imgui.selectable(f"  {name}", is_selected)
                            if clicked:
                                v.available_selected_category = category
                                v.available_selected_muscle = name
                            # Double-click to add
                            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                                v.add_muscle_mesh(name, path)
                            imgui.unindent(15)

                imgui.end_child()
            else:
                imgui.text("(none)")

            # Arrow buttons for add/remove
            if imgui.button("Add", width=button_width):
                if v.available_selected_muscle and v.available_selected_category:
                    # Find the path for selected muscle
                    for name, path in v.available_muscle_by_category.get(v.available_selected_category, []):
                        if name == v.available_selected_muscle:
                            v.add_muscle_mesh(name, path)
                            break
            imgui.same_line()
            if imgui.button("Remove", width=button_width):
                if len(v.zygote_muscle_meshes) > 0:
                    loaded_names = list(v.zygote_muscle_meshes.keys())
                    name = loaded_names[v.loaded_muscle_selected]
                    v.remove_muscle_mesh(name)

            imgui.text("Loaded:")
            loaded_names = list(v.zygote_muscle_meshes.keys())
            if len(loaded_names) > 0:
                # Use child region with selectables for double-click support
                imgui.begin_child("##loaded_muscles_child", width=0, height=150, border=True)
                for i, name in enumerate(loaded_names):
                    is_selected = (v.loaded_muscle_selected == i)
                    clicked, _ = imgui.selectable(name, is_selected)
                    if clicked:
                        v.loaded_muscle_selected = i
                    # Double-click to remove
                    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                        v.remove_muscle_mesh(name)
                imgui.end_child()
            else:
                imgui.text("(none)")

            # Bulk add/remove buttons
            imgui.separator()

            if imgui.button("Remove All", width=button_width):
                v.remove_all_muscles()

            # Add L/R muscles by group
            groups = v.get_available_muscle_groups()
            if len(groups) > 0:
                imgui.text("Add by group:")
                for group in groups:
                    short_name = group[:8] if len(group) > 8 else group
                    if imgui.button(f"L {short_name}##L{group}", width=73):
                        v.add_muscles_by_group(group, "L_")
                    imgui.same_line()
                    if imgui.button(f"R {short_name}##R{group}", width=73):
                        v.add_muscles_by_group(group, "R_")

            imgui.tree_pop()

        if imgui.tree_node("Activation levels"):
            if v.env.zygote_activation_levels is not None:
                # Use shorter slider width to leave room for name
                slider_width = 120
                for i, (name, obj) in enumerate(v.env.muscle_info.items()):
                    # Bounds check for zygote_activation_levels
                    if i >= len(v.env.zygote_activation_levels):
                        continue
                    # Show name first (truncate if too long)
                    display_name = name[:25] + "..." if len(name) > 25 else name
                    imgui.text(f"{display_name}")
                    imgui.same_line(position=180)
                    imgui.push_item_width(slider_width)
                    changed, v.env.zygote_activation_levels[i] = imgui.slider_float(f"##zygote_act{i}", v.env.zygote_activation_levels[i], 0.0, 1.0)
                    imgui.pop_item_width()
                    if changed:
                        # Bounds check for activation indices
                        if i + 1 < len(v.env.zygote_activation_indices):
                            start_fiber = v.env.zygote_activation_indices[i]
                            end_fiber = v.env.zygote_activation_indices[i + 1]
                            if end_fiber <= len(v.env.muscle_activation_levels):
                                v.env.muscle_activation_levels[start_fiber:end_fiber] = v.env.zygote_activation_levels[i]
            imgui.tree_pop()

        # Motion Browser
        if v.env.skel is not None and imgui.tree_node("Motion Browser", imgui.TREE_NODE_DEFAULT_OPEN):
            v._draw_motion_browser_ui()
            imgui.tree_pop()

        # Skeleton joint angle sliders
        if v.env.skel is not None and imgui.tree_node("Skeleton Joint Angles"):
            # Initialize DOF array if not present
            if not hasattr(v, '_skel_dofs'):
                v._skel_dofs = v.env.skel.getPositions().copy()
                v._skel_dof_names = []
                # Build DOF names from joints
                for jn_idx in range(v.env.skel.getNumJoints()):
                    joint = v.env.skel.getJoint(jn_idx)
                    jn_name = joint.getName()
                    num_dofs = joint.getNumDofs()
                    if num_dofs == 1:
                        v._skel_dof_names.append(jn_name)
                    elif num_dofs == 3:
                        v._skel_dof_names.extend([f"{jn_name}_x", f"{jn_name}_y", f"{jn_name}_z"])
                    elif num_dofs == 6:
                        v._skel_dof_names.extend([f"{jn_name}_tx", f"{jn_name}_ty", f"{jn_name}_tz",
                                                    f"{jn_name}_rx", f"{jn_name}_ry", f"{jn_name}_rz"])
                    else:
                        for d in range(num_dofs):
                            v._skel_dof_names.append(f"{jn_name}_{d}")

            # Sync DOF array size with skeleton
            num_dofs = v.env.skel.getNumDofs()
            if len(v._skel_dofs) != num_dofs:
                v._skel_dofs = v.env.skel.getPositions().copy()

            # Reset all button
            if imgui.button("Reset All##skel_dofs", width=100):
                v._skel_dofs = np.zeros(num_dofs)
                v.env.skel.setPositions(v._skel_dofs)
                # Update soft bodies and waypoints
                for mname, mobj in v.zygote_muscle_meshes.items():
                    if mobj.soft_body is not None:
                        mobj._update_tet_positions_from_skeleton(v.env.skel)
                        mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)
            imgui.same_line()
            if imgui.button("Sync from Skel##skel_dofs", width=120):
                v._skel_dofs = v.env.skel.getPositions().copy()

            # Sliders for each DOF
            label_width = 140
            slider_width = 120
            reset_btn_width = 22
            any_changed = False
            for i in range(num_dofs):
                # Get DOF name (truncate if too long)
                dof_name = v._skel_dof_names[i] if i < len(v._skel_dof_names) else f"DOF {i}"
                display_name = dof_name[:15] if len(dof_name) > 15 else dof_name
                imgui.text(f"{i:2d} {display_name:<15}")
                imgui.same_line(position=label_width)
                imgui.push_item_width(slider_width)
                changed, v._skel_dofs[i] = imgui.slider_float(f"##skel_dof{i}", v._skel_dofs[i], -3.14, 3.14)
                imgui.pop_item_width()
                if changed:
                    any_changed = True
                imgui.same_line()
                if imgui.button(f"0##reset_dof{i}", width=reset_btn_width):
                    v._skel_dofs[i] = 0.0
                    any_changed = True

            # Apply changes to skeleton
            if any_changed:
                v.env.skel.setPositions(v._skel_dofs)
                # Update soft bodies and waypoints
                for mname, mobj in v.zygote_muscle_meshes.items():
                    if mobj.soft_body is not None:
                        mobj._update_tet_positions_from_skeleton(v.env.skel)
                        mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)

            imgui.tree_pop()

        if imgui.button("Export Muscle Waypoints", width=wide_button_width):
            from core.dartHelper import exportMuscleWaypoints
            exportMuscleWaypoints(v.zygote_muscle_meshes, list(v.zygote_skeleton_meshes.keys()))
        if imgui.button("Import zygote_muscle", width=wide_button_width):
            muscle_file = "data/zygote_muscle.xml"
            if not os.path.exists(muscle_file):
                print(f"Error: Muscle file not found: {muscle_file}")
                print("  Run 'Export Muscle Waypoints' first to create it.")
            else:
                try:
                    v.env.muscle_info = v.env.saveZygoteMuscleInfo(muscle_file)
                    if not v.env.muscle_info:
                        print("No muscles loaded from file (empty or invalid)")
                    else:
                        v.env.loading_zygote_muscle_info(v.env.muscle_info)
                        v.env.muscle_activation_levels = np.zeros(v.env.muscles.getNumMuscles())

                        v.draw_obj = True
                        # Disable skeleton drawing when importing muscle waypoints
                        v.is_draw_zygote_skeleton = False
                        for name, obj in v.zygote_skeleton_meshes.items():
                            obj.is_draw = False
                        print(f"Imported {v.env.muscles.getNumMuscles()} muscles from {muscle_file}")
                except Exception as e:
                    print(f"Error importing muscle waypoints: {e}")

        # Load all tet meshes and init soft bodies
        if imgui.button("Load All Tets", width=wide_button_width):
            load_count = 0
            init_count = 0
            already_init_count = 0
            for mname, mobj in v.zygote_muscle_meshes.items():
                try:
                    # If tet already loaded, just check if soft body needs init
                    if mobj.tet_vertices is not None:
                        if mobj.soft_body is None:
                            # Tet loaded but soft body not initialized - init it
                            skeleton_names = list(v.zygote_skeleton_meshes.keys())
                            mobj.resolve_skeleton_attachments(skeleton_names)
                            mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                            if mobj.soft_body is not None:
                                init_count += 1
                        else:
                            already_init_count += 1
                        continue
                    # Load new tet
                    mobj.soft_body = None  # Reset soft body when loading new tet
                    if mobj.load_tetrahedron_mesh(mname):
                        if mobj.tet_vertices is not None:
                            mobj.is_draw = False  # Disable mesh draw
                            mobj.is_draw_contours = False
                            mobj.is_draw_tet_mesh = True
                            mobj.is_draw_fiber_architecture = True  # Enable fiber draw
                            load_count += 1
                            # Resolve skeleton attachments from names to current indices
                            skeleton_names = list(v.zygote_skeleton_meshes.keys())
                            mobj.resolve_skeleton_attachments(skeleton_names)
                            # Also init soft body
                            mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                            if mobj.soft_body is not None:
                                init_count += 1
                except Exception as e:
                    print(f"[{mname}] Load Tet error: {e}")
            print(f"Loaded {load_count} new tets, initialized {init_count} soft bodies ({already_init_count} already initialized)")
        # Run soft body simulation for all muscles at once
        if imgui.button("Run All Tet Sim", width=wide_button_width):
            count = 0
            collision_count = 0
            for mname, mobj in v.zygote_muscle_meshes.items():
                if mobj.tet_vertices is not None:
                    if mobj.soft_body is None:
                        mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                    if mobj.soft_body is not None:
                        # Respect each muscle's individual collision setting
                        iterations, residual = mobj.run_soft_body_to_convergence(
                            v.zygote_skeleton_meshes,
                            v.env.skel,
                            max_iterations=100,
                            tolerance=1e-4,
                            enable_collision=mobj.soft_body_collision,
                            collision_margin=mobj.soft_body_collision_margin,
                            verbose=False,
                            use_arap=mobj.use_arap
                        )
                        count += 1
                        if mobj.soft_body_collision:
                            collision_count += 1
            print(f"Ran tet sim for {count} muscles ({collision_count} with collision)")

        # Inter-muscle constraints section
        imgui.separator()
        imgui.text("Inter-Muscle Constraints")

        # Threshold slider
        imgui.push_item_width(100)
        changed, v.inter_muscle_constraint_threshold = imgui.slider_float(
            "Threshold (m)", v.inter_muscle_constraint_threshold, 0.001, 0.05
        )
        imgui.pop_item_width()
        imgui.same_line()
        imgui.text(f"({v.inter_muscle_constraint_threshold*100:.1f}cm)")

        # Find constraints button
        if imgui.button("Find Constraints", width=wide_button_width):
            count = v.find_inter_muscle_constraints()

        imgui.text(f"{len(v.inter_muscle_constraints)} constraints")
        _, v.draw_inter_muscle_constraints = imgui.checkbox(
            "Draw##inter_constraints", getattr(v, 'draw_inter_muscle_constraints', False)
        )

        # Unified volume checkbox
        _, v.coupled_as_unified_volume = imgui.checkbox(
            "Unified Volume", v.coupled_as_unified_volume
        )
        if v.coupled_as_unified_volume:
            imgui.same_line()
            imgui.text_colored("(all muscles as one system)", 0.5, 0.8, 0.5)

        # Backend selection (radio-button style with checkboxes)
        imgui.text("Backend:")
        imgui.same_line()

        # CPU (always available, default)
        use_cpu = not v.use_gpu_arap and not v.use_taichi_arap
        if imgui.checkbox("CPU", use_cpu)[1] and not use_cpu:
            v.use_gpu_arap = False
            v.use_taichi_arap = False

        # GPU (PyTorch)
        if v.gpu_available:
            imgui.same_line()
            changed, checked = imgui.checkbox("GPU", v.use_gpu_arap)
            if changed and checked:
                v.use_gpu_arap = True
                v.use_taichi_arap = False
            elif changed and not checked:
                v.use_gpu_arap = False

        # Taichi
        if v.taichi_available:
            imgui.same_line()
            changed, checked = imgui.checkbox("Taichi", v.use_taichi_arap)
            if changed and checked:
                v.use_taichi_arap = True
                v.use_gpu_arap = False
            elif changed and not checked:
                v.use_taichi_arap = False

        # Run coupled simulation button
        if imgui.button("Run Coupled Tet Sim", width=wide_button_width):
            v.run_all_tet_sim_with_constraints()

        imgui.separator()

        for name, obj in v.zygote_muscle_meshes.items():
            if imgui.tree_node(name):
                # Two-column layout: "Process All" button on left, individual buttons on right
                imgui.columns(2, f"cols##{name}", border=False)
                imgui.set_column_width(0, 120)

                # Left column: Process button with vertical slider
                num_process_buttons = 12  # Match number of buttons on right
                process_all_height = num_process_buttons * imgui.get_frame_height() + (num_process_buttons - 1) * imgui.get_style().item_spacing[1]

                # Initialize process step slider value
                if not hasattr(obj, '_process_step'):
                    obj._process_step = 12

                # Vertical slider for step selection (top=1, bottom=12)
                # v_slider_int format: label, width, height, value, min, max
                # To get 1 at top and 12 at bottom, we invert: display (13 - value)
                display_step = 13 - obj._process_step  # Convert for display
                changed, new_display = imgui.v_slider_int(
                    f"##step{name}", 20, process_all_height, display_step, 1, 12)
                if changed:
                    obj._process_step = 13 - new_display  # Convert back
                imgui.same_line()

                # Step names matching button order (1=top, 12=bottom)
                # 1:Scalar, 2:Contours, 3:FillGap, 4:Transitions, 5:Smooth, 6:Cut, 7:StreamSmooth, 8:Select, 9:Build, 10:Resample, 11:Mesh, 12:Tet
                step_names = ['', 'Scalar', 'Contours', 'Fill Gap', 'Transitions', 'Smooth', 'Cut', 'StreamSmooth', 'Select', 'Build', 'Resample', 'Mesh', 'Tet']

                # Check if pipeline is paused waiting for manual step
                pipeline_paused = hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None

                # Button label changes if paused
                if pipeline_paused:
                    btn_label = f"Resume\nfrom {obj._pipeline_paused_at}\n({step_names[obj._pipeline_paused_at]})"
                else:
                    btn_label = f"Process\n1 to {obj._process_step}\n({step_names[obj._process_step]})"

                if imgui.button(f"{btn_label}##{name}", width=75, height=process_all_height):
                    try:
                        max_step = obj._process_step
                        # If resuming, start from paused step
                        if pipeline_paused:
                            start_step = obj._pipeline_paused_at
                            obj._pipeline_paused_at = None
                            print(f"[{name}] Resuming pipeline from step {start_step} to {max_step}...")
                        else:
                            start_step = 1
                            print(f"[{name}] Running pipeline steps 1 to {max_step}...")

                        # Step 1: Scalar Field
                        if start_step <= 1 <= max_step and len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                            print(f"  [1/{max_step}] Computing Scalar Field...")
                            obj.compute_scalar_field()

                        # Step 2: Find Contours
                        if start_step <= 2 <= max_step and obj.scalar_field is not None:
                            print(f"  [2/{max_step}] Finding Contours...")
                            obj.find_contours(skeleton_meshes=v.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale)
                            obj.is_draw_bounding_box = True

                        # Step 3: Fill Gaps
                        if start_step <= 3 <= max_step and obj.contours is not None and len(obj.contours) > 0:
                            print(f"  [3/{max_step}] Filling Gaps...")
                            obj.refine_contours(max_spacing_threshold=0.01)

                        # Step 4: Find Transitions
                        if start_step <= 4 <= max_step and obj.scalar_field is not None:
                            print(f"  [4/{max_step}] Finding Transitions...")
                            field_min = float(obj.scalar_field.min())
                            field_max = float(obj.scalar_field.max())
                            scalar_min, scalar_max = field_min, field_max
                            if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                                o_val, i_val = obj.origin_contour_value, obj.insertion_contour_value
                                if o_val != i_val:
                                    proposed_min, proposed_max = min(o_val, i_val), max(o_val, i_val)
                                    if proposed_min >= field_min and proposed_max <= field_max:
                                        scalar_min, scalar_max = proposed_min, proposed_max
                            exp_origin = len(obj.contours[0]) if obj.contours and len(obj.contours) > 0 else None
                            exp_insertion = len(obj.contours[-1]) if obj.contours and len(obj.contours) > 0 else None
                            obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                                    expected_origin=exp_origin, expected_insertion=exp_insertion)
                            if obj.contours is not None and len(obj.contours) > 0:
                                obj.add_transitions_to_contours()

                        # Step 5: Smooth (z, x, bp - before cut)
                        if start_step <= 5 <= max_step and obj.contours is not None and len(obj.contours) > 0:
                            print(f"  [5/{max_step}] Smoothening (z, x, bp)...")
                            obj.smoothen_contours_z()
                            obj.smoothen_contours_x()
                            obj.smoothen_contours_bp()

                        # Step 6: Cut
                        if start_step <= 6 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                            print(f"  [6/{max_step}] Cutting streams...")
                            obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name)
                            # Check if waiting for manual cut
                            if hasattr(obj, '_manual_cut_pending') and obj._manual_cut_pending or hasattr(obj, '_manual_cut_data') and obj._manual_cut_data is not None:
                                obj._pipeline_paused_at = 7  # Resume from step 7 after cut is complete
                                print(f"  [6/{max_step}] Waiting for manual cut - pipeline paused")
                                raise StopIteration("Manual cut pending")

                        # Step 7: Stream Smooth (z, x, bp - after cut)
                        if start_step <= 7 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                            print(f"  [7/{max_step}] Stream Smoothening (z, x, bp)...")
                            obj.smoothen_contours_z()
                            obj.smoothen_contours_x()
                            obj.smoothen_contours_bp()

                        # Step 8: Contour Select
                        if start_step <= 8 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                            print(f"  [8/{max_step}] Selecting contours...")
                            obj.select_levels()
                            # Check if waiting for manual level selection
                            if hasattr(obj, '_level_select_window_open') and obj._level_select_window_open:
                                obj._pipeline_paused_at = 9  # Resume from step 9 after selection
                                print(f"  [8/{max_step}] Waiting for level selection - pipeline paused")
                                raise StopIteration("Level selection pending")

                        # Step 9: Build Fiber
                        if start_step <= 9 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                            print(f"  [9/{max_step}] Building fibers...")
                            obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes)

                        # Step 10: Resample Contours
                        if start_step <= 10 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                            print(f"  [10/{max_step}] Resampling Contours...")
                            obj.resample_contours(base_samples=32)

                        # Step 11: Build Contour Mesh
                        if start_step <= 11 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                            print(f"  [11/{max_step}] Building Contour Mesh...")
                            obj.build_contour_mesh()

                        # Step 12: Tetrahedralize
                        if start_step <= 12 <= max_step and obj.contour_mesh_vertices is not None:
                            print(f"  [12/{max_step}] Tetrahedralizing...")
                            obj.soft_body = None
                            obj.tetrahedralize_contour_mesh()
                            if obj.tet_vertices is not None:
                                obj.is_draw_contours = False
                                obj.is_draw_tet_mesh = True

                        obj._pipeline_paused_at = None  # Clear pause state on completion
                        print(f"[{name}] Pipeline complete (steps {start_step}-{max_step})!")
                    except StopIteration:
                        pass  # Manual cut pending - pipeline paused gracefully
                    except Exception as e:
                        print(f"[{name}] Pipeline error: {e}")
                        traceback.print_exc()

                # Right column: Individual buttons (use -1 to auto-fill column width)
                imgui.next_column()
                col_button_width = 180  # Fits in the right column

                # Helper for button coloring based on process step
                def colored_button(label, step_num, width):
                    will_run = step_num <= obj._process_step
                    if will_run:
                        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
                        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.7, 0.3, 1.0)
                    clicked = imgui.button(label, width=width)
                    if will_run:
                        imgui.pop_style_color(2)
                    return clicked

                if colored_button(f"Scalar Field##{name}", 1, col_button_width):
                    if len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                        try:
                            obj.compute_scalar_field()
                        except Exception as e:
                            print(f"[{name}] Scalar Field error: {e}")
                    else:
                        print(f"[{name}] Need edge_groups and edge_classes")

                if colored_button(f"Find Contours##{name}", 2, col_button_width):
                    if obj.scalar_field is not None:
                        try:
                            obj.find_contours(skeleton_meshes=v.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale)
                            obj.is_draw_bounding_box = True
                        except Exception as e:
                            print(f"[{name}] Find Contours error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Scalar Field' first")
                if colored_button(f"Fill Gaps##{name}", 3, col_button_width):
                    if obj.contours is not None and len(obj.contours) > 0:
                        try:
                            obj.refine_contours(max_spacing_threshold=0.01)
                        except Exception as e:
                            print(f"[{name}] Fill Gaps error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                # Find Transitions button - fast scan for contour count changes (step 4)
                if colored_button(f"Find Transitions##{name}", 4, col_button_width):
                    if hasattr(obj, 'scalar_field') and obj.scalar_field is not None:
                        try:
                            # Use actual scalar field range
                            field_min = float(obj.scalar_field.min())
                            field_max = float(obj.scalar_field.max())
                            scalar_min = field_min
                            scalar_max = field_max
                            # Optionally narrow to origin/insertion if set AND within field range
                            if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                                o_val = obj.origin_contour_value
                                i_val = obj.insertion_contour_value
                                # Only use if different AND within the actual scalar field range
                                if o_val != i_val:
                                    proposed_min = min(o_val, i_val)
                                    proposed_max = max(o_val, i_val)
                                    if proposed_min >= field_min and proposed_max <= field_max:
                                        scalar_min = proposed_min
                                        scalar_max = proposed_max
                                    else:
                                        print(f"[{name}] Origin/insertion values ({o_val}, {i_val}) outside field range [{field_min:.4f}, {field_max:.4f}], using field range")
                            print(f"[{name}] Scanning scalar range: {scalar_min:.4f} to {scalar_max:.4f}")
                            # Try to get expected origin/insertion counts from existing contours
                            exp_origin = None
                            exp_insertion = None
                            if hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0:
                                exp_origin = len(obj.contours[0])
                                exp_insertion = len(obj.contours[-1])
                                print(f"[{name}] Expected counts from contours: origin={exp_origin}, insertion={exp_insertion}")
                            obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                                    expected_origin=exp_origin, expected_insertion=exp_insertion)
                            # Auto-add transitions to contours if contours exist
                            if obj.contours is not None and len(obj.contours) > 0:
                                obj.add_transitions_to_contours()
                        except Exception as e:
                            print(f"[{name}] Find Transitions error: {e}")
                            traceback.print_exc()
                    else:
                        print(f"[{name}] Prerequisites: Run 'Scalar Field' first")

                # Step 5: Smoothen buttons: z, x, bp (3 buttons in same row - before cut)
                sub_button_width = (col_button_width - 8) // 3  # 3 buttons with small margins
                if colored_button(f"z##{name}", 5, sub_button_width):
                    if obj.contours is not None and len(obj.contours) > 0:
                        try:
                            obj.smoothen_contours_z()
                        except Exception as e:
                            print(f"[{name}] Smoothen Z error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                imgui.same_line(spacing=4)
                if colored_button(f"x##{name}", 5, sub_button_width):
                    if obj.contours is not None and len(obj.contours) > 0:
                        try:
                            obj.smoothen_contours_x()
                        except Exception as e:
                            print(f"[{name}] Smoothen X error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                imgui.same_line(spacing=4)
                if colored_button(f"bp##{name}", 5, sub_button_width):
                    if obj.contours is not None and len(obj.contours) > 0:
                        try:
                            obj.smoothen_contours_bp()
                        except Exception as e:
                            print(f"[{name}] Smoothen BP error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                # Step 6: Cut (standalone button)
                if colored_button(f"Cut##{name}", 6, col_button_width):
                    if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None and len(obj.bounding_planes) > 0:
                        try:
                            obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name)
                        except Exception as e:
                            print(f"[{name}] Cut Streams error: {e}")
                            traceback.print_exc()
                    else:
                        print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                # Step 7: Stream Smoothen buttons: z, x, bp (3 buttons in same row - after cut)
                if colored_button(f"z##stream{name}", 7, sub_button_width):
                    if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                        try:
                            obj.smoothen_contours_z()
                        except Exception as e:
                            print(f"[{name}] Stream Smoothen Z error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Cut' first")
                imgui.same_line(spacing=4)
                if colored_button(f"x##stream{name}", 7, sub_button_width):
                    if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                        try:
                            obj.smoothen_contours_x()
                        except Exception as e:
                            print(f"[{name}] Stream Smoothen X error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Cut' first")
                imgui.same_line(spacing=4)
                if colored_button(f"bp##stream{name}", 7, sub_button_width):
                    if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                        try:
                            obj.smoothen_contours_bp()
                        except Exception as e:
                            print(f"[{name}] Stream Smoothen BP error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Cut' first")

                # Step 8: Contour Select (standalone button)
                if colored_button(f"Contour Select##{name}", 8, col_button_width):
                    if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                        try:
                            obj.select_levels()
                        except Exception as e:
                            print(f"[{name}] Select Levels error: {e}")
                            traceback.print_exc()
                    else:
                        print(f"[{name}] Prerequisites: Run 'Cut' first")

                # Step 9: Build Fiber (standalone button)
                if colored_button(f"Build Fiber##{name}", 9, col_button_width):
                    if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                        try:
                            obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes)
                        except Exception as e:
                            print(f"[{name}] Build Fibers error: {e}")
                            traceback.print_exc()
                    else:
                        print(f"[{name}] Prerequisites: Run 'Cut' first")

                # Step 10: Resample Contours
                if colored_button(f"Resample Contours##{name}", 10, col_button_width):
                    if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                        try:
                            obj.resample_contours(base_samples=32)
                        except Exception as e:
                            print(f"[{name}] Resample Contours error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Smoothen Contours' first")

                # Step 11: Build Contour Mesh
                if colored_button(f"Build Contour Mesh##{name}", 11, col_button_width):
                    if obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                        try:
                            obj.build_contour_mesh()
                        except Exception as e:
                            print(f"[{name}] Build Contour Mesh error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Build Fiber' first")

                # Step 12: Tetrahedralize
                if colored_button(f"Tetrahedralize##{name}", 12, col_button_width):
                    if obj.contour_mesh_vertices is not None:
                        try:
                            obj.soft_body = None  # Reset soft body when re-tetrahedralizing
                            obj.tetrahedralize_contour_mesh()
                            if obj.tet_vertices is not None:
                                obj.is_draw_contours = False
                                obj.is_draw_tet_mesh = True
                        except Exception as e:
                            print(f"[{name}] Tetrahedralize error: {e}")
                    else:
                        print(f"[{name}] Prerequisites: Run 'Build Contour Mesh' first")

                # End two-column layout - back to full width for remaining GUI elements
                imgui.columns(1)

                # Reset process button
                reset_width = button_width * 2 + imgui.get_style().item_spacing[0]
                if imgui.button(f"Reset Process##{name}", width=reset_width):
                    obj.reset_process()

                # Save/Load contours buttons
                contour_filepath = f"{v.zygote_muscle_dir}{name}.contours.json"
                if imgui.button(f"Save Contour##{name}", width=button_width):
                    if obj.contours is not None and len(obj.contours) > 0:
                        try:
                            obj.save_contours(contour_filepath)
                        except Exception as e:
                            print(f"[{name}] Save Contours error: {e}")
                    else:
                        print(f"[{name}] No contours to save")
                imgui.same_line()
                if imgui.button(f"Load Contour##{name}", width=button_width):
                    try:
                        obj.load_contours(contour_filepath)
                    except Exception as e:
                        print(f"[{name}] Load Contours error: {e}")

                if imgui.button(f"Save Tet##{name}", width=button_width):
                    if hasattr(obj, 'tet_vertices') and obj.tet_vertices is not None:
                        try:
                            obj.save_tetrahedron_mesh(name)
                        except Exception as e:
                            print(f"[{name}] Save Tet error: {e}")
                    else:
                        print(f"[{name}] No tetrahedron mesh to save")
                imgui.same_line()
                if imgui.button(f"Load Tet##{name}", width=button_width):
                    try:
                        obj.soft_body = None  # Reset soft body when loading new tet
                        obj.load_tetrahedron_mesh(name)
                        if obj.tet_vertices is not None:
                            obj.is_draw = False  # Disable mesh draw
                            obj.is_draw_contours = False
                            obj.is_draw_tet_mesh = True
                            obj.is_draw_fiber_architecture = True  # Enable fiber draw
                            # Resolve skeleton attachments from names to current indices
                            skeleton_names = list(v.zygote_skeleton_meshes.keys())
                            obj.resolve_skeleton_attachments(skeleton_names)
                    except Exception as e:
                        print(f"[{name}] Load Tet error: {e}")

                # Inspect 2D button - opens visualization window for contours (and fiber samples if available)
                inspect_width = button_width * 2 + imgui.get_style().item_spacing[0]
                has_contour_data = (hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0)
                if not has_contour_data:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                if imgui.button(f"Inspect 2D##{name}", width=inspect_width):
                    if has_contour_data:
                        v.inspect_2d_open[name] = True
                        if name not in v.inspect_2d_stream_idx:
                            v.inspect_2d_stream_idx[name] = 0
                        if name not in v.inspect_2d_contour_idx:
                            v.inspect_2d_contour_idx[name] = 0
                    else:
                        print(f"[{name}] No contour data. Run 'Find Contours' first.")
                if not has_contour_data:
                    imgui.pop_style_var()

                # Focus camera on muscle button
                if imgui.button(f"Focus##{name}", width=inspect_width):
                    if obj.vertices is not None and len(obj.vertices) > 0:
                        # Compute bounding box center
                        min_pt = np.min(obj.vertices, axis=0)
                        max_pt = np.max(obj.vertices, axis=0)
                        center = (min_pt + max_pt) / 2
                        bbox_size = np.linalg.norm(max_pt - min_pt)
                        # Set camera target (trans is scaled by 0.001 in render, so multiply by 1000)
                        v.trans = -center * 1000.0
                        # Adjust eye distance based on bounding box size
                        distance = bbox_size * 2.0
                        eye_dir = v.eye / (np.linalg.norm(v.eye) + 1e-10)
                        v.eye = eye_dir * max(distance, MIN_EYE_DISTANCE)
                    else:
                        print(f"[{name}] No vertices to focus on")

                _, obj.is_one_fiber = imgui.checkbox(f"One Fiber##{name}", obj.is_one_fiber)

                # Bounding box method selector
                bbox_methods = ['farthest_vertex', 'pca', 'bbox']
                current_method = getattr(obj, 'bounding_box_method', 'farthest_vertex')
                current_idx = bbox_methods.index(current_method) if current_method in bbox_methods else 0
                changed, new_idx = imgui.combo(f"BBox Method##{name}", current_idx, bbox_methods)
                if changed:
                    obj.bounding_box_method = bbox_methods[new_idx]

                # Contour spacing scale slider (lower = more contours)
                changed, obj.contour_spacing_scale = imgui.slider_float(
                    f"Spacing##{name}", obj.contour_spacing_scale, 0.1, 2.0, "%.2f")

                # Min level distance for Select Levels (as % of muscle length)
                if not hasattr(obj, 'min_level_distance'):
                    obj.min_level_distance = 0.01  # Default 1%
                changed, obj.min_level_distance = imgui.slider_float(
                    f"Min Dist##{name}", obj.min_level_distance, 0.001, 0.1, "%.3f")

                # Error threshold for level selection (as % of muscle length)
                if not hasattr(obj, 'level_select_error_threshold'):
                    obj.level_select_error_threshold = 0.005  # Default 0.5%
                changed, obj.level_select_error_threshold = imgui.slider_float(
                    f"Err Thresh##{name}", obj.level_select_error_threshold, 0.001, 0.05, "%.3f")

                # Sampling method selector
                sampling_methods = ['sobol_unit_square', 'sobol_min_contour']
                current_idx = sampling_methods.index(obj.sampling_method) if obj.sampling_method in sampling_methods else 0
                changed, new_idx = imgui.combo(f"Sampling##{name}", current_idx, sampling_methods)
                if changed:
                    obj.sampling_method = sampling_methods[new_idx]

                # Cutting method selector
                cutting_methods = ['bp', 'area_based', 'voronoi', 'angular', 'gradient', 'ratio', 'cumulative_area', 'projected_area']
                current_cut_idx = cutting_methods.index(obj.cutting_method) if obj.cutting_method in cutting_methods else 0
                changed, new_cut_idx = imgui.combo(f"Cutting##{name}", current_cut_idx, cutting_methods)
                if changed:
                    obj.cutting_method = cutting_methods[new_cut_idx]

                imgui.same_line()
                imgui.text(obj.link_mode)
                changed1, obj.specific_contour_value = imgui.slider_float(f"Ori##{name}", obj.specific_contour_value, 1.0, obj.contour_value_min, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                changed2, obj.specific_contour_value = imgui.slider_float(f"Mid##{name}", obj.specific_contour_value, obj.contour_value_min, obj.contour_value_max, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                changed3, obj.specific_contour_value = imgui.slider_float(f"Ins##{name}", obj.specific_contour_value, obj.contour_value_max, 10.0, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                if imgui.tree_node(f"MinMax##{name}"):
                    _, obj.contour_value_min = imgui.input_float(f"Min##{name}", obj.contour_value_min)
                    _, obj.contour_value_max = imgui.input_float(f"Max##{name}", obj.contour_value_max)
                    imgui.tree_pop()
                if changed1 or changed2 or changed3:
                    obj.find_contour_with_value(obj.specific_contour_value)
                # if imgui.button(f"Find Value Contour##{name}"):
                #     obj.find_contour_with_value()
                if imgui.button(f"Switch Link Mode##{name}"):
                    if obj.link_mode == 'mean':
                        obj.link_mode = 'vertex'
                    else:
                        obj.link_mode = 'mean'

                # Minimum contour distance threshold
                if hasattr(obj, 'min_contour_distance'):
                    _, obj.min_contour_distance = imgui.slider_float(
                        f"Min Dist##{name}", obj.min_contour_distance, 0.001, 0.02, "%.3f")

                # if imgui.button("Find Interesecting Bones"):
                #     for other_name, other_obj in v.zygote_muscle_meshes.items():
                #         other_obj.is_draw = False
                #     obj.is_draw = True

                #     intersecting_meshes = obj.find_intersections(v.zygote_skeleton_meshes)
                #     # print(bb_intersect)
                #     for skel_name, skel_obj in v.zygote_skeleton_meshes.items():
                #         if skel_name in intersecting_meshes:
                #             skel_obj.color = np.array([0.0, 0.0, 1.0])
                #         else:
                #             skel_obj.color = np.array([0.9, 0.9, 0.9])

                #     v.zygote_muscle_meshes_intersection_bones[name] = intersecting_meshes

                changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
                if changed and obj.vertex_colors is not None:
                    obj.vertex_colors[:, 3] = obj.transparency

                if imgui.tree_node("Edge Classes"):
                    for i in range(len(obj.edge_classes)):
                        # Fixed width: "insertion" is 9 chars, pad "origin" to match
                        label = f"{obj.edge_classes[i]:9s}"
                        imgui.text(label)
                        imgui.same_line()
                        if imgui.button(f"Flip class##{name}_{i}"):
                            obj.edge_classes[i] = 'insertion' if obj.edge_classes[i] == 'origin' else 'origin'
                    imgui.tree_pop()
                if obj.draw_contour_stream is not None:
                    if imgui.tree_node("Contour Stream"):
                        if imgui.button("All Stream Off"):
                            for i in range(len(obj.draw_contour_stream)):
                                obj.draw_contour_stream[i] = False
                        imgui.same_line()
                        if imgui.button(f"Auto Detect##{name}"):
                            obj.auto_detect_attachments(v.zygote_skeleton_meshes)
                        # Ensure attach_skeletons arrays are properly sized
                        num_streams = len(obj.draw_contour_stream)
                        while len(obj.attach_skeletons) < num_streams:
                            obj.attach_skeletons.append([0, 0])
                        while len(obj.attach_skeletons_sub) < num_streams:
                            obj.attach_skeletons_sub.append([0, 0])
                        for i in range(num_streams):
                            _, obj.draw_contour_stream[i] = imgui.checkbox(f"Stream {i}", obj.draw_contour_stream[i])
                            imgui.push_item_width(100)
                            changed, obj.attach_skeletons[i][0] = imgui.input_int(f"Origin##{name}_stream{i}_origin", obj.attach_skeletons[i][0])
                            if changed:
                                if obj.attach_skeletons[i][0] < 0:
                                    obj.attach_skeletons[i][0] = 0
                                elif obj.attach_skeletons[i][0] > len(v.zygote_skeleton_meshes) - 1:
                                    obj.attach_skeletons[i][0] = len(v.zygote_skeleton_meshes) - 1
                            changed, obj.attach_skeletons_sub[i][0] = imgui.input_int(f"Subpart##{name}_stream{i}_origin_sub", obj.attach_skeletons_sub[i][0])
                            if changed:
                                if obj.attach_skeletons_sub[i][0] < 0:
                                    obj.attach_skeletons_sub[i][0] = 0
                                elif obj.attach_skeletons_sub[i][0] > 1:
                                    obj.attach_skeletons_sub[i][0] = 1

                            imgui.text(list(v.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][0]] + f"{obj.attach_skeletons_sub[i][0]}")
                            changed, obj.attach_skeletons[i][1] = imgui.input_int(f"Insertion##{name}_stream{i}_insertion", obj.attach_skeletons[i][1])
                            if changed:
                                if obj.attach_skeletons[i][1] < 0:
                                    obj.attach_skeletons[i][1] = 0
                                elif obj.attach_skeletons[i][1] > len(v.zygote_skeleton_meshes) - 1:
                                    obj.attach_skeletons[i][1] = len(v.zygote_skeleton_meshes) - 1
                            changed, obj.attach_skeletons_sub[i][1] = imgui.input_int(f"Subpart##{name}_stream{i}_insertion_sub", obj.attach_skeletons_sub[i][1])
                            if changed:
                                if obj.attach_skeletons_sub[i][1] < 0:
                                    obj.attach_skeletons_sub[i][1] = 0
                                elif obj.attach_skeletons_sub[i][1] > 1:
                                    obj.attach_skeletons_sub[i][1] = 1
                            imgui.text(list(v.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][1]] + f"{obj.attach_skeletons_sub[i][1]}")
                            imgui.pop_item_width()

                            # if imgui.button(f"Print Contour points##{i}"):
                            #     print(f"Print {i}th contour stream")
                            #     for j, contour in enumerate(obj.contours[i]):
                            #         print(f"Contour {j}")
                            #         for v in contour:
                            #             print(v)
                            #         print()
                        imgui.tree_pop()

                changed, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
                if changed and obj.is_draw and v.is_draw_one_zygote_muscle:
                    for other_name, other_obj in v.zygote_muscle_meshes.items():
                        other_obj.is_draw = False
                    obj.is_draw = True
                _, obj.is_draw_open_edges = imgui.checkbox("Draw Open Edges", obj.is_draw_open_edges)
                _, obj.is_draw_scalar_field = imgui.checkbox("Draw Scalar Field", obj.is_draw_scalar_field)
                _, obj.is_draw_contours = imgui.checkbox("Draw Contours", obj.is_draw_contours)
                imgui.same_line()
                _, obj.is_draw_contour_vertices = imgui.checkbox("Vertices", obj.is_draw_contour_vertices)
                imgui.same_line()
                _, obj.is_draw_farthest_pair = imgui.checkbox("Farthest Pair", obj.is_draw_farthest_pair)
                _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                _, obj.is_draw_centroid = imgui.checkbox("Draw Centroid", obj.is_draw_centroid)
                _, obj.is_draw_bounding_box = imgui.checkbox("Draw Bounding Box", obj.is_draw_bounding_box)
                _, obj.is_draw_discarded = imgui.checkbox("Draw Discarded", obj.is_draw_discarded)
                _, obj.is_draw_fiber_architecture = imgui.checkbox("Draw Fiber Architecture", obj.is_draw_fiber_architecture)
                _, obj.is_draw_contour_mesh = imgui.checkbox("Draw Contour Mesh", obj.is_draw_contour_mesh)
                _, obj.is_draw_tet_mesh = imgui.checkbox("Draw Tet Mesh", obj.is_draw_tet_mesh)
                imgui.same_line()
                _, obj.is_draw_tet_edges = imgui.checkbox("Tet Edges", obj.is_draw_tet_edges)
                imgui.same_line()
                _, obj.is_draw_constraints = imgui.checkbox("Constraints", obj.is_draw_constraints)

                # Soft body simulation controls (quasistatic, on-demand)
                if imgui.tree_node(f"Soft Body##{name}"):
                    # Initialize soft body if tet mesh exists but soft body doesn't
                    if obj.soft_body is None and obj.tet_vertices is not None:
                        if imgui.button(f"Init Soft Body##{name}", width=wide_button_width):
                            try:
                                obj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                            except Exception as e:
                                print(f"[{name}] Init Soft Body error: {e}")
                    elif obj.tet_vertices is not None:
                        # Parameters: Volume is primary (muscle), Edge is secondary
                        _, obj.soft_body_volume_stiffness = imgui.slider_float(f"Volume##soft_{name}", obj.soft_body_volume_stiffness, 0.1, 1.0)
                        _, obj.soft_body_stiffness = imgui.slider_float(f"Edge##soft_{name}", obj.soft_body_stiffness, 0.0, 1.0)
                        _, obj.soft_body_damping = imgui.slider_float(f"Damping##soft_{name}", obj.soft_body_damping, 0.0, 0.95)
                        _, obj.soft_body_collision = imgui.checkbox(f"Collision##soft_{name}", obj.soft_body_collision)
                        _, obj.waypoints_from_tet_sim = imgui.checkbox(f"Update Waypoints##soft_{name}", obj.waypoints_from_tet_sim)
                        if obj.soft_body_collision:
                            imgui.same_line()
                            imgui.push_item_width(80)
                            _, obj.soft_body_collision_margin = imgui.slider_float(f"Margin##soft_{name}", obj.soft_body_collision_margin, 0.001, 0.02)
                            imgui.pop_item_width()

                        # Run simulation button
                        if imgui.button(f"Run Tet Sim##{name}", width=wide_button_width):
                            try:
                                iterations, residual = obj.run_soft_body_to_convergence(
                                    v.zygote_skeleton_meshes,
                                    v.env.skel,
                                    max_iterations=100,
                                    tolerance=1e-4,
                                    enable_collision=obj.soft_body_collision,
                                    collision_margin=obj.soft_body_collision_margin,
                                    use_arap=obj.use_arap
                                )
                                print(f"{name}: Converged in {iterations} iterations, residual={residual:.2e} (ARAP={obj.use_arap})")
                            except Exception as e:
                                print(f"[{name}] Run Tet Sim error: {e}")

                        # Test button to verify deformation works
                        if imgui.button(f"Test Deform##{name}", width=wide_button_width):
                            try:
                                if obj.soft_body is not None and obj.soft_body.fixed_targets is not None:
                                    test_offset = np.array([0.02, 0.0, 0.0])  # 2cm in X
                                    obj.soft_body.fixed_targets += test_offset
                                    iterations, residual = obj.soft_body.solve_to_convergence(100, 1e-4)
                                    obj.tet_vertices = obj.soft_body.get_positions().astype(np.float32)
                                    obj._prepare_tet_draw_arrays()
                                    print(f"{name}: Test deform - {iterations} iters, residual={residual:.2e}")
                            except Exception as e:
                                print(f"[{name}] Test Deform error: {e}")

                        if imgui.button(f"Reset Soft Body##{name}", width=wide_button_width):
                            try:
                                obj.reset_soft_body()
                            except Exception as e:
                                print(f"[{name}] Reset Soft Body error: {e}")

                    imgui.tree_pop()

                # VIPER rod simulation controls
                if imgui.tree_node(f"VIPER Rods##{name}"):
                    if obj.viper_available:
                        # Initialize VIPER if waypoints exist but VIPER doesn't
                        if obj.viper_sim is None and len(obj.waypoints) > 0:
                            if imgui.button(f"Init VIPER##{name}", width=wide_button_width):
                                try:
                                    obj.init_viper(v.zygote_skeleton_meshes, v.env.skel)
                                except Exception as e:
                                    print(f"[{name}] Init VIPER error: {e}")
                        elif obj.viper_sim is not None:
                            # VIPER parameters
                            changed, obj.viper_sim.stretch_stiffness = imgui.slider_float(
                                f"Stretch##viper_{name}", obj.viper_sim.stretch_stiffness, 0.1, 1.0)
                            changed, obj.viper_sim.volume_stiffness = imgui.slider_float(
                                f"Volume##viper_{name}", obj.viper_sim.volume_stiffness, 0.1, 1.0)
                            changed, obj.viper_sim.bend_stiffness = imgui.slider_float(
                                f"Bend##viper_{name}", obj.viper_sim.bend_stiffness, 0.1, 1.0)
                            changed, obj.viper_sim.damping = imgui.slider_float(
                                f"Damping##viper_{name}", obj.viper_sim.damping, 0.8, 0.999)
                            changed, obj.viper_sim.iterations = imgui.slider_int(
                                f"Iterations##viper_{name}", obj.viper_sim.iterations, 1, 50)

                            # Volume preservation toggle (VIPER key feature)
                            changed, obj.viper_sim.enable_volume_constraint = imgui.checkbox(
                                f"Volume Preserve##viper_{name}", obj.viper_sim.enable_volume_constraint)

                            # Collision toggle
                            _, obj.viper_sim.enable_collision = imgui.checkbox(
                                f"Collision##viper_{name}", obj.viper_sim.enable_collision)
                            if obj.viper_sim.enable_collision:
                                imgui.same_line()
                                imgui.push_item_width(80)
                                _, obj.viper_sim.collision_margin = imgui.slider_float(
                                    f"Margin##viper_col_{name}", obj.viper_sim.collision_margin, 0.001, 0.01, "%.3f")
                                imgui.pop_item_width()

                            imgui.separator()
                            imgui.text("Visualization:")
                            # Visualization controls
                            _, obj.is_draw_viper = imgui.checkbox(
                                f"Draw Rods##viper_{name}", obj.is_draw_viper)
                            imgui.same_line()
                            _, obj.viper_only_mode = imgui.checkbox(
                                f"VIPER Only##viper_{name}", obj.viper_only_mode)
                            _, obj.is_draw_viper_tubes = imgui.checkbox(
                                f"Tube Mode##viper_{name}", obj.is_draw_viper_tubes)
                            # Show VIPER rod-based mesh (built from rods)
                            _, obj.is_draw_viper_rod_mesh = imgui.checkbox(
                                f"Rod Mesh##viper_{name}", getattr(obj, 'is_draw_viper_rod_mesh', False))
                            imgui.push_item_width(100)
                            _, obj.viper_rod_radius = imgui.slider_float(
                                f"Rod Radius##viper_{name}", obj.viper_rod_radius, 0.001, 0.01, "%.3f")
                            _, obj.viper_point_size = imgui.slider_float(
                                f"Point Size##viper_{name}", obj.viper_point_size, 2.0, 15.0)
                            imgui.pop_item_width()

                            imgui.separator()
                            # Show rod info
                            if hasattr(obj, 'viper_waypoints') and obj.viper_waypoints:
                                num_rods = len(obj.viper_waypoints)
                                total_pts = sum(len(rod) for rod in obj.viper_waypoints)
                            else:
                                num_rods = 0
                                total_pts = 0
                            imgui.text(f"Rods: {num_rods}, Points: {total_pts}")

                            # Run VIPER simulation button
                            if imgui.button(f"Run VIPER##{name}", width=wide_button_width):
                                try:
                                    skel = v.env.skel if hasattr(v, 'env') and v.env is not None else None
                                    skel_meshes = v.zygote_skeleton_meshes if obj.viper_sim.enable_collision else None
                                    iterations = obj.run_viper_to_convergence(max_iterations=100, tolerance=1e-5, skeleton=skel, skeleton_meshes=skel_meshes)
                                    print(f"{name}: VIPER converged in {iterations} iterations")
                                except Exception as e:
                                    print(f"[{name}] Run VIPER error: {e}")

                            # Single step button
                            if imgui.button(f"VIPER Step##{name}", width=wide_button_width):
                                try:
                                    skel = v.env.skel if hasattr(v, 'env') and v.env is not None else None
                                    skel_meshes = v.zygote_skeleton_meshes if obj.viper_sim.enable_collision else None
                                    max_disp = obj.run_viper_step(skeleton=skel, skeleton_meshes=skel_meshes)
                                    print(f"{name}: VIPER step, max_disp={max_disp:.2e}")
                                except Exception as e:
                                    print(f"[{name}] VIPER Step error: {e}")

                            # Reset button
                            if imgui.button(f"Reset VIPER##{name}", width=wide_button_width):
                                try:
                                    obj.reset_viper()
                                    print(f"{name}: VIPER reset")
                                except Exception as e:
                                    print(f"[{name}] Reset VIPER error: {e}")

                            # Debug button - show constraint residuals
                            if imgui.button(f"Debug Info##{name}", width=wide_button_width):
                                try:
                                    from viewer.viper_rods import get_viper_backend
                                    info = obj.viper_sim.get_debug_info()
                                    residuals = obj.viper_sim.get_constraint_residuals()
                                    backend = get_viper_backend()
                                    print(f"\n=== {name} VIPER Debug ===")
                                    print(f"  Backend: {backend.upper() if backend else 'Not initialized'} (use_gpu={info.get('use_gpu', False)})")
                                    print(f"  Rods: {info.get('num_rods', 0)}, Verts/rod: {info.get('num_vertices_per_rod', 0)}")
                                    print(f"  Scale: min={info.get('min_scale', 0):.3f}, max={info.get('max_scale', 0):.3f}, avg={info.get('avg_scale', 0):.3f}")
                                    print(f"  Stretch error: mean={residuals.get('stretch_error_mean', 0):.4f}, max={residuals.get('stretch_error_max', 0):.4f}")
                                    print(f"  Volume error: mean={residuals.get('volume_error_mean', 0):.4f}, max={residuals.get('volume_error_max', 0):.4f}")
                                    print(f"  Bend error: mean={residuals.get('bend_error_mean', 0):.4f}, max={residuals.get('bend_error_max', 0):.4f}")
                                    print(f"  Collision: {obj.viper_sim.enable_collision} (margin={obj.viper_sim.collision_margin:.3f})")
                                except Exception as e:
                                    print(f"[{name}] Debug error: {e}")
                        else:
                            imgui.text("Generate waypoints first")
                    else:
                        imgui.text("VIPER requires Taichi")
                    imgui.tree_pop()

                if imgui.button("Export Muscle Waypoints", width=wide_button_width):
                    pass
                    from core.dartHelper import exportMuscleWaypoints
                    exportMuscleWaypoints(v.zygote_muscle_meshes, list(v.zygote_skeleton_meshes.keys()))
                if imgui.button("Import zygote_muscle", width=wide_button_width):
                    muscle_file = "data/zygote_muscle.xml"
                    if not os.path.exists(muscle_file):
                        print(f"Error: Muscle file not found: {muscle_file}")
                        print("  Run 'Export Muscle Waypoints' first to create it.")
                    else:
                        try:
                            v.env.muscle_info = v.env.saveZygoteMuscleInfo(muscle_file)
                            if not v.env.muscle_info:
                                print("No muscles loaded from file (empty or invalid)")
                            else:
                                v.env.loading_zygote_muscle_info(v.env.muscle_info)
                                v.env.muscle_activation_levels = np.zeros(v.env.muscles.getNumMuscles())

                                v.draw_obj = True
                                # Disable skeleton drawing when importing muscle waypoints
                                v.is_draw_zygote_skeleton = False
                                for sname, sobj in v.zygote_skeleton_meshes.items():
                                    sobj.is_draw = False
                                print(f"Imported {v.env.muscles.getNumMuscles()} muscles from {muscle_file}")
                        except Exception as e:
                            print(f"Error importing muscle waypoints: {e}")

                # End column layout
                imgui.columns(1)
                imgui.tree_pop()
        imgui.tree_pop()


def draw_zygote_skeleton_ui(v):
    """Skeleton section inside the Zygote tree node."""
    if imgui.tree_node("Skeleton"):
        changed, v.is_draw_zygote_skeleton = imgui.checkbox("Draw", v.is_draw_zygote_skeleton)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.is_draw = v.is_draw_zygote_skeleton
        _, v.is_draw_one_zygote_skeleton = imgui.checkbox("Draw One Skeleton", v.is_draw_one_zygote_skeleton)
        changed, v.zygote_skeleton_color = imgui.color_edit3("Color", *v.zygote_skeleton_color)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.color = v.zygote_skeleton_color
        if imgui.button("Draw All"):
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.is_draw = True
        changed, v.zygote_skeleton_transparency = imgui.slider_float("Transparency##Skeleton", v.zygote_skeleton_transparency, 0.0, 1.0)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.transparency = v.zygote_skeleton_transparency

        for i, (name, obj) in enumerate(v.zygote_skeleton_meshes.items()):
            if imgui.tree_node(f"{i}: {name}"):
                changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
                if changed and obj.vertex_colors is not None:
                    obj.vertex_colors[:, 3] = obj.transparency
                _, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
                _, obj.is_draw_corners = imgui.checkbox("Draw Corners", obj.is_draw_corners)
                _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                _, obj.is_contact = imgui.checkbox("Contact", obj.is_contact)
                if obj.is_draw and v.is_draw_one_zygote_skeleton:
                    for other_name, other_obj in v.zygote_skeleton_meshes.items():
                        other_obj.is_draw = False
                        obj.is_draw = True
                imgui.tree_pop()

                # Auto checkbox and num boxes slider
                _, obj.auto_num_boxes = imgui.checkbox(f"Auto##{name}_auto", obj.auto_num_boxes)
                imgui.same_line()
                if obj.auto_num_boxes:
                    imgui.text(f"Boxes: {obj.num_boxes} (auto)")
                else:
                    imgui.push_item_width(100)
                    _, obj.num_boxes = imgui.slider_int(f"Boxes##{name}", obj.num_boxes, 1, 10)
                    imgui.pop_item_width()

                # Axis alignment checkboxes
                imgui.text("Align:")
                imgui.same_line()
                _, obj.bb_align_x = imgui.checkbox(f"X##{name}_bbx", obj.bb_align_x)
                imgui.same_line()
                _, obj.bb_align_y = imgui.checkbox(f"Y##{name}_bby", obj.bb_align_y)
                imgui.same_line()
                _, obj.bb_align_z = imgui.checkbox(f"Z##{name}_bbz", obj.bb_align_z)
                imgui.same_line()
                _, obj.bb_enforce_symmetry = imgui.checkbox(f"Sym##{name}_sym", obj.bb_enforce_symmetry)

                # Build axis string from checkboxes
                axis_str = ''
                if obj.bb_align_x:
                    axis_str += 'x'
                if obj.bb_align_y:
                    axis_str += 'y'
                if obj.bb_align_z:
                    axis_str += 'z'
                axis_param = axis_str if axis_str else None

                # Show current alignment
                align_label = axis_str.upper() if axis_str else "PCA"
                sym_label = "+Sym" if obj.bb_enforce_symmetry else ""
                auto_label = "+Auto" if obj.auto_num_boxes else ""
                if imgui.button(f"Find BB ({align_label}{sym_label}{auto_label})##{name}", width=140):
                    obj.find_bounding_box(axis=axis_param, symmetry=obj.bb_enforce_symmetry)

                imgui.text(f"Parent: {obj.parent_name}")
                if imgui.button(f"<##{name+'_parent'}"):
                    obj.cand_parent_index -= 1
                    if obj.cand_parent_index < 0:
                        obj.cand_parent_index = len(v.zygote_skeleton_meshes) - 1
                imgui.same_line()
                if imgui.button(f">##{name+'_parent'}"):
                    obj.cand_parent_index += 1
                    if obj.cand_parent_index >= len(v.zygote_skeleton_meshes):
                        obj.cand_parent_index = 0
                imgui.same_line()
                cand_name = list(v.zygote_skeleton_meshes.keys())[obj.cand_parent_index]
                imgui.push_item_width(100)
                changed, obj.cand_parent_index = imgui.input_int(f"Parent##{name}", obj.cand_parent_index)
                imgui.pop_item_width()
                if changed:
                    if obj.cand_parent_index > len(v.zygote_skeleton_meshes) - 1:
                        obj.cand_parent_index = len(v.zygote_skeleton_meshes) - 1
                    elif obj.cand_parent_index < 0:
                        obj.cand_parent_index = 0
                imgui.text("%3d: %s   " % (obj.cand_parent_index, cand_name))

                if imgui.button(f"Set as root##{name}"):
                    for other_name, other_obj in v.zygote_skeleton_meshes.items():
                        other_obj.is_root = False
                    obj.is_root = True
                    print(f"{name} set as root")
                if imgui.button(f"Connect to parent##{name}"):

                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            if not name in parent_mesh.children_names:
                                parent_mesh.children_names.append(name)
                            print(f"{name} connected to {cand_name}")
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = False
                imgui.same_line()
                if imgui.button(f"Connect to parent as Weld##{name}"):
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            parent_mesh.children_names.append(name)
                            print(f"{name} connected to {cand_name} as Weld")
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = True

                # Revolute joint connection buttons
                if imgui.button(f"Revolute (Knee)##{name}"):
                    # Knee: bends backward, axis = x
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            if name not in parent_mesh.children_names:
                                parent_mesh.children_names.append(name)
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = False
                            obj.is_revolute = True
                            obj.bends_backward = True  # Knee bends backward
                            obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                            obj.revolute_lower = 0.0
                            obj.revolute_upper = 2.5
                            print(f"{name} connected to {cand_name} as Revolute (Knee, bends backward)")

                imgui.same_line()
                if imgui.button(f"Revolute (Elbow)##{name}"):
                    # Elbow/Finger/Toe: bends forward, axis = x
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            if name not in parent_mesh.children_names:
                                parent_mesh.children_names.append(name)
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = False
                            obj.is_revolute = True
                            obj.bends_backward = False  # Elbow/finger/toe bends forward
                            obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                            obj.revolute_lower = -2.5
                            obj.revolute_upper = 0.0
                            print(f"{name} connected to {cand_name} as Revolute (Elbow/Finger/Toe, bends forward)")

                if imgui.button(f"Revolute (Finger/Toe)##{name}"):
                    # Finger/Toe: smaller range than elbow
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            if name not in parent_mesh.children_names:
                                parent_mesh.children_names.append(name)
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = False
                            obj.is_revolute = True
                            obj.bends_backward = False
                            obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                            obj.revolute_lower = -1.57  # ~90 degrees for fingers/toes
                            obj.revolute_upper = 0.3    # Slight hyperextension allowed
                            print(f"{name} connected to {cand_name} as Revolute (Finger/Toe)")

                # Show revolute joint settings if connected as revolute
                if obj.is_revolute:
                    imgui.text(f"Revolute: {'Knee (backward)' if obj.bends_backward else 'Elbow (forward)'}")
                    imgui.push_item_width(80)
                    changed, obj.revolute_lower = imgui.input_float(f"Lower##{name}_rev", obj.revolute_lower, 0.1)
                    imgui.same_line()
                    changed, obj.revolute_upper = imgui.input_float(f"Upper##{name}_rev", obj.revolute_upper, 0.1)
                    imgui.pop_item_width()

                if imgui.button("Export Bounding boxes", width=wide_button_width):
                    from core.dartHelper import exportBoundingBoxes
                    exportBoundingBoxes(v.zygote_skeleton_meshes)
                if imgui.button("Import zygote_skel", width=wide_button_width):
                    if v.env.skel is not None:
                        v.env.world.removeSkeleton(v.env.skel)

                    from core.dartHelper import saveSkeletonInfo
                    from core.dartHelper import buildFromInfo
                    skel_info, root_name, _, _, _, _ = saveSkeletonInfo("data/zygote_skel.xml")
                    v.env.skel_info = skel_info
                    v.env.skel = buildFromInfo(skel_info, "zygote")
                    v.env.world.addSkeleton(v.env.skel)
                    v.env.kp = 300.0 * np.ones(v.env.skel.getNumDofs())
                    v.env.kv = 20.0 * np.ones(v.env.skel.getNumDofs())
                    v.env.kp[:6] = 0.0
                    v.env.kv[:6] = 0.0
                    v.env.num_action = len(v.env.get_zero_action()) * (3 if v.env.learning_gain else 1)
                    v.motion_skel = v.env.skel.clone()

        imgui.tree_pop()
