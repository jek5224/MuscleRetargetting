#!/usr/bin/env python3
"""Swap between original mesh and contour mesh display for baked results.

Usage:
    python tools/swap_mesh_mode.py original   # Switch to original mesh display
    python tools/swap_mesh_mode.py contour    # Switch to contour mesh display
    python tools/swap_mesh_mode.py status     # Show current mode
"""
import os
import sys
import shutil

MUSCLES_WITH_ORIG = [
    'L_Vastus_Lateralis',
    # Add more muscles as original mesh tets become available
]

TET_DIR = 'tet'
ORIG_DIR = 'tet_orig_open'
ORIG_CACHE = 'data/motion_cache/walk/layered_coll_indep'
CONTOUR_CACHE = 'data/motion_cache/walk/layered_contour'
ACTIVE_CACHE = 'data/motion_cache/walk/layered'  # Viewer loads from here


def get_status():
    """Check current mode for each muscle."""
    import pickle
    for name in MUSCLES_WITH_ORIG:
        tet_path = os.path.join(TET_DIR, f'{name}_tet.npz')
        backup_path = tet_path + '.contour_backup'
        if not os.path.exists(tet_path):
            print(f'  {name}: tet file missing')
            continue
        with open(tet_path, 'rb') as f:
            d = pickle.load(f)
        n_verts = len(d['vertices'])
        is_orig = 'cap_vertex_types' in d
        has_backup = os.path.exists(backup_path)
        print(f'  {name}: {n_verts} verts, mode={"ORIGINAL" if is_orig else "CONTOUR"}, backup={"yes" if has_backup else "no"}')


def swap_to_original():
    """Swap to original mesh display + cache."""
    for name in MUSCLES_WITH_ORIG:
        tet_path = os.path.join(TET_DIR, f'{name}_tet.npz')
        backup_path = tet_path + '.contour_backup'
        orig_path = os.path.join(ORIG_DIR, f'{name}_tet.npz')

        if not os.path.exists(orig_path):
            print(f'  {name}: no original mesh, skipping')
            continue

        # Backup contour tet if not already done
        if not os.path.exists(backup_path) and os.path.exists(tet_path):
            shutil.copy2(tet_path, backup_path)

        shutil.copy2(orig_path, tet_path)
        print(f'  {name}: tet → ORIGINAL mesh')

    # Symlink cache
    if os.path.exists(ORIG_CACHE):
        if os.path.islink(ACTIVE_CACHE):
            os.unlink(ACTIVE_CACHE)
        elif os.path.isdir(ACTIVE_CACHE):
            # Backup if it's a real directory
            if not os.path.exists(ACTIVE_CACHE + '.bak'):
                os.rename(ACTIVE_CACHE, ACTIVE_CACHE + '.bak')
            else:
                shutil.rmtree(ACTIVE_CACHE)
        os.symlink(os.path.abspath(ORIG_CACHE), ACTIVE_CACHE)
        print(f'  cache → {ORIG_CACHE}')


def swap_to_contour():
    """Swap to contour mesh display + cache."""
    for name in MUSCLES_WITH_ORIG:
        tet_path = os.path.join(TET_DIR, f'{name}_tet.npz')
        backup_path = tet_path + '.contour_backup'

        if os.path.exists(backup_path):
            shutil.copy2(backup_path, tet_path)
            print(f'  {name}: tet → CONTOUR mesh')
        else:
            print(f'  {name}: no contour backup, skipping')

    # Symlink cache
    if os.path.exists(CONTOUR_CACHE):
        if os.path.islink(ACTIVE_CACHE):
            os.unlink(ACTIVE_CACHE)
        elif os.path.isdir(ACTIVE_CACHE):
            if not os.path.exists(ACTIVE_CACHE + '.bak'):
                os.rename(ACTIVE_CACHE, ACTIVE_CACHE + '.bak')
            else:
                shutil.rmtree(ACTIVE_CACHE)
        os.symlink(os.path.abspath(CONTOUR_CACHE), ACTIVE_CACHE)
        print(f'  cache → {CONTOUR_CACHE}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/swap_mesh_mode.py [original|contour|status]')
        return

    mode = sys.argv[1].lower()
    if mode == 'original':
        print('Swapping to ORIGINAL mesh display:')
        swap_to_original()
        print('\nNote: bake cache must have original mesh positions.')
        print('Run: python tools/bake_layered.py --tet-dir tet_hybrid ...')
    elif mode == 'contour':
        print('Swapping to CONTOUR mesh display:')
        swap_to_contour()
        print('\nNote: bake cache must have contour mesh positions.')
        print('Run: python tools/bake_layered.py --tet-dir tet ...')
    elif mode == 'status':
        print('Current mesh mode:')
        get_status()
    else:
        print(f'Unknown mode: {mode}')
        print('Usage: python tools/swap_mesh_mode.py [original|contour|status]')


if __name__ == '__main__':
    main()
