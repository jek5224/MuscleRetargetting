# IPC backend options

Checked on 2026-07-28. No package was installed and the environment was not
modified.

## Current environment

- The project interpreter is CPython 3.8 (`pyMAC/bin/python`).
- `uipc`, `pyuipc`, `ipctk`, and `igl` do not import.
- `pip list` contains no IPC or libigl binding.
- The repository has no vendored `ipc-toolkit`, libigl, or libuipc source and
  no IPC entry in a requirements/Conda/CMake manifest.
- There is no `conda` executable. CMake is available.
- Existing scripts (`tools/bake_pyuipc.py`,
  `tools/bake_ipc_phase2.py`, and related files) target a separately
  configured A6000 environment. They are not a locally callable backend.

## Options

### IPC Toolkit Python bindings (`ipctk`)

The official IPC Toolkit documentation supports `pip install ipctk`, a
source install, or a CMake build with `IPC_TOOLKIT_BUILD_PYTHON=ON`:
https://ipctk.xyz/build/python.html

This is the most plausible CPU geometry/barrier library to integrate with the
existing Python constitutive solver. Compatibility with this project's Python
3.8 ABI must be checked before installation. No local wheel or source checkout
exists, so using it requires network/package-install permission.

### PyUIPC (`pyuipc`, imported as `uipc`)

PyPI now publishes Linux wheels, but the supported range is Python 3.10–3.13
and CUDA 12.8:
https://pypi.org/project/pyuipc/

The current Python 3.8 environment therefore cannot consume the published
wheel. A separate Python environment plus the required CUDA runtime would be
needed. The old repository scripts also replace this solver's explicit fiber
material and moving hard patches, so integration would require a data/model
adapter rather than simply launching them.

### libigl

Official Python bindings are pip-installable:
https://libigl.github.io/libigl-python-bindings/

libigl supplies useful geometry processing and AABB queries, but it is not
itself an IPC contact solver. Installing it would not replace the local
augmented-Lagrangian implementation.

### External/subprocess solver

No compatible executable or vendored C++ IPC application was found. The
existing A6000 workflow could only be used through its server environment and
would require explicit coordination and an adapter preserving the current
mesh indexing, attachments, and fibers.

## Decision

The local augmented-Lagrangian active-set implementation remains the default.
`ipctk` is the best optional future backend to evaluate if package installation
is authorized. PyUIPC is not compatible with the current interpreter and is
not a drop-in replacement for the present material model.
