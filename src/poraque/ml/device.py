# -*- coding: utf-8 -*-
# file: device.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Hardware accelerator selection: CUDA, Apple Metal (MPS), or CPU.

:func:`resolve_device` is the single place the rest of the package asks "where
should this run?". It prefers CUDA, then MPS, then CPU, and **degrades
gracefully**: an explicit request for a backend that is not present produces a
warning and a CPU fallback rather than an exception, so a configuration file
written on a workstation still runs on a laptop.

Apple Metal caveats that actually matter here
---------------------------------------------
MPS is not a drop-in CUDA replacement, and two of its gaps hit this codebase
directly:

* **No float64 at all.** ``Cannot convert a MPS Tensor to float64`` is raised
  on any attempt. Poraquê used double precision for the reciprocal-lattice
  inverse, where it is genuinely wanted — a badly conditioned cell gives a
  reciprocal metric that single precision resolves poorly, and every
  :math:`\mathbf G` vector inherits the error.
* **``torch.linalg.det`` is unimplemented.** It is used to obtain the cell
  volume for integrals.

Both operate on a single :math:`3\times3` matrix per structure, i.e.
:math:`\mathcal{O}(1)` work beside a :math:`128^3` FFT. The fix, in
:func:`~poraque.ml.physics.cell_reciprocal` and
:func:`~poraque.ml.physics.cell_volume`, is therefore to evaluate the cell
metric **on the CPU in float64** and move only the result to the accelerator.
That is faster to write than a single-precision workaround, keeps the extra
accuracy on every backend, and costs nothing measurable.

Everything the Fourier layers need — ``fft.rfftn``/``irfftn``, complex
``einsum``, ``Conv3d``, ``GroupNorm``, ``softplus`` — is supported on MPS and
verified by ``tests/test_device.py``.
"""

import warnings

import torch

#: Backends in preference order when ``device="auto"``.
PREFERENCE_ORDER = ("cuda", "mps", "cpu")


def cuda_available():
    """Whether a usable CUDA device is present."""
    return bool(torch.cuda.is_available())


def mps_available():
    """
    Whether Apple Metal Performance Shaders can be used.

    Both checks are required: ``is_built`` reports whether this PyTorch binary
    has MPS compiled in, ``is_available`` whether the machine and macOS version
    can actually run it.
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_built() and backend.is_available())


def available_devices():
    """
    List the backends usable on this machine, most preferred first.

    Returns
    -------
    list of str
        Always contains at least ``"cpu"``.
    """
    found = []
    if cuda_available():
        found.append("cuda")
    if mps_available():
        found.append("mps")
    found.append("cpu")
    return found


def resolve_device(preference="auto", verbose=False):
    """
    Resolve a device request to a concrete :class:`torch.device`.

    Parameters
    ----------
    preference : str or torch.device or None, optional
        ``"auto"`` (or ``None``) picks the best available backend in
        :data:`PREFERENCE_ORDER`. An explicit ``"cuda"``, ``"cuda:1"``,
        ``"mps"`` or ``"cpu"`` is honoured when possible and falls back to CPU
        with a warning when not.
    verbose : bool, optional
        Print the resolved device and the reason.

    Returns
    -------
    torch.device
    """
    if isinstance(preference, torch.device):
        preference = preference.type if preference.index is None else str(preference)
    requested = "auto" if preference is None else str(preference).strip().lower()

    if requested in ("auto", ""):
        chosen = torch.device(available_devices()[0])
        if verbose:
            print(f"[poraque] device: {describe_device(chosen)} (auto)")
        return chosen

    kind = requested.split(":")[0]

    if kind == "cuda" and not cuda_available():
        warnings.warn(
            "CUDA was requested but no CUDA device is available; "
            "falling back to CPU.", RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")

    if kind == "mps" and not mps_available():
        backend = getattr(torch.backends, "mps", None)
        reason = ("this PyTorch build has no MPS support"
                  if backend is None or not backend.is_built()
                  else "no compatible Apple Silicon GPU / macOS version")
        warnings.warn(
            f"MPS was requested but is unavailable ({reason}); "
            f"falling back to CPU.", RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")

    if kind not in ("cuda", "mps", "cpu"):
        warnings.warn(
            f"Unknown device {preference!r}; falling back to CPU.",
            RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")

    chosen = torch.device(requested)
    if verbose:
        print(f"[poraque] device: {describe_device(chosen)}")
    return chosen


def describe_device(device):
    """
    Human-readable description of a device, including the hardware name.

    Parameters
    ----------
    device : torch.device or str

    Returns
    -------
    str
        e.g. ``"cuda:0 (NVIDIA A100-SXM4-40GB, 39.6 GiB)"`` or
        ``"mps (Apple Metal)"``.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        memory = properties.total_memory / (1024 ** 3)
        return f"cuda:{index} ({properties.name}, {memory:.1f} GiB)"

    if device.type == "mps":
        return "mps (Apple Metal Performance Shaders)"

    return f"cpu ({torch.get_num_threads()} threads)"


def supports_float64(device):
    """
    Whether ``device`` can hold float64 tensors.

    MPS cannot, which is why cell-metric arithmetic is pinned to the CPU.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    return device.type != "mps"


def synchronize(device):
    """
    Block until queued work on ``device`` has finished.

    Both CUDA and MPS dispatch asynchronously, so any wall-clock measurement
    that does not synchronise first reports queueing time rather than compute
    time.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_cache(device):
    """Release cached allocator blocks on ``device``, where supported."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
