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

Graceful is the wrong default on a cluster
------------------------------------------
That fallback is defensible on a workstation and expensive in a queue. A batch
job waits for its GPU allocation, ``resolve_device("cuda")`` warns, returns
``cpu``, and the run trains on the CPU *inside* the GPU allocation until the
wall clock ends. The warning is in a log nobody reads until the job is over.

So the fallback is kept and made **opt-out and visible**:
``resolve_device(..., strict=True)`` — reached from a configuration file as
``training.strict_device`` — raises instead, with the same message plus a
one-sentence diagnosis of the probable cause. :func:`device_report` prints
everything that diagnosis is drawn from, and ``python -m poraque.ml.device
--check`` runs it and exits non-zero, which is one line at the top of a job
script.

The check that ``torch.cuda.is_available()`` is not
---------------------------------------------------
``is_available()`` answers "is there a driver and a device", not "can this
binary generate code for this device". Those come apart: CUDA 13 dropped
Maxwell, Pascal and **Volta**, so a ``+cu130`` wheel on a V100 (``sm_70``)
reports available and then aborts on the first kernel launch with ``no kernel
image is available for execution on the device``. That is a failure after the
queue time has been spent, which is the most expensive moment to have it.
:func:`cuda_capability_supported` asks the second question, and
:func:`resolve_device` refuses a device this build has no kernels for.

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

import os
import warnings

import torch

#: Backends in preference order when ``device="auto"``.
PREFERENCE_ORDER = ("cuda", "mps", "cpu")


def cuda_available():
    """Whether a usable CUDA device is present."""
    return bool(torch.cuda.is_available())


def _listed_capabilities():
    """
    Compute capabilities this PyTorch build carries kernels for, as tuples.

    ``torch.cuda.get_arch_list()`` mixes ``"sm_70"`` with ``"compute_80"`` and,
    on some builds, an architecture suffix (``"sm_90a"``). Only the ``sm_``
    entries are real binary kernels, and only the leading digits are the
    capability.
    """
    listed = []
    for name in torch.cuda.get_arch_list():
        if not name.startswith("sm_"):
            continue
        digits = "".join(ch for ch in name[3:] if ch.isdigit())
        if len(digits) >= 2:
            listed.append((int(digits[:-1]), int(digits[-1])))
    return sorted(listed)


def cuda_capability_supported(index=0):
    """
    Whether this PyTorch build carries kernels for the GPU at ``index``.

    :func:`torch.cuda.is_available` answers "is there a driver and a device",
    not "can this binary run on it". CUDA 13 dropped Maxwell, Pascal and Volta,
    so a ``+cu130`` wheel on a V100 (``sm_70``) reports available and then fails
    on the first launch with ``no kernel image is available for execution on the
    device`` — after the queue time has been spent.

    A build ships PTX for its newest listed architecture, which JITs forward, so
    only a capability *below* everything listed is reported unsupported.

    Parameters
    ----------
    index : int, optional
        CUDA device ordinal.

    Returns
    -------
    bool
        ``False`` when there is no CUDA at all, so this is safe to call
        anywhere.
    """
    if not cuda_available():
        return False
    listed = _listed_capabilities()
    return not listed or torch.cuda.get_device_capability(index) >= listed[0]


def _driver_version():
    """
    The highest CUDA version the installed driver supports, as ``"12.6"``.

    Guarded and private: the query lives on a private torch symbol that has
    moved between releases, and a missing driver version is a cosmetic gap in a
    report rather than a reason for it to fail.
    """
    try:
        raw = torch._C._cuda_getDriverVersion()
    except Exception:  # pragma: no cover - depends on the torch build
        return None
    if not raw:
        return None
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _cuda_diagnosis():
    """
    One sentence naming the probable reason CUDA is unusable here.

    This is the difference between "it did not work" and "install ``cu126``".
    Every branch reports something a reader can act on, and the fallthrough
    says plainly that it does not know rather than inventing a cause.
    """
    build = f"This build is torch {torch.__version__}"
    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        return (f"{build}, which is a CPU-only wheel -- it has no CUDA runtime "
                f"at all. Install a +cuXXX build from "
                f"https://download.pytorch.org/whl/.")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        return (f"CUDA_VISIBLE_DEVICES={visible!r} hides every GPU from this "
                f"process; on a cluster that usually means the job asked for "
                f"no GPU resource.")

    if not torch.cuda.is_available():
        return (f"{build} (CUDA {cuda_build}) but no device initialised: "
                f"either the machine has no NVIDIA GPU, or its driver is older "
                f"than CUDA {cuda_build} requires. `nvidia-smi` prints the "
                f"highest CUDA version the driver supports.")

    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:  # pragma: no cover - only on a broken driver
        return f"{build} (CUDA {cuda_build}); the device could not be queried."

    if not cuda_capability_supported(0):
        return (f"{build} (CUDA {cuda_build}), built for "
                f"{torch.cuda.get_arch_list()}, which does not include "
                f"sm_{major}{minor} -- this GPU. Install a wheel that lists "
                f"this architecture; see the installation guide.")

    return (f"{build} (CUDA {cuda_build}) and sm_{major}{minor} is supported, "
            f"so the cause is not the build.")


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
    checks = {"cuda": cuda_available, "mps": mps_available,
              "cpu": lambda: True}
    return [name for name in PREFERENCE_ORDER if checks[name]()]


def _refuse(message, strict):
    """
    Raise or warn-and-fall-back, from one message.

    The two paths share their text on purpose: a run that failed strictly and a
    run that degraded quietly should be searchable by the same sentence, and
    the strict form adds the diagnosis rather than replacing the description.
    """
    if strict:
        raise RuntimeError(message)
    warnings.warn(f"{message} Falling back to CPU.", RuntimeWarning,
                  stacklevel=3)
    return torch.device("cpu")


def resolve_device(preference="auto", verbose=False, strict=False):
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
    strict : bool, optional
        Raise instead of falling back. A workstation wants the fallback: a
        configuration written on a machine with a GPU should still run on a
        laptop. A batch job does not. By the time the fallback happens the job
        has already taken its place in the GPU queue, and a run that quietly
        moves to the CPU costs hours and produces the same numbers far too
        late.

        The message is the one the warning would have carried, plus a sentence
        from :func:`_cuda_diagnosis` naming the probable cause.

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        Under ``strict`` only, where the non-strict path warns.
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
        return _refuse("CUDA was requested but no CUDA device is available. "
                       + _cuda_diagnosis(), strict)

    # Present, initialised, and unable to run a single kernel. Caught here
    # rather than left to surface as `no kernel image is available for
    # execution on the device` from inside the first forward pass, which is
    # both later and far harder to read.
    if kind == "cuda" and not cuda_capability_supported(
            torch.device(requested).index or 0):
        major, minor = torch.cuda.get_device_capability(
            torch.device(requested).index or 0)
        return _refuse(
            f"CUDA was requested but this PyTorch build ({torch.__version__}) "
            f"has no kernels for sm_{major}{minor}; it was built for "
            f"{torch.cuda.get_arch_list()}. Install a wheel that lists this "
            f"architecture -- see the installation guide.", strict)

    if kind == "mps" and not mps_available():
        backend = getattr(torch.backends, "mps", None)
        reason = ("this PyTorch build has no MPS support"
                  if backend is None or not backend.is_built()
                  else "no compatible Apple Silicon GPU / macOS version")
        return _refuse(f"MPS was requested but is unavailable ({reason}).",
                       strict)

    if kind not in ("cuda", "mps", "cpu"):
        return _refuse(f"Unknown device {preference!r}.", strict)

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
        e.g. ``"cuda:0 (NVIDIA A100-SXM4-40GB, sm_80, 39.6 GiB)"`` or
        ``"mps (Apple Metal)"``.

    Notes
    -----
    The compute capability is in there because it is the one field that makes a
    wheel/GPU mismatch diagnosable at a glance: a log line reading ``sm_70``
    beside a build that lists ``sm_75`` upwards says immediately what four
    hours of CPU training would otherwise have to be traced back to.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        memory = properties.total_memory / (1024 ** 3)
        return (f"cuda:{index} ({properties.name}, "
                f"sm_{properties.major}{properties.minor}, {memory:.1f} GiB)")

    if device.type == "mps":
        return "mps (Apple Metal Performance Shaders)"

    return f"cpu ({torch.get_num_threads()} threads)"


def device_report(preference="auto"):
    """
    Lines describing every accelerator this build can and cannot use.

    Written for a log that will be read after the fact, when the question is
    "why did this run on the CPU?". Everything the answer needs is here: the
    torch build and its CUDA runtime, where it was imported from (on a cluster
    a ``~/.local`` install outranks the active environment and is the one that
    gets imported), the architectures it was compiled for, and one line per GPU
    saying whether this build has kernels for it.

    Parameters
    ----------
    preference : str or torch.device or None, optional
        The device the caller intends to use. The last line says whether it
        resolves to itself or would silently degrade.

    Returns
    -------
    list of str
    """
    lines = [
        f"torch        : {torch.__version__}",
        f"  installed  : {os.path.dirname(getattr(torch, '__file__', '') or '')}",
        f"  cuda build : {getattr(torch.version, 'cuda', None) or 'none (CPU-only wheel)'}",
    ]

    if getattr(torch.version, "cuda", None):
        lines.append(f"  arch list  : {torch.cuda.get_arch_list()}")

    if cuda_available():
        lines.append(f"cuda devices : {torch.cuda.device_count()}")
        driver = _driver_version()
        if driver is not None:
            lines.append(f"  driver     : CUDA {driver}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            usable = ("usable" if cuda_capability_supported(index)
                      else "NOT USABLE by this build")
            lines.append(
                f"  cuda:{index}     : {properties.name}, "
                f"sm_{properties.major}{properties.minor}, "
                f"{properties.total_memory / (1024 ** 3):.1f} GiB -- {usable}")
    else:
        lines.append("cuda devices : none")
        lines.append(f"  diagnosis  : {_cuda_diagnosis()}")

    lines.append(f"mps          : {'available' if mps_available() else 'not available'}")
    lines.append(f"cpu threads  : {torch.get_num_threads()}")
    lines.append(f"available    : {available_devices()}")

    requested = "auto" if preference is None else str(preference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resolved = resolve_device(requested)
    lines.append(f"requested    : {requested!r} -> {describe_device(resolved)}")
    return lines


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


def enable_tf32(device, enabled=True):
    """
    Allow TensorFloat-32 matmuls and convolutions on ``device``.

    TF32 keeps float32's range and drops its mantissa to ten bits inside the
    tensor cores, which is a real speedup on Ampere and later and **exactly
    nothing** before it: a V100 has no TF32 path, so this is a no-op there
    rather than a regression. It is a process-wide backend flag, so it is set
    by the run — never by the library on a caller's behalf.

    Parameters
    ----------
    device : torch.device or str
        Ignored unless CUDA: the flags do not exist on other backends.
    enabled : bool, optional

    Returns
    -------
    bool
        Whether the flags were actually set, so a caller can log the truth
        rather than the request.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type != "cuda":
        return False
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    return bool(enabled)


def _main(argv=None):
    """
    ``python -m poraque.ml.device --check``: is the accelerator usable?

    One line at the top of a job script (``python -m poraque.ml.device --check
    || exit 1``) costs three seconds and catches the wheel/driver/architecture
    mismatch that otherwise costs a slot in the GPU queue and a wall-clock
    limit's worth of CPU training.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poraque.ml.device",
        description="Report the accelerators this PyTorch build can use.")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the requested device is not usable, "
                             "instead of reporting and exiting 0")
    parser.add_argument("--device", default="auto",
                        help="the device to check (default: auto)")
    arguments = parser.parse_args(argv)

    for line in device_report(arguments.device):
        print(line)

    if not arguments.check:
        return 0

    try:
        resolve_device(arguments.device, strict=True)
    except RuntimeError as error:
        print(f"\nFAILED: {error}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(_main())
