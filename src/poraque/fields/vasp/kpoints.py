"""VASP regular meshes, explicit weighted points and paired line-mode paths."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Kpoints:
    """Expanded fractional reciprocal coordinates (rows), weights and path data.

    Mesh weights sum to one; path weights are None, so line-mode samples cannot
    accidentally be used as a Brillouin-zone quadrature. Segment starts mark
    discontinuities; repeated endpoints are retained for plotting.
    """

    points: np.ndarray
    weights: np.ndarray | None
    segment_starts: tuple[int, ...] = ()
    labels: tuple[tuple[int, str], ...] = ()

    @classmethod
    def from_file(cls, path: str | Path, cell: np.ndarray,
                  scale: float | None = None) -> 'Kpoints':
        """Read a file. Cartesian VASP units require its POSCAR length scale."""
        return cls.from_string(Path(path).read_text(), cell, scale)

    @classmethod
    def from_string(cls, text: str, cell: np.ndarray,
                    scale: float | None = None) -> 'Kpoints':
        """Expand full unsymmetrized meshes or endpoint-paired band paths.

        Supports Gamma and Monkhorst-Pack meshes with optional shifts in mesh
        units, reciprocal or Cartesian line/explicit coordinates. Cartesian
        values are in 2*pi/scale, exactly as in VASP (not Angstrom^-1).
        """
        lines = text.splitlines()
        if len(lines) < 4:
            raise ValueError('KPOINTS requires at least four lines')
        def clean(line):
            return re.split(r'[!#]', line, maxsplit=1)[0].strip()

        count = int(clean(lines[1]))
        mode = clean(lines[2]).lower()
        if count < 0:
            raise ValueError('Negative KPOINTS point count')
        if count == 0:
            if not mode.startswith(('m', 'g')):
                raise ValueError('Only Gamma/Monkhorst-Pack automatic meshes are supported')
            sizes = np.array([int(x) for x in clean(lines[3]).split()])
            shift = np.zeros(3)
            rest = [clean(line) for line in lines[4:] if clean(line)]
            if rest:
                shift = np.array([float(x) for x in rest[0].split()])
            if (sizes.shape != (3,) or np.any(sizes <= 0) or shift.shape != (3,) or
                    not np.isfinite(shift).all()):
                raise ValueError('Mesh requires three positive dimensions and three shifts')
            axes = [(np.arange(n) + s - ((n - 1) / 2 if mode.startswith('m') else 0)) / n
                    for n, s in zip(sizes, shift)]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            return cls(points, np.full(len(points), 1 / len(points)))
        is_line = mode.startswith('l')
        coordinate_mode = clean(lines[3]).lower() if is_line else mode
        if not coordinate_mode.startswith(('r', 'c', 'k')):
            raise ValueError('Coordinate mode must be Reciprocal or Cartesian')
        entries = [line for line in lines[4 if is_line else 3:] if clean(line)]
        values, labels = [], []
        for entry in entries:
            fields = clean(entry).split()
            required = 3 if is_line else 4
            if len(fields) < required:
                raise ValueError('Incomplete k-point coordinates/weight')
            values.append([float(x.replace('D', 'E').replace('d', 'e')) for x in fields[:required]])
            label = re.split(r'[!#]', entry, maxsplit=1)
            labels.append(label[1].strip() if len(label) > 1 else '')
        array = np.asarray(values, dtype=float)
        if not values or not np.isfinite(array).all():
            raise ValueError('Empty/nonfinite k-point list')
        points = array[:, :3]
        if coordinate_mode.startswith(('c', 'k')):
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError('Cartesian KPOINTS require a positive POSCAR length scale')
            reciprocal = 2 * np.pi * np.linalg.inv(cell).T
            points = (points * 2 * np.pi / scale) @ np.linalg.inv(reciprocal)
        if is_line:
            if count < 2 or len(points) % 2:
                raise ValueError('Line mode requires >=2 points per segment and paired endpoints')
            expanded, ticks, starts = [], [], []
            for i in range(0, len(points), 2):
                offset = len(expanded)
                starts.append(offset)
                expanded.extend(np.linspace(points[i], points[i + 1], count))
                ticks.extend([(offset, labels[i]), (offset + count - 1, labels[i + 1])])
            return cls(np.asarray(expanded), None, tuple(starts), tuple(ticks))
        if len(points) != count or np.any(array[:, 3] < 0) or array[:, 3].sum() <= 0:
            raise ValueError('Explicit point count or weights invalid')
        return cls(points, array[:, 3] / array[:, 3].sum())

    def distances(self, cell: np.ndarray) -> np.ndarray:
        """Cumulative path distance in Angstrom^-1, excluding jumps at breaks."""
        if self.weights is not None:
            raise ValueError('Distances are defined only for a line-mode path')
        cartesian = self.points @ (2 * np.pi * np.linalg.inv(cell).T)
        steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        for start in self.segment_starts[1:]:
            steps[start - 1] = 0
        return np.r_[0, np.cumsum(steps)]
