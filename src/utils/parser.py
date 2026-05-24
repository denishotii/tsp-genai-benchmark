"""Parse TSPLIB .tsp files into coordinate lists."""

from pathlib import Path


def parse_tsp(path: str | Path) -> list[tuple[float, float]]:
    """Read a TSPLIB EUC_2D .tsp file and return its node coordinates.

    The i-th returned tuple corresponds to TSPLIB node (i + 1).
    Raises ValueError if the file is not EUC_2D or is malformed.
    """
    lines = Path(path).read_text().splitlines()

    dimension: int | None = None
    edge_weight_type: str | None = None
    coord_start: int | None = None

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            coord_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key == "DIMENSION":
                dimension = int(value)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = value.upper()

    if coord_start is None:
        raise ValueError(f"{path}: NODE_COORD_SECTION not found")
    if edge_weight_type != "EUC_2D":
        raise ValueError(
            f"{path}: EDGE_WEIGHT_TYPE must be EUC_2D, got {edge_weight_type!r}"
        )
    if dimension is None:
        raise ValueError(f"{path}: DIMENSION missing in header")

    coords: list[tuple[float, float]] = []
    for raw in lines[coord_start:]:
        line = raw.strip()
        if not line or line == "EOF":
            break
        parts = line.split()
        coords.append((float(parts[1]), float(parts[2])))

    if len(coords) != dimension:
        raise ValueError(
            f"{path}: expected {dimension} coordinates, parsed {len(coords)}"
        )

    return coords
