# ============================================================
# BYTE NDT - Twin focal laws
# Conditions:
#   A: PA1 -> C1
#   B: PA2 -> C2
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "validated_geometry"

OUT_DIR = Path(r"D:\PROJET_BYTENDT_AI\01_SCRIPTS\09_RESULTS_REPORTS\LSB941_TWIN_V1\02_FOCAL_LAWS")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_curve(filename: str) -> np.ndarray:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{filename} must contain at least 3 columns: X,Y,Z")

    return df.iloc[:, 0:3].to_numpy(dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def resample_curve(curve: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a 3D curve to n points using curvilinear distance.
    """
    if len(curve) == n:
        return curve.copy()

    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])

    if s[-1] < 1e-12:
        return np.repeat(curve[:1], n, axis=0)

    s_new = np.linspace(0.0, s[-1], n)

    out = np.zeros((n, 3))
    for j in range(3):
        out[:, j] = np.interp(s_new, s, curve[:, j])

    return out


def local_tangent(curve: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return unit(curve[1] - curve[0])
    if k == len(curve) - 1:
        return unit(curve[-1] - curve[-2])
    return unit(curve[k + 1] - curve[k - 1])


def make_pa_1d_elements(center: np.ndarray, tangent: np.ndarray, n_elements: int, pitch_mm: float) -> np.ndarray:
    """
    1D PA array placed along the local PA trajectory tangent.
    """
    ids = np.arange(n_elements) - (n_elements - 1) / 2.0
    return center[None, :] + ids[:, None] * pitch_mm * tangent[None, :]


def make_pa_2d_elements(
    center: np.ndarray,
    tangent: np.ndarray,
    beam_dir: np.ndarray,
    nx: int,
    ny: int,
    pitch_x_mm: float,
    pitch_y_mm: float,
) -> np.ndarray:
    """
    2D PA array.
    X direction = tangent along PA curve.
    Y direction = transverse direction in the local probe plane.
    """
    ex = unit(tangent)
    ez = unit(beam_dir)
    ey = unit(np.cross(ez, ex))

    if np.linalg.norm(ey) < 1e-12:
        ey = np.array([0.0, 1.0, 0.0])

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0

    elements = []
    for j in range(ny):
        for i in range(nx):
            p = center + ix[i] * pitch_x_mm * ex + iy[j] * pitch_y_mm * ey
            elements.append(p)

    return np.asarray(elements)


def compute_delays(elements: np.ndarray, focus: np.ndarray, c_mm_us: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Delay law in microseconds.
    The farthest element is used as reference; delays are positive.
    """
    distances = np.linalg.norm(elements - focus[None, :], axis=1)
    times = distances / c_mm_us
    delays = times.max() - times
    return delays, distances


def compute_focal_laws(
    pa_curve: np.ndarray,
    indication_curve: np.ndarray,
    condition_name: str,
    probe_type: str = "1D",
    n_elements_1d: int = 32,
    nx: int = 8,
    ny: int = 8,
    pitch_x_mm: float = 0.60,
    pitch_y_mm: float = 0.60,
    c_steel_shear_mm_us: float = 3.23,
    frequency_mhz: float = 5.0,
) -> pd.DataFrame:

    n_scan = min(len(pa_curve), len(indication_curve))
    pa = resample_curve(pa_curve, n_scan)
    focus_curve = resample_curve(indication_curve, n_scan)

    rows = []

    for k in range(n_scan):
        pa_point = pa[k]
        focus = focus_curve[k]
        beam_dir = unit(focus - pa_point)
        tangent = local_tangent(pa, k)

        if probe_type.upper() == "1D":
            elements = make_pa_1d_elements(
                center=pa_point,
                tangent=tangent,
                n_elements=n_elements_1d,
                pitch_mm=pitch_x_mm,
            )
        elif probe_type.upper() == "2D":
            elements = make_pa_2d_elements(
                center=pa_point,
                tangent=tangent,
                beam_dir=beam_dir,
                nx=nx,
                ny=ny,
                pitch_x_mm=pitch_x_mm,
                pitch_y_mm=pitch_y_mm,
            )
        else:
            raise ValueError("probe_type must be '1D' or '2D'")

        delays, distances = compute_delays(elements, focus, c_steel_shear_mm_us)

        for e, element in enumerate(elements):
            rows.append(
                {
                    "condition": condition_name,
                    "probe_type": probe_type.upper(),
                    "scan_id": k,
                    "element_id": e + 1,
                    "frequency_mhz": frequency_mhz,
                    "pitch_x_mm": pitch_x_mm,
                    "pitch_y_mm": pitch_y_mm,
                    "pa_x": pa_point[0],
                    "pa_y": pa_point[1],
                    "pa_z": pa_point[2],
                    "focus_x": focus[0],
                    "focus_y": focus[1],
                    "focus_z": focus[2],
                    "element_x": element[0],
                    "element_y": element[1],
                    "element_z": element[2],
                    "distance_mm": distances[e],
                    "delay_us": delays[e],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    PA1 = load_curve("PA1_ORIGINAL_METHOD_VALIDATED.csv")
    PA2 = load_curve("PA2_ORIGINAL_METHOD_VALIDATED.csv")
    C1 = load_curve("C1_INDICATION_VALIDATED.csv")
    C2 = load_curve("C2_INDICATION_VALIDATED.csv")

    # First validated Twin outputs: 1D PA 32 elements
    laws_A_1d32 = compute_focal_laws(PA1, C1, "A_PA1_to_C1", probe_type="1D", n_elements_1d=32)
    laws_B_1d32 = compute_focal_laws(PA2, C2, "B_PA2_to_C2", probe_type="1D", n_elements_1d=32)

    # Option 1D PA 16 elements
    laws_A_1d16 = compute_focal_laws(PA1, C1, "A_PA1_to_C1", probe_type="1D", n_elements_1d=16)
    laws_B_1d16 = compute_focal_laws(PA2, C2, "B_PA2_to_C2", probe_type="1D", n_elements_1d=16)

    # Option 2D PA 8x8
    laws_A_2d8 = compute_focal_laws(PA1, C1, "A_PA1_to_C1", probe_type="2D", nx=8, ny=8)
    laws_B_2d8 = compute_focal_laws(PA2, C2, "B_PA2_to_C2", probe_type="2D", nx=8, ny=8)

    laws_A_1d32.to_csv(OUT_DIR / "focal_laws_A_PA1_to_C1_1D32.csv", index=False)
    laws_B_1d32.to_csv(OUT_DIR / "focal_laws_B_PA2_to_C2_1D32.csv", index=False)

    laws_A_1d16.to_csv(OUT_DIR / "focal_laws_A_PA1_to_C1_1D16.csv", index=False)
    laws_B_1d16.to_csv(OUT_DIR / "focal_laws_B_PA2_to_C2_1D16.csv", index=False)

    laws_A_2d8.to_csv(OUT_DIR / "focal_laws_A_PA1_to_C1_2D8x8.csv", index=False)
    laws_B_2d8.to_csv(OUT_DIR / "focal_laws_B_PA2_to_C2_2D8x8.csv", index=False)

    print("BYTE NDT Twin focal laws exported:")
    print(OUT_DIR / "focal_laws_A_PA1_to_C1_1D32.csv")
    print(OUT_DIR / "focal_laws_B_PA2_to_C2_1D32.csv")
    print(OUT_DIR / "focal_laws_A_PA1_to_C1_1D16.csv")
    print(OUT_DIR / "focal_laws_B_PA2_to_C2_1D16.csv")
    print(OUT_DIR / "focal_laws_A_PA1_to_C1_2D8x8.csv")
    print(OUT_DIR / "focal_laws_B_PA2_to_C2_2D8x8.csv")


if __name__ == "__main__":
    main()
#export_wide_delay_table(laws_A)
