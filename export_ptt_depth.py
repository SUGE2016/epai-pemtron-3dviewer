from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def read_ptt(path: Path) -> tuple[int, int, float, float, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 76:
        raise ValueError(f"{path.name}: too small for PTT header")
    height, width = struct.unpack_from("<II", data, 0)
    pitch_x, pitch_y = struct.unpack_from("<ff", data, 8)
    expected = width * height * 3 * 2
    actual = len(data) - 76
    if actual != expected:
        raise ValueError(f"{path.name}: expected {expected} payload bytes, got {actual}")
    planes = np.frombuffer(data, dtype="<u2", offset=76).reshape(3, height, width)
    return width, height, pitch_x, pitch_y, planes


def normalize_u8(depth: np.ndarray, mask: np.ndarray, percentiles: tuple[float, float], invert: bool) -> np.ndarray:
    valid = depth[mask].astype(np.float32)
    low, high = np.percentile(valid, percentiles)
    if high <= low:
        high = low + 1.0
    if invert:
        norm = (high - depth.astype(np.float32)) / (high - low)
    else:
        norm = (depth.astype(np.float32) - low) / (high - low)
    out = np.clip(norm * 255, 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def colorize(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((g * 1.7 - 0.45) * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(g - 0.52) / 0.52) * 235, 0, 235).astype(np.uint8)
    rgb[..., 2] = np.clip((1.15 - g * 1.45) * 255, 0, 255).astype(np.uint8)
    return rgb


def export_one(path: Path, output_dir: Path, plane: int, invert: bool, percentiles: tuple[float, float]) -> None:
    width, height, pitch_x, pitch_y, planes = read_ptt(path)
    depth = planes[plane].copy()
    mask = depth < 60000
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{path.stem}_plane{plane}_raw16.png"
    preview_path = output_dir / f"{path.stem}_plane{plane}_preview.png"
    color_path = output_dir / f"{path.stem}_plane{plane}_color.png"
    npy_path = output_dir / f"{path.stem}_plane{plane}_raw.npy"

    raw16 = depth.astype(np.uint16)
    raw16[~mask] = 0
    Image.fromarray(raw16).save(raw_path)
    gray = normalize_u8(depth, mask, percentiles, invert)
    Image.fromarray(gray).save(preview_path)
    Image.fromarray(colorize(gray)).save(color_path)
    np.save(npy_path, depth)

    valid = depth[mask]
    print(
        f"{path.name}: {width}x{height}, pitch=({pitch_x:.8f},{pitch_y:.8f}), "
        f"plane={plane}, valid={mask.mean()*100:.1f}%, "
        f"p1/p50/p99={np.percentile(valid, [1, 50, 99]).round(1).tolist()}"
    )
    print(f"  raw16:   {raw_path}")
    print(f"  preview: {preview_path}")
    print(f"  color:   {color_path}")
    print(f"  npy:     {npy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export depth/height planes from Bentron/Pemtron .ptt files.")
    parser.add_argument("paths", nargs="+", type=Path, help=".ptt files or directories containing .ptt files")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("exported_depth"))
    parser.add_argument("--plane", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.ptt")))
        else:
            files.append(path)
    for file_path in files:
        export_one(file_path, args.output_dir, args.plane, args.invert, (args.percentiles[0], args.percentiles[1]))


if __name__ == "__main__":
    main()
