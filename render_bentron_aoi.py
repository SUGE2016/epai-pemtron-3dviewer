from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_to_u8(array: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    data = array.astype(np.float32)
    low, high = np.percentile(data[np.isfinite(data)], [low_pct, high_pct])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def save_plane(path: Path, plane: np.ndarray) -> None:
    Image.fromarray(normalize_to_u8(plane)).save(path)


def contact_sheet(images: list[tuple[Path, str]], output: Path, thumb_size: tuple[int, int] = (220, 226)) -> None:
    thumbs: list[Image.Image] = []
    tw, th = thumb_size
    label_h = 24
    for path, label in images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((tw, th))
        canvas = Image.new("RGB", (tw, th + label_h), "white")
        canvas.paste(image, ((tw - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((6, th + 6), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = min(5, max(1, len(thumbs)))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), (245, 245, 245))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % cols) * tw, (index // cols) * (th + label_h)))
    sheet.save(output)


def render_ptt(path: Path, output_dir: Path) -> list[Path]:
    data = path.read_bytes()
    if len(data) < 76:
        raise ValueError(f"{path.name}: too small for a PTT header")

    height, width = struct.unpack_from("<II", data, 0)
    x_pitch, y_pitch = struct.unpack_from("<ff", data, 8)
    payload = len(data) - 76
    expected = width * height * 3 * 2
    if payload != expected:
        raise ValueError(
            f"{path.name}: expected payload {expected} bytes for "
            f"{width}x{height}x3 uint16 planes, got {payload}"
        )

    planes = np.frombuffer(data, dtype="<u2", offset=76).reshape(3, height, width)
    written: list[Path] = []
    for index, plane in enumerate(planes):
        out = output_dir / f"{path.stem}_ptt_plane{index}.png"
        save_plane(out, plane)
        written.append(out)

    rgb = np.dstack([normalize_to_u8(planes[index]) for index in range(3)])
    rgb_out = output_dir / f"{path.stem}_ptt_rgb_preview.png"
    Image.fromarray(rgb).save(rgb_out)
    written.append(rgb_out)

    print(
        f"{path.name}: PTT header height={height}, width={width}, "
        f"x_pitch={x_pitch:.8f}, y_pitch={y_pitch:.8f}, format=76-byte header + 3 uint16 planes"
    )
    return written


def render_pot(path: Path, output_dir: Path) -> list[Path]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path.name}: too small for a POT header")

    width_f, height_f, x_pitch, y_pitch, unknown = struct.unpack_from("<fffff", data, 0)
    width, height = int(round(width_f)), int(round(height_f))
    payload = len(data) - 20
    expected = width * height * 5
    if payload != expected:
        raise ValueError(
            f"{path.name}: expected payload {expected} bytes for "
            f"{width}x{height}x5 uint8 planes, got {payload}"
        )

    planes = np.frombuffer(data, dtype=np.uint8, offset=20).reshape(5, height, width)
    written: list[Path] = []
    for index, plane in enumerate(planes):
        out = output_dir / f"{path.stem}_pot_plane{index}.png"
        save_plane(out, plane)
        written.append(out)

    rgb = np.dstack([planes[0], planes[2], planes[4]])
    rgb_out = output_dir / f"{path.stem}_pot_rgb_024_preview.png"
    Image.fromarray(rgb).save(rgb_out)
    written.append(rgb_out)

    print(
        f"{path.name}: POT header width={width}, height={height}, "
        f"x_pitch={x_pitch:.8f}, y_pitch={y_pitch:.8f}, field4={unknown:.3f}, "
        "format=20-byte header + 5 uint8 planes"
    )
    return written


def render_file(path: Path, output_dir: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix == ".ptt":
        return render_ptt(path, output_dir)
    if suffix == ".pot":
        return render_pot(path, output_dir)
    raise ValueError(f"{path.name}: unsupported extension {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Bentron AOI .ptt/.pot binary image files to PNG previews.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("renders"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_inputs: list[tuple[Path, str]] = []
    for file_path in args.files:
        written = render_file(file_path, args.output_dir)
        sheet_inputs.extend((path, path.stem) for path in written)

    if sheet_inputs:
        sheet_path = args.output_dir / "render_contact_sheet.png"
        contact_sheet(sheet_inputs, sheet_path)
        print(f"contact sheet: {sheet_path.resolve()}")


if __name__ == "__main__":
    main()
