"""Rebuild the README demo GIF from its original high-frame-rate MP4 segment."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def build_gif(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    output_fps: int,
    width: int,
    colors: int,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise ValueError("Source video does not report a valid frame rate")

    start_frame = round(start_seconds * source_fps)
    output_frame_count = round(duration_seconds * output_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[Image.Image] = []
    next_source_offset = 0
    for output_index in range(output_frame_count):
        wanted_offset = round(output_index * source_fps / output_fps)
        while next_source_offset <= wanted_offset:
            ok, frame = capture.read()
            if not ok:
                break
            next_source_offset += 1
        if not ok:
            break

        height = round(frame.shape[0] * width / frame.shape[1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame).resize((width, height), Image.Resampling.LANCZOS)
        frames.append(image)

    capture.release()
    if not frames:
        raise RuntimeError("No frames were decoded from the source video")

    # One shared palette reduces file size and decoder work while preventing
    # the visible color flicker caused by a different palette on every frame.
    palette_samples = frames[:: max(1, len(frames) // 20)]
    palette_strip = Image.new("RGB", (width, sum(frame.height for frame in palette_samples)))
    y_offset = 0
    for sample in palette_samples:
        palette_strip.paste(sample, (0, y_offset))
        y_offset += sample.height
    palette = palette_strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]

    frame_duration_ms = round(1000 / output_fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("demo.mp4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("smart_vehicle_detector-demo.gif"),
    )
    parser.add_argument("--start", type=float, default=8.8)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--colors", type=int, default=128)
    args = parser.parse_args()
    build_gif(
        args.input,
        args.output,
        args.start,
        args.duration,
        args.fps,
        args.width,
        args.colors,
    )
