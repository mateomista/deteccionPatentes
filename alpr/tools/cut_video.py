"""Corta un tramo de un video con OpenCV (alternativa a ffmpeg).

    python -m alpr.tools.cut_video --source "E:\\...\\grabacion.avi" \
        --start 00:10:00 --duration 00:05:00 --out data/videos/yrigoyen_01.mp4

Lee el video en orden y descarta hasta `--start` (más lento que saltar, pero
EXACTO: los frames del clip coinciden con los que verá el pipeline, que es lo que
necesita el ground_truth.csv). Los frames del GT cuentan desde el inicio del CLIP.
`--start` / `--duration` aceptan SS, MM:SS, HH:MM:SS o segundos con decimales.
"""
import argparse
import os
import sys
import time

import cv2


def parse_time(s):
    parts = [float(p) for p in str(s).split(":")]
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise ValueError(f"Tiempo inválido: {s!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def cut_video(source, out, start_s=0.0, duration_s=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV no pudo abrir {source!r} (ruta o códec)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("El video no informa fps válidos")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(round(start_s * fps))
    last = None if duration_s is None else first + int(round(duration_s * fps))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"No pude crear {out!r}")

    idx = written = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (last is not None and idx >= last):
                break
            if idx >= first:
                writer.write(frame)
                written += 1
            idx += 1
            if idx % 2000 == 0:
                print(f"  {idx} frames leídos ({time.time() - t0:.0f}s)", file=sys.stderr)
    finally:
        cap.release()
        writer.release()
    if written == 0:
        raise RuntimeError(f"No se escribió ningún frame: ¿--start ({start_s}s) supera la duración del video?")
    return {"fps": fps, "size": (w, h), "frames_written": written, "first_source_frame": first}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", default="0")
    p.add_argument("--duration", default=None, help="Si se omite, hasta el final")
    args = p.parse_args(argv)
    info = cut_video(args.source, args.out, parse_time(args.start),
                     parse_time(args.duration) if args.duration else None)
    print(f"{info['frames_written']} frames ({info['frames_written'] / info['fps']:.1f}s, "
          f"{info['fps']:.2f} fps, {info['size'][0]}x{info['size'][1]}) -> {args.out}")
    print(f"El frame 0 del clip es el frame {info['first_source_frame']} del video original.")


if __name__ == "__main__":
    main()
