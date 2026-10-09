"""Corta un tramo de un video con OpenCV (alternativa a ffmpeg).

    python -m alpr.tools.cut_video --source "E:\\...\\grabacion.avi" \
        --start 00:10:00 --duration 00:05:00 --out data/videos/yrigoyen_01.mp4

Lee el video en orden y descarta hasta `--start` (más lento que saltar, pero
EXACTO). El tramo se decide con el RELOJ del video (timestamps), no con el fps
declarado, porque ciertos exports (Avigilon ACC) declaran un fps falso. El clip se
escribe con el fps real medido. Los frames del GT cuentan desde el inicio del CLIP.
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
    """Corta [start_s, start_s + duration_s) según el RELOJ del video (POS_MSEC).

    Los AVI exportados por algunos VMS declaran un fps falso (p. ej. 120) y son de
    fps variable: contar frames / fps declarado da tiempos errados. Por eso el
    tramo se decide con el timestamp de cada frame y el clip se escribe con el fps
    REAL medido (frames / tiempo transcurrido), de modo que el pipeline
    (frame_idx / fps) obtenga segundos reales. Si el backend no entrega
    timestamps, se cae al fps declarado y se avisa.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV no pudo abrir {source!r} (ruta o códec)")
    declared_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_ms = start_s * 1000.0
    end_ms = None if duration_s is None else (start_s + duration_s) * 1000.0
    eps = 0.5   # ms: tolera redondeo de pts en los bordes

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames, times, idx = [], [], 0
    used_fallback = False
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t <= 0 and idx > 0:        # el backend no da timestamps
                used_fallback = True
                if not declared_fps or declared_fps <= 0:
                    raise RuntimeError("El video no informa timestamps ni fps válidos")
                t = idx * 1000.0 / declared_fps
            if end_ms is not None and t >= end_ms - eps:
                break
            if t >= start_ms - eps:
                frames.append(frame)
                times.append(t)
            idx += 1
            if idx % 2000 == 0:
                print(f"  {idx} frames leídos ({time.time() - t0:.0f}s, t={t / 1000:.0f}s de video)",
                      file=sys.stderr)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No se escribió ningún frame: ¿--start ({start_s}s) supera la duración del video?")

    if len(times) > 1 and times[-1] > times[0]:
        real_fps = (len(times) - 1) * 1000.0 / (times[-1] - times[0])
    else:
        real_fps = declared_fps
    deltas = sorted(b - a for a, b in zip(times, times[1:])) or [0.0]

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), real_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"No pude crear {out!r}")
    for f in frames:
        writer.write(f)
    writer.release()
    return {"declared_fps": declared_fps, "fps": real_fps, "size": (w, h),
            "frames_written": len(frames), "first_source_frame": idx - len(frames),
            "seconds": len(frames) / real_fps, "used_fallback": used_fallback,
            "delta_ms_min": deltas[0], "delta_ms_median": deltas[len(deltas) // 2],
            "delta_ms_max": deltas[-1]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", default="0")
    p.add_argument("--duration", default=None, help="Si se omite, hasta el final")
    args = p.parse_args(argv)
    info = cut_video(args.source, args.out, parse_time(args.start),
                     parse_time(args.duration) if args.duration else None)
    print(f"{info['frames_written']} frames, {info['seconds']:.1f}s reales, "
          f"{info['fps']:.2f} fps medidos (el archivo declaraba {info['declared_fps']:.2f}), "
          f"{info['size'][0]}x{info['size'][1]} -> {args.out}")
    print(f"Intervalo entre frames (ms): mín {info['delta_ms_min']:.1f} / mediana "
          f"{info['delta_ms_median']:.1f} / máx {info['delta_ms_max']:.1f}")
    if info["delta_ms_max"] > 2.5 * max(info["delta_ms_median"], 1e-6):
        print("AVISO: fps variable (hay saltos grandes entre frames). El clip se escribe a fps "
              "constante, así que los tiempos son aproximados dentro del clip.")
    if info["used_fallback"]:
        print("AVISO: el backend no entregó timestamps; se usó el fps declarado del archivo.")
    if abs(info["fps"] - info["declared_fps"]) > 0.2 * max(info["declared_fps"], 1e-6):
        print("AVISO: el fps declarado del archivo NO es confiable; el clip usa el medido.")


if __name__ == "__main__":
    main()
