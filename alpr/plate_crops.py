"""Exporta los recortes de patente que ve el pipeline, con su resultado, para revisarlos a ojo.

    python -m alpr.inference --source clip.mp4 --plate-model outputs/final_model --strict-plate \
        --export-plate-crops outputs/crops_patente

Cada recorte que RT-DETR produjo se guarda en una carpeta según qué pasó con él:

    ocr_ok/               el OCR devolvió una lectura válida (el nombre lleva el texto)
    ocr_sin_lectura/      el OCR no devolvió ninguna lectura válida
    focus_mask/           lo descartó el filtro de forma/nitidez antes del OCR
    sin_patente_vehiculo/ RT-DETR no encontró patente: se guarda el VEHÍCULO, para ver si había una

Al terminar el video se escriben `indice.json` (todos los recortes, con lo que leyó el OCR en cada
intento) e `indice.html`, una hoja de contacto agrupada por vehículo que se abre con el navegador.
Sirve para distinguir un OCR flojo de un recorte mal hecho o de una patente que el detector no vio.
"""
import html
import json
import re
from pathlib import Path

from .contracts import write_image_atomic

KINDS = ("ocr_ok", "ocr_sin_lectura", "focus_mask", "sin_patente_vehiculo")
_KIND_LABEL = {
    "ocr_ok": "lectura válida", "ocr_sin_lectura": "OCR sin lectura válida",
    "focus_mask": "descartado por focus mask", "sin_patente_vehiculo": "RT-DETR no encontró patente",
}
_KIND_COLOR = {"ocr_ok": "#2e7d32", "ocr_sin_lectura": "#ef6c00", "focus_mask": "#6a1b9a",
               "sin_patente_vehiculo": "#c62828"}


def _safe(text):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))[:24]


class PlateCropExporter:
    def __init__(self, out_dir, video_stem="video"):
        self.out_dir = Path(out_dir)
        self.video_stem = video_stem
        self.items = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, kind, frame_idx, track_id, image, vehicle_class=None, text=None, conf=None,
            detail=None, attempts=None):
        """Guarda `image` y anota el recorte. Devuelve la ruta relativa, o None si no se pudo escribir."""
        if kind not in KINDS:
            raise ValueError(f"kind desconocido: {kind}")
        if image is None or image.size == 0:
            return None
        name = f"F{int(frame_idx):06d}_ID{int(track_id)}"
        if text:
            name += f"_{_safe(text)}"
        rel = f"{kind}/{name}.jpg"
        try:
            write_image_atomic(self.out_dir / rel, image, jpeg_quality=92)
        except (OSError, ValueError):
            return None
        self.items.append({
            "archivo": rel, "tipo": kind, "frame": int(frame_idx), "id_vehiculo": int(track_id),
            "tipo_vehiculo": vehicle_class, "texto": text,
            "confianza": None if conf is None else round(float(conf), 3), "detalle": detail,
            "intentos": attempts or [], "ancho": int(image.shape[1]), "alto": int(image.shape[0])})
        return rel

    def summary(self):
        counts = {k: 0 for k in KINDS}
        for it in self.items:
            counts[it["tipo"]] += 1
        return counts

    def finish(self):
        """Escribe indice.json e indice.html. Devuelve el conteo por tipo."""
        counts = self.summary()
        data = {"video": self.video_stem, "conteo": counts, "recortes": self.items}
        (self.out_dir / "indice.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        (self.out_dir / "indice.html").write_text(render_html(data), encoding="utf-8")
        return counts


def _caption(it):
    parts = [f"frame {it['frame']}", _KIND_LABEL[it["tipo"]]]
    if it["texto"]:
        conf = "" if it["confianza"] is None else f" ({it['confianza']:.2f})"
        parts.append(f"lectura: {it['texto']}{conf}")
    if it["detalle"]:
        parts.append(str(it["detalle"]))
    parts.append(f"{it['ancho']}x{it['alto']} px")
    return " · ".join(parts)


def _attempts(it):
    """Una línea con lo que leyó el OCR en cada intento: variante: leído -> corregido (resultado)."""
    parts = []
    for a in it["intentos"]:
        if not isinstance(a, dict):
            continue
        leido, corregido = a.get("texto_leido"), a.get("texto_corregido")
        txt = "sin texto" if not leido else (f"{leido} → {corregido}" if corregido and corregido != leido
                                             else str(leido))
        parts.append(f"{a.get('variant', '?')}: {txt} ({a.get('resultado', '?')})")
    return "OCR: " + " | ".join(parts) if parts else ""


def render_html(data):
    by_vehicle = {}
    for it in data["recortes"]:
        by_vehicle.setdefault(it["id_vehiculo"], []).append(it)
    out = ["<!doctype html><html lang='es'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>Recortes de patente — {html.escape(data['video'])}</title><style>",
           ":root{--bg:#fff;--fg:#1a1a1a;--card:#f4f4f4}"
           "@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--card:#262626}}"
           "body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}"
           ".veh{margin:0 0 24px}.row{display:flex;flex-wrap:wrap;gap:10px}"
           ".it{background:var(--card);border-left:6px solid;padding:6px;max-width:340px}"
           ".it img{min-width:160px;max-width:328px;max-height:240px;display:block}.cap{font-size:12px;margin-top:4px}"
           "</style></head><body>", f"<h1>{html.escape(data['video'])}</h1>",
           "<p>" + " · ".join(f"{html.escape(_KIND_LABEL[k])}: {v}" for k, v in data["conteo"].items())
           + "</p>"]
    for tid in sorted(by_vehicle):
        items = sorted(by_vehicle[tid], key=lambda i: i["frame"])
        cls = items[0].get("tipo_vehiculo") or ""
        out.append(f"<div class='veh'><h2>Vehículo #{tid} {html.escape(str(cls))}</h2><div class='row'>")
        for it in items:
            att = _attempts(it)
            out.append(
                f"<div class='it' style='border-color:{_KIND_COLOR[it['tipo']]}'>"
                f"<img loading='lazy' src='{html.escape(it['archivo'], quote=True)}'>"
                f"<div class='cap'>{html.escape(_caption(it))}"
                + (f"<br>{html.escape(att)}" if att else "") + "</div></div>")
        out.append("</div></div>")
    out.append("</body></html>")
    return "\n".join(out)
