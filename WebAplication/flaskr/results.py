from io import BytesIO
from pathlib import Path

from flask import Blueprint, render_template, session, url_for, redirect, abort, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# Reuse helpers from analyze.py
from .analyze import _interfaces_paths  # provides Input/PreProcess/Processed directories

bp = Blueprint("results", __name__)

@bp.route("/results")
def show_results():
    payload = session.get("latest_result")
    if not payload:
        return redirect("/")
    image_url = url_for(
        "analyze.serve_interfaces_image",
        area=payload["image_area"],
        filename=payload["image_name"]
    )
    return render_template(
        "results/results.html",
        image_url=image_url,
        num_findings=payload.get("num_findings"),
        findings=payload.get("findings"),
        patient=payload.get("patient"),
        generated_at=payload.get("generated_at"),
        overall_conf=payload.get("overall_conf"),
    )

@bp.route("/results/export-pdf")
def export_pdf():
    payload = session.get("latest_result")
    if not payload:
        abort(404, "No result in session.")

    # Resolve the image path from the saved area/name
    _, input_dir, preproc_dir, processed_dir = _interfaces_paths()
    base = {"Input": input_dir, "PreProcess": preproc_dir, "Processed": processed_dir}.get(
        payload.get("image_area")
    )
    image_path = (base / payload.get("image_name")).resolve() if base else None

    # Fallback if the file was ephemeral or missing
    if not image_path or not image_path.exists():
        # Last resort: try to show the original upload instead of failing the export
        try:
            image_path = (input_dir / payload.get("image_name")).resolve()
        except Exception:
            image_path = None

    # Build PDF in-memory
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    margin = 36
    y = page_h - margin

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Smart X-Ray Screening — Results")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {payload.get('generated_at', '—')}")
    y -= 24

    # Image (keep aspect ratio; fit in upper half)
    max_w = page_w - 2*margin
    max_h = page_h * 0.45
    if image_path and image_path.exists():
        try:
            img = ImageReader(str(image_path))
            iw, ih = img.getSize()
            scale = min(max_w/iw, max_h/ih)
            draw_w, draw_h = iw*scale, ih*scale
            c.drawImage(img, margin, y - draw_h, draw_w, draw_h, preserveAspectRatio=True, anchor='n')
            y -= draw_h + 18
        except Exception:
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(margin, y, "(Image could not be embedded)")
            y -= 18
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, y, "(No image available)")
        y -= 18

    # Findings (per-finding confidence)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Detected Findings")
    y -= 14
    c.setFont("Helvetica", 10)
    findings = payload.get("findings") or []
    if not findings:
        c.drawString(margin, y, "None")
        y -= 12
    else:
        for i, f in enumerate(findings, 1):
            label = f.get("label", "—")
            score = f.get("score")
            score_txt = f"{round(float(score)*100,1)}%" if score is not None else "—"
            bbox = f.get("bbox")
            line = f"{i}. {label} — {score_txt}"
            if bbox:
                line += f"   bbox: {bbox}"
            # New page if needed
            if y < margin + 12:
                c.showPage(); y = page_h - margin; c.setFont("Helvetica", 10)
            c.drawString(margin, y, line)
            y -= 12

    # Patient section
    patient = payload.get("patient")
    if patient:
        if y < margin + 40:
            c.showPage(); y = page_h - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Patient")
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Sex: {patient.get('sex','—')}    Age: {patient.get('age','—')}")
        y -= 12

    c.showPage()
    c.save()
    buf.seek(0)
    return send_file(buf, download_name="xray_results.pdf", mimetype="application/pdf", as_attachment=True)