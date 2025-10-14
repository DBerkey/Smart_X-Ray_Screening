
from flask import (
    Blueprint, render_template, session, url_for, redirect, send_file
)
from io import BytesIO

bp = Blueprint("results", __name__, url_prefix="/results")



@bp.route("/", methods=["GET"])
def show_results():
    data = session.get("latest_result")
    if not data:
        return redirect(url_for("home.index"))

    image_filename = data.get("image_filename")
    image_url = url_for("analyze.get_uploaded", filename=image_filename) if image_filename else None

    findings = data.get("findings", []) or []
    overall_conf = data.get("overall_conf")  # <-- no fallback computation

    return render_template(
        "results/results.html",
        image_url=image_url,
        num_findings=data.get("num_findings", len(findings)),
        overall_conf=overall_conf,
        findings=findings,
        patient=data.get("patient"),
        generated_at=data.get("generated_at"),
    )


@bp.route("/export", methods=["GET"])
def export_pdf():
    """
    Generate a PDF export of the current session results.
    Requires 'reportlab' (pip install reportlab). If not installed, we redirect back.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except Exception:
        # ReportLab not installed — just go back to results page
        return redirect(url_for("results.show_results"))

    data = session.get("latest_result") or {}
    findings = data.get("findings", []) or []
    overall_conf = data.get("overall_conf")

    generated_at = data.get("generated_at", "")
    patient = data.get("patient", {}) or {}

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 20 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Smart X-Ray Screening — Results")
    y -= 10 * mm

    c.setFont("Helvetica", 11)
    if generated_at:
        c.drawString(20 * mm, y, f"Generated: {generated_at}")
        y -= 8 * mm

    # Patient summary line
    p_sex = patient.get("sex", "—")
    p_age = patient.get("age", "—")
    c.drawString(20 * mm, y, f"Patient: Sex={p_sex}, Age={p_age}")
    y -= 12 * mm

    # Summary cards
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Diseases Found: {len(findings)}")
    y -= 8 * mm
    if overall_conf is not None:
     pct = round(float(overall_conf) * 100.0, 1)
     c.drawString(20*mm, y, f"Overall Confidence: {pct}%")
    else:
     c.drawString(20*mm, y, "Overall Confidence: No confidence rate available")
    y -= 10*mm

    # Findings list
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Findings:")
    y -= 7 * mm
    c.setFont("Helvetica", 11)

    for i, f in enumerate(findings, start=1):
        label = f.get("label", "—")
        score = f.get("score")
        bbox = f.get("bbox")
        pct_str = f"{round(float(score) * 100.0, 1)}%" if score is not None else "—"
        line = f"{i}. {label} — {pct_str}"
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            line += f", bbox={bbox}"
        c.drawString(25 * mm, y, line)
        y -= 6 * mm

        # paginate if we run out of space
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="results.pdf",
    )
