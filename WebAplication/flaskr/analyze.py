# flaskr/analyze.py
import imghdr
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import mimetypes
from io import BytesIO
from flask import Blueprint, current_app, request, redirect, url_for, session, abort, send_file
from werkzeug.utils import secure_filename

bp = Blueprint("analyze", __name__)

ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "gif"}



def _uploads_dir() -> Path:
    """Where we keep user uploads & the final annotated image we will serve."""
    p = Path(current_app.instance_path) / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _interfaces_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Returns (root, input_dir, preproc_dir, processed_dir) for your Interfaces layout:
      Interfaces/Interfaces/{Input, PreProcess, Processed}
    """
    # WebAplication/flaskr -> parent() -> WebAplication
    project_root = Path(current_app.root_path).parent
    root = project_root / "Interfaces" / "Interfaces"
    input_dir = root / "Input"
    preproc_dir = root / "PreProcess"
    processed_dir = root / "Processed"

    input_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return root, input_dir, preproc_dir, processed_dir

def _is_valid_image(path: Path) -> bool:
    kind = imghdr.what(path)
    return kind in {"png", "jpeg", "gif", "bmp"}


# -------------------------
# Helpers: pipeline I/O
# -------------------------
def _copy_to_interfaces_input(upload_path: Path) -> Path:
    """Copy the just-uploaded image into Interfaces/Input (keeping the filename)."""
    _, input_dir, _, _ = _interfaces_paths()
    target = input_dir / upload_path.name
    shutil.copy2(upload_path, target)
    return target

def _try_run_interfaces_pipeline(copied_input: Path) -> None:
    """
    OPTIONAL: If your Interfaces repo is driven by a script, invoke it here.
    Leave as-is if you’re manually populating the PreProcess/Processed folders for now.
    """
    # Example if you later add a runner script:
    # root, _, _, _ = _interfaces_paths()
    # subprocess.check_call(
    #     ["python", "main.py", "--input", str(copied_input)],
    #     cwd=root
    # )
    return

def _load_interfaces_outputs(preproc_dir: Path, processed_dir: Path):
    """
    Always returns (annotated_path: Optional[Path], findings: list, overall_conf: Optional[float]).
    Never returns None. Swallows errors and falls back to safe defaults.
    """
    annotated_path = None
    findings = []
    overall_conf = None

    try:
        # ---- 1) Find annotated image (prefer known names, else first image) ----
        if preproc_dir and preproc_dir.exists():
            candidates = [
                preproc_dir / "pre_processed_img.png",
                preproc_dir / "pre_processed_img.jpg",
                preproc_dir / "pre_processed_img.jpeg",
            ]
            annotated_path = next((p for p in candidates if p.exists()), None)

            if annotated_path is None:
                # fallback: any image file in PreProcess
                imgs = sorted([p for p in preproc_dir.glob("*") if _is_valid_image(p)])
                if imgs:
                    annotated_path = imgs[0]

        # ---- 2) Load findings from processed_data.json (if present) ----
        if processed_dir and processed_dir.exists():
            data_json = processed_dir / "processed_data.json"
            if data_json.exists():
                try:
                    data = json.loads(data_json.read_text(encoding="utf-8"))
                    # Two supported shapes:
                    # A) {"findings":[{label,score,bbox}], "overall_conf":0.87}
                    if isinstance(data, dict) and "findings" in data:
                        findings = data.get("findings") or []
                        overall_conf = data.get("overall_conf", None)
                    # B) {"DiseaseName":{"Sureness":x,"Box_x":...,"Box_y":...}, ...}
                    elif isinstance(data, dict):
                        tmp = []
                        for label, attrs in data.items():
                            if not isinstance(attrs, dict):
                                continue
                            score = attrs.get("Sureness")
                            bx = attrs.get("Box_x"); by = attrs.get("Box_y")
                            # If you only have a point (x,y), leave bbox None (template handles it)
                            bbox = None
                            tmp.append({"label": label, "score": score, "bbox": bbox})
                        findings = tmp
                        # derive overall_conf as avg of available scores
                        scores = [f["score"] for f in findings if f.get("score") is not None]
                        overall_conf = (sum(scores) / len(scores)) if scores else None
                except Exception:
                    # keep defaults on JSON error
                    pass

    except Exception:
        # keep defaults on any unexpected error
        pass

    # Always return a tuple
    return annotated_path, findings, overall_conf


# -------------------------
# Main routes
# -------------------------
@bp.route("/analyze", methods=["POST"])
def analyze_upload():
    # 0) Validate file in form
    f = request.files.get("image")
    if not f or not f.filename:
        abort(400, description="No image selected.")
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXTS:
        abort(400, description="Unsupported image type.")

    # 1) Save uploaded image to instance/uploads with unique name
    uploads_dir = _uploads_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    safe_name = secure_filename(f.filename)
    saved_name = f"{timestamp}_{safe_name}"
    saved_path = uploads_dir / saved_name
    f.save(saved_path)

    # sanity check content
    if not _is_valid_image(saved_path):
        saved_path.unlink(missing_ok=True)
        abort(400, description="Uploaded file is not a valid image.")

    # 2) Optional patient fields (from your checkbox section)
    sex = request.form.get("sex") or None
    age_raw = request.form.get("age") or None
    if age_raw == "100+":
        age_val = "100+"
    elif age_raw == "0":
        age_val = "Below 1"
    elif age_raw:
        try:
            age_val = int(age_raw)
        except ValueError:
            age_val = None
    else:
        age_val = None
    patient = {"sex": sex, "age": age_val} if (sex or age_val) else None

    # 3) Copy uploaded image into Interfaces/Input and (optionally) run the pipeline
    copied_to_input = _copy_to_interfaces_input(saved_path)
    _try_run_interfaces_pipeline(copied_to_input)

    # 4) Read outputs from Interfaces/{PreProcess,Processed}
    _, _, preproc_dir, processed_dir = _interfaces_paths()
    annotated_path, findings, overall_conf = _load_interfaces_outputs(preproc_dir, processed_dir)

    # If no annotated image produced yet, fall back to original upload
    final_annotated = annotated_path if annotated_path and annotated_path.exists() else saved_path

    # Ensure the annotated file is accessible via our /uploads route:
    # copy it into instance/uploads/ if it isn’t already there.
    if final_annotated.parent != uploads_dir:
        # name it with a suffix to avoid colliding with the original
        annotated_name = f"{timestamp}_annotated{final_annotated.suffix or '.png'}"
        target = uploads_dir / annotated_name
        shutil.copy2(final_annotated, target)
        final_annotated = target

    # 5) Build payload for the Results page
    payload = {
        "image_filename": final_annotated.name,
        "num_findings": len(findings),
        "overall_conf": overall_conf,
        "findings": findings,
        "patient": patient,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # 6) Stash & redirect
    session["latest_result"] = payload
    return redirect(url_for("results.show_results"))


@bp.route("/uploads/<path:filename>")
def get_uploaded(filename):
    path = _uploads_dir() / filename
    if not path.exists():
        abort(404)

    # Read into memory, then delete the file so it doesn't persist
    data = path.read_bytes()
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

    mime, _ = mimetypes.guess_type(str(filename))
    return send_file(
        BytesIO(data),
        mimetype=mime or "application/octet-stream",
        as_attachment=False,
        download_name=filename,
    )