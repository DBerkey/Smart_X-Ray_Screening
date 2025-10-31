# flaskr/analyze.py
from PIL import Image, UnidentifiedImageError
import json
import shutil
import mimetypes
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional

from flask import (
    Blueprint, current_app, request, redirect, url_for, session, abort, send_file
)
from werkzeug.utils import secure_filename
from PIL import Image  

bp = Blueprint("analyze", __name__)

ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "gif"}

# Return Interfaces, Interfaces/Input, Interfaces/PreProcess, Interfaces/Processed
def _interfaces_paths() -> Tuple[Path, Path, Path, Path]:
   
    project_root = Path(current_app.root_path).parent  # WebAplication/
    root = project_root / "Interfaces"
    input_dir = root / "Input"
    preproc_dir = root / "PreProcess"
    processed_dir = root / "Processed"

    input_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return root, input_dir, preproc_dir, processed_dir

def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # validate file integrity
            fmt = (im.format or "").lower()
        return fmt in {"png", "jpeg", "jpg", "gif", "bmp"}
    except UnidentifiedImageError:
        return False
    except Exception:
        return False

# Save file to Interfaces/Input, returns path
def _save_direct_to_interfaces_input(upload_file) -> Path:
    _, input_dir, _, _ = _interfaces_paths()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    orig_name = secure_filename(upload_file.filename)
    saved_name = f"{ts}_{orig_name}"
    saved_path = input_dir / saved_name
    upload_file.save(saved_path)
    return saved_path

# Run recognition pipeline
def _try_run_interfaces_pipeline(copied_input: Path) -> None:
    """
    OPTIONAL: invoke your pipeline here (script or Python API).
    Example:
        import subprocess
        root, _, _, _ = _interfaces_paths()
        subprocess.check_call(
            ["python", "main.py", "--input", str(copied_input)],
            cwd=root
        )
    """
    return


def _load_interfaces_outputs(preproc_dir: Path, processed_dir: Path):
    """
    Returns (annotated_path: Optional[Path], findings: list, overall_conf: None).

    JSON supported:
      A) {"findings":[{"label": "...", "score": 0.87, "bbox": [x,y,w,h]?}]}
      B) {"Disease":{"Sureness": 87, "Box_x": a, "Box_y": b, ...}, ...}
    """
    def _norm_score(v):
        try:
            v = float(v)
        except Exception:
            return None
        return v/100.0 if v > 1.0 else v

    annotated_path: Optional[Path] = None
    findings = []

    try:
        # Prefer known pre-process preview; else first image in PreProcess
        if preproc_dir and preproc_dir.exists():
            candidates = [
                preproc_dir / "pre_processed_img.png",
                preproc_dir / "pre_processed_img.jpg",
                preproc_dir / "pre_processed_img.jpeg",
            ]
            annotated_path = next((p for p in candidates if p.exists()), None)
            if annotated_path is None:
                imgs = sorted([p for p in preproc_dir.glob("*") if _is_valid_image(p)])
                if imgs:
                    annotated_path = imgs[0]

        # Findings only (no overall)
        if processed_dir and processed_dir.exists():
            data_json = processed_dir / "processed_data.json"
            if data_json.exists():
                try:
                    data = json.loads(data_json.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "findings" in data:
                        raw = data.get("findings") or []
                        out = []
                        for f in raw:
                            label = f.get("label")
                            score = _norm_score(f.get("score"))
                            bbox = f.get("bbox")
                            out.append({"label": label, "score": score, "bbox": bbox})
                        findings = out
                    elif isinstance(data, dict):
                        out = []
                        for label, attrs in data.items():
                            if not isinstance(attrs, dict):
                                continue
                            score = _norm_score(attrs.get("Sureness"))
                            # Keep bbox only if you actually write it in JSON; else None
                            bbox = None
                            out.append({"label": label, "score": score, "bbox": bbox})
                        findings = out
                except Exception:
                    pass
    except Exception:
        pass

    return annotated_path, findings, None

# Form helpers
def _coerce_age(value):
    if value in (None, "", "Choose..."):
        return None
    if value == "0":
        return "Below 1"
    if value == "100+":
        return "100+"
    try:
        return int(value)
    except Exception:
        return None

def _now_utc_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Routes
@bp.route("/analyze", methods=["POST"])
def analyze_upload():
    # Validate file
    f = request.files.get("image")
    if not f or not f.filename:
        abort(400, description="No image selected.")
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXTS:
        abort(400, description="Unsupported image type.")

    # Save directly into Interfaces/Input
    saved_in_input = _save_direct_to_interfaces_input(f)
    if not _is_valid_image(saved_in_input):
        saved_in_input.unlink(missing_ok=True)
        abort(400, description="Uploaded file is not a valid image.")

    # Parse patient fields
    sex = request.form.get("sex") or None
    age_raw = request.form.get("age")
    view = request.form.get("view") or None

    # Enforce required fields
    if not sex or not age_raw or not view:
        abort(400, description="Sex, age, and view are required.")

    age = _coerce_age(age_raw)
    if age is None:
        abort(400, description="Invalid age selection.")

    # Auto-build JSON next to the image in Interfaces/Input
    mime = f.mimetype or mimetypes.guess_type(f.filename)[0]
    filesize = saved_in_input.stat().st_size
    width = height = None
    fmt = None
    try:
        with Image.open(saved_in_input) as im:
            width, height = im.size
            fmt = im.format
    except Exception:
        pass

    auto_json = {
        "Image_Index": secure_filename(f.filename),  # original name
        "Age": age,
        "Sex": sex,
        "View": view,
        "ImageSize": {"width": width, "height": height} if width and height else None,
        "ImagePixelSpacing": [0, 0],
        "Uploaded_Filename": saved_in_input.name,
        "Mime_Type": mime,
        "File_Size_Bytes": filesize,
        "Format": fmt,
    }

    try:
        _, input_dir, _, _ = _interfaces_paths()
        json_name = f"{Path(f.filename).stem}_patient_data.json"
        (input_dir / json_name).write_text(
            json.dumps(auto_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

    # Run your Interfaces pipeline
    _try_run_interfaces_pipeline(saved_in_input)

    # Read Interfaces outputs
    _, _, preproc_dir, processed_dir = _interfaces_paths()
    annotated_path, findings, overall_conf = _load_interfaces_outputs(preproc_dir, processed_dir)

    if annotated_path and annotated_path.exists():
        if processed_dir and annotated_path.parent == processed_dir:
            area, image_name = "Processed", annotated_path.name
        else:
            area, image_name = "PreProcess", annotated_path.name
    else:
        area, image_name = "Input", saved_in_input.name

    # Build session payload & redirect
    payload = {
    "image_area": area,
    "image_name": image_name,
    "num_findings": len(findings),
    "findings": findings,
    "patient": {"sex": sex, "age": age, "view": view} if (sex or age or view is not None) else None,
    "generated_at": _now_utc_z(),
    }
    session["latest_result"] = payload
    return redirect(url_for("results.show_results"))

@bp.route("/interfaces-image/<area>/<path:filename>")
def serve_interfaces_image(area, filename):
    """
    Serve images from Interfaces/{Input,PreProcess,Processed}.

    - Input (original upload):       read -> send (KEEP)
    - PreProcess (intermediate):     read -> send -> DELETE (ephemeral)
    - Processed (final annotated):   read -> send (KEEP)
    """
    _, input_dir, preproc_dir, processed_dir = _interfaces_paths()
    if area not in {"Input", "PreProcess", "Processed"}:
        abort(404)

    if area == "Input":
        path = input_dir / filename
    elif area == "PreProcess":
        path = preproc_dir / filename
    else:  # "Processed"
        path = processed_dir / filename

    if not path.exists() or not path.is_file():
        abort(404)

    data = path.read_bytes()

    # Only delete PreProcess preview images after serving
    if area == "PreProcess":
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    mime, _ = mimetypes.guess_type(str(filename))
    return send_file(BytesIO(data), mimetype=mime or "application/octet-stream",
                     as_attachment=False, download_name=filename)
