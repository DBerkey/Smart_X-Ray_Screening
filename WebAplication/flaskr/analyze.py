"""
X-Ray Image Analysis Module for Smart X-Ray Screening Web Application.

This module handles the processing and analysis of uploaded X-ray images, including:
- File upload validation and processing
- Image preprocessing
- ML model prediction
- Result generation and storage
- Image serving functionality

The module integrates with the preprocessing pipeline and ML models to analyze
X-ray images and detect various medical conditions.
"""

import os
from PIL import Image, UnidentifiedImageError
import json
import mimetypes
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
import sys, json
from typing import Tuple, Optional
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Preprocessing.preprocessing import preprocess_xray
from Model.predict_with_models import predict

from flask import (
    Blueprint, current_app, request, redirect, url_for, session, abort, send_file
)
from werkzeug.utils import secure_filename
from PIL import Image  

# Blueprint for handling X-ray image analysis routes
bp = Blueprint("analyze", __name__)

# Set of allowed file extensions for X-ray image uploads
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "gif"}

def _interfaces_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Get paths to the Interfaces directories used for image processing.

    Returns:
        Tuple[Path, Path, Path, Path]: A tuple containing paths to:
            - Interfaces root directory
            - Input directory (for uploaded images)
            - PreProcess directory (for intermediate processing)
            - Processed directory (for final results)

    Note:
        Creates the directories if they don't exist.
    """
    repo_root = Path(current_app.root_path).parents[1]   
    root = repo_root / "Interfaces"                      
    input_dir = root / "Input"
    preproc_dir = root / "PreProcess"
    processed_dir = root / "Processed"

    input_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return root, input_dir, preproc_dir, processed_dir

def _is_valid_image(path: Path) -> bool:
    """
    Validate if a file is a supported image format.

    Args:
        path (Path): Path to the image file to validate.

    Returns:
        bool: True if the file is a valid image in a supported format,
              False otherwise.
    """
    try:
        with Image.open(path) as im:
            im.verify()  # validate file integrity
            fmt = (im.format or "").lower()
        return fmt in {"png", "jpeg", "jpg", "bmp"}
    except UnidentifiedImageError:
        return False
    except Exception:
        return False

def _save_direct_to_interfaces_input(upload_file) -> Path:
    """
    Save an uploaded file directly to the Interfaces/Input directory.

    Args:
        upload_file: The uploaded file object from Flask's request.files

    Returns:
        Path: Path to the saved file in the Input directory.
            The filename includes a timestamp to ensure uniqueness.
    """
    _, input_dir, _, _ = _interfaces_paths()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    orig_name = secure_filename(upload_file.filename)
    saved_name = f"{ts}_{orig_name}"
    saved_path = input_dir / saved_name
    upload_file.save(saved_path)
    return saved_path

def _try_run_preprocessing_pipeline(input_image_path: Path) -> Path:
    """
    Run the preprocessing pipeline on the uploaded image.
    Args:
        input_image_path (Path): Path to the input image file.
    Returns:
        out_img_path (Path): Path to the pre-processed image file.
    Raises:
        ValueError: If the preprocessing does not return a valid numpy array.
    """
    _, _, preproc_dir, processed_dir = _interfaces_paths()
    preproc_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Run the existing preprocessing on a single image
    out = preprocess_xray(
        str(input_image_path),
        output_size=(500, 500),
        process_types=['standard']  
    )
    if not isinstance(out, np.ndarray):
        raise ValueError("preprocess_xray() did not return a numpy array")

    # Ensure a 2D uint8 image for cv2.imwrite
    if out.ndim == 3 and out.shape[2] > 1:
        gray = np.mean(out, axis=2).astype(np.uint8)
    elif out.ndim == 3:
        gray = out[:, :, 0].astype(np.uint8)
    else:
        gray = out.astype(np.uint8)

    out_img_path = preproc_dir / "pre_processed_img.png"
    cv2.imwrite(str(out_img_path), gray)

    # Minimal results JSON (update later with real detections)
    results_json = {
        "findings": [],
        "overall_conf": None
    }
    (processed_dir / "processed_data.json").write_text(
        json.dumps(results_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return out_img_path



def _load_interfaces_outputs(preproc_dir: Path, processed_dir: Path):
    """
    Returns (annotated_path: Optional[Path], findings: list, overall_conf: None).

    JSON supported:
      A) {"findings":[{"label": "...", "score": 0.87, "bbox": [x,y,w,h]?}]}
      B) {"Disease":{"Sureness": 87, "Box_x": a, "Box_y": b, ...}, ...}
    Args:
        preproc_dir (Path): Path to the PreProcess directory.
        processed_dir (Path): Path to the Processed directory.
    Returns:
        Tuple[Optional[Path], list, None]:
            - annotated_path (Optional[Path]): Path to the annotated image if available, else None.
            - findings (list): List of findings extracted from the processed_data.json.
            - overall_conf (None): Placeholder for overall confidence, currently None.
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
    """
    Convert age input to a standardized format.

    Args:
        value: The age value from the form input.

    Returns:
        Union[int, str, None]: Standardized age value where:
            - None for invalid or empty input
            - "Below 1" for age 0
            - "100+" for age 100 or greater
            - int for all other valid ages
    """
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
    """
    Get the current UTC time in ISO 8601 format with 'Z' timezone designator.

    Returns:
        str: Current UTC time in format YYYY-MM-DDTHH:mm:ss.sssZ
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Routes
@bp.route("/analyze", methods=["POST"])
def analyze_upload():
    """
    Handle the upload and analysis of X-ray images.

    Processes form data including:
    - Uploaded X-ray image
    - Patient information (sex, age, view)
    
    Workflow:
    1. Validates the uploaded file and patient data
    2. Saves the image and metadata to Interfaces/Input
    3. Runs preprocessing pipeline
    4. Executes ML model prediction
    5. Stores results and redirects to results page

    Returns:
        Response: Redirect to results page on success
        
    Raises:
        400: If required fields are missing or invalid
        404: If processing fails
    """
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
    preproc_img_path = _try_run_preprocessing_pipeline(saved_in_input)

    # ML model prediction
    # --- CONFIG ---
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../Model')
    TRAINED_MODELS_DIR = os.path.join(MODEL_DIR, 'trained_models')
    SCALER_PATH = os.path.join(TRAINED_MODELS_DIR, 'feature_scaler.pkl')
    stage1_model_path = os.path.join(TRAINED_MODELS_DIR, 'svm_stage1_binary.pkl')
    stage2_model_paths = {
        "Atelectasis": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Atelectasis.pkl'),
        "Cardiomegaly": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Cardiomegaly.pkl'),
        "Consolidation": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Consolidation.pkl'),
        "Edema": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Edema.pkl'),
        "Effusion": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Effusion.pkl'),
        "Emphysema": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Emphysema.pkl'),
        "Fibrosis": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Fibrosis.pkl'),
        "Hernia": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Hernia.pkl'),
        "Infiltration": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Infiltration.pkl'),
        "Mass": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Mass.pkl'),
        "Nodule": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Nodule.pkl'),
        "Pleural_Thickening": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Pleural_Thickening.pkl'),
        "Pneumonia": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Pneumonia.pkl'),
        "Pneumothorax": os.path.join(TRAINED_MODELS_DIR, 'svm_stage2_Pneumothorax.pkl'),
    }

    # Predict using the ML models
    finding_pred, stage2_results = predict(
        str(preproc_img_path), age, sex, view,
        stage1_model_path, stage2_model_paths, scaler_path=SCALER_PATH
    )

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

    # Convert numpy types to native Python types for JSON serialization
    def to_py(val):
        if hasattr(val, 'item'):
            return val.item()
        return int(val) if isinstance(val, (np.integer,)) else float(val) if isinstance(val, (np.floating,)) else val

    py_finding_pred = to_py(finding_pred)
    py_stage2_results = {k: to_py(v) for k, v in stage2_results.items()} if stage2_results else None

    # Determine result stage
    if py_finding_pred == 0:
        result_stage = 1  # No findings
    elif py_finding_pred == 1 and py_stage2_results:
        if all(v == 0 for v in py_stage2_results.values()):
            result_stage = 2  # Findings, but unknown which
        else:
            result_stage = 3  # Findings, likely this
    else:
        result_stage = 2  # Default to stage 2 if ambiguous

    # If stage 3, write detected findings to processed_data.json for display
    if result_stage == 3 and py_stage2_results:
        findings_list = []
        for label, value in py_stage2_results.items():
            if value == 1:
                findings_list.append({"label": label, "score": 1.0, "bbox": None})
        _, _, preproc_dir, processed_dir = _interfaces_paths()
        results_json = {
            "findings": findings_list,
            "overall_conf": None
        }
        (processed_dir / "processed_data.json").write_text(
            json.dumps(results_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        # Update findings for payload
        findings = findings_list
    # Build session payload & redirect
    payload = {
        "image_area": area,
        "image_name": image_name,
        "input_image_name": saved_in_input.name,
        "num_findings": len(findings),
        "findings": findings,
        "patient": {"sex": sex, "age": age, "view": view} if (sex or age or view is not None) else None,
        "generated_at": _now_utc_z(),
        "ml_finding_pred": py_finding_pred,
        "ml_stage2_results": py_stage2_results,
        "result_stage": result_stage,
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
    Args:
        area (str): One of "Input", "PreProcess", "Processed".
        filename (str): Filename to serve.
    Returns:
        Response: Flask response with the image file.
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
