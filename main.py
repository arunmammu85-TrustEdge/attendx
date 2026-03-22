"""
AttendX - Python Backend (FastAPI + InsightFace)
Free, open-source, commercial-OK
Handles: Face recognition, Geofence validation, Attendance logging
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import json
import os
import math
from datetime import datetime, date
from io import BytesIO
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AttendX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")   # service role key (backend only)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# InsightFace — ArcFace model (99%+ accuracy, Apache 2.0)
print("Loading InsightFace ArcFace model…")
face_app = FaceAnalysis(
    name="buffalo_l",             # ArcFace ResNet100 — most accurate free model
    providers=["CPUExecutionProvider"]  # use CUDAExecutionProvider if GPU available
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model loaded ✓")

# Office geofence config — update with your actual coords
OFFICE_LAT  = float(os.getenv("OFFICE_LAT",  "17.3850"))
OFFICE_LON  = float(os.getenv("OFFICE_LON",  "78.4867"))
FENCE_RADIUS_M = int(os.getenv("FENCE_RADIUS", "500"))   # metres

# Similarity threshold for face match (0.0–1.0). 0.45 = strict, 0.35 = lenient
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.42"))

security = HTTPBearer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in metres between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def decode_image(data: str) -> np.ndarray:
    """Decode base64 image string → OpenCV numpy array."""
    if "base64," in data:
        data = data.split("base64,")[1]
    img_bytes = base64.b64decode(data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def get_face_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """Run InsightFace on image, return 512-dim ArcFace embedding or None."""
    faces = face_app.get(img)
    if not faces:
        return None
    # Use the largest / highest-confidence face
    face = max(faces, key=lambda f: f.det_score)
    if face.det_score < 0.70:   # low confidence detection
        return None
    return face.embedding        # shape: (512,)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def liveness_check(img: np.ndarray) -> bool:
    """
    Basic liveness: checks face is not a flat printed photo.
    Uses Laplacian variance (blur detection) as a simple anti-spoofing layer.
    For production, use a dedicated liveness model.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > 80   # printed photos tend to be blurry / low-variance


# ── Models ────────────────────────────────────────────────────────────────────

class PunchRequest(BaseModel):
    employee_id: str
    image_base64: str         # webcam snapshot (JPEG base64)
    latitude: float
    longitude: float
    punch_type: str           # "in" or "out"
    method: str               # "face", "thumb", "dual"


class RegisterFaceRequest(BaseModel):
    employee_id: str
    image_base64: str


class LocationCheckRequest(BaseModel):
    latitude: float
    longitude: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "InsightFace ArcFace buffalo_l", "time": datetime.utcnow().isoformat()}


# ── 1. Register employee face ─────────────────────────────────────────────────
@app.post("/face/register")
async def register_face(req: RegisterFaceRequest):
    """
    Enroll an employee's face. Stores the 512-dim ArcFace embedding in Supabase.
    Call this once per employee during onboarding (can re-enroll anytime).
    """
    try:
        img = decode_image(req.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid image data")

    embedding = get_face_embedding(img)
    if embedding is None:
        raise HTTPException(422, "No face detected in the image. Please use a clear front-facing photo.")

    # Store embedding as JSON array in Supabase
    result = supabase.table("face_embeddings").upsert({
        "employee_id": req.employee_id,
        "embedding": embedding.tolist(),
        "registered_at": datetime.utcnow().isoformat(),
    }).execute()

    return {"success": True, "message": "Face registered successfully"}


# ── 2. Location check ─────────────────────────────────────────────────────────
@app.post("/location/check")
def check_location(req: LocationCheckRequest):
    """Returns whether coordinates are within the office geofence."""
    dist = haversine_distance(OFFICE_LAT, OFFICE_LON, req.latitude, req.longitude)
    in_zone = dist <= FENCE_RADIUS_M
    return {
        "in_zone": in_zone,
        "distance_m": round(dist),
        "fence_radius_m": FENCE_RADIUS_M,
        "message": f"{'Within' if in_zone else 'Outside'} office zone ({round(dist)}m away)"
    }


# ── 3. Main punch endpoint ────────────────────────────────────────────────────
@app.post("/attendance/punch")
async def punch(req: PunchRequest):
    """
    Full attendance punch flow:
    1. Geofence check (Option D — enforced for both IN and OUT)
    2. Liveness detection
    3. Face recognition (ArcFace cosine similarity)
    4. Log to Supabase attendance_logs table
    """
    now = datetime.utcnow()
    today = date.today().isoformat()

    # ── Step 1: Geofence check ────────────────────────────────────────────────
    dist = haversine_distance(OFFICE_LAT, OFFICE_LON, req.latitude, req.longitude)
    if dist > FENCE_RADIUS_M:
        raise HTTPException(403, {
            "code": "OUTSIDE_GEOFENCE",
            "message": f"Punch-{req.punch_type.upper()} blocked. You are {round(dist)}m from office (limit: {FENCE_RADIUS_M}m).",
            "distance_m": round(dist),
        })

    # ── Step 2: Decode image + liveness ──────────────────────────────────────
    if req.method in ("face", "dual"):
        try:
            img = decode_image(req.image_base64)
        except Exception:
            raise HTTPException(400, "Invalid image data")

        if not liveness_check(img):
            raise HTTPException(422, {
                "code": "LIVENESS_FAILED",
                "message": "Liveness check failed. Please use a live camera, not a photo."
            })

        # ── Step 3: Face recognition ──────────────────────────────────────────
        live_embedding = get_face_embedding(img)
        if live_embedding is None:
            raise HTTPException(422, {
                "code": "NO_FACE_DETECTED",
                "message": "No face detected. Please look directly at the camera."
            })

        # Fetch stored embedding from Supabase
        stored = supabase.table("face_embeddings") \
            .select("embedding") \
            .eq("employee_id", req.employee_id) \
            .single() \
            .execute()

        if not stored.data:
            raise HTTPException(404, {
                "code": "NOT_ENROLLED",
                "message": "Face not enrolled. Please register your face with HR first."
            })

        stored_embedding = np.array(stored.data["embedding"])
        similarity = cosine_similarity(live_embedding, stored_embedding)

        if similarity < FACE_THRESHOLD:
            # Log failed attempt
            supabase.table("failed_attempts").insert({
                "employee_id": req.employee_id,
                "attempt_type": "face_mismatch",
                "similarity_score": round(similarity, 4),
                "latitude": req.latitude,
                "longitude": req.longitude,
                "attempted_at": now.isoformat(),
            }).execute()
            raise HTTPException(401, {
                "code": "FACE_MISMATCH",
                "message": f"Face not recognised (confidence: {round(similarity*100, 1)}%). Please try again.",
                "similarity": round(similarity, 4),
            })

    # ── Step 4: Check for duplicate punch ────────────────────────────────────
    existing = supabase.table("attendance_logs") \
        .select("id, punch_in_time, punch_out_time") \
        .eq("employee_id", req.employee_id) \
        .eq("date", today) \
        .execute()

    if req.punch_type == "in":
        if existing.data:
            raise HTTPException(409, {"code": "ALREADY_PUNCHED_IN", "message": "Already punched in today."})

        # Insert new attendance record
        result = supabase.table("attendance_logs").insert({
            "employee_id": req.employee_id,
            "date": today,
            "punch_in_time": now.isoformat(),
            "punch_in_lat": req.latitude,
            "punch_in_lon": req.longitude,
            "punch_in_dist_m": round(dist),
            "method": req.method,
            "status": "present" if now.hour < 9 or (now.hour == 9 and now.minute <= 15) else "late",
        }).execute()

    elif req.punch_type == "out":
        if not existing.data:
            raise HTTPException(400, {"code": "NOT_PUNCHED_IN", "message": "No punch-in record found for today."})
        if existing.data[0].get("punch_out_time"):
            raise HTTPException(409, {"code": "ALREADY_PUNCHED_OUT", "message": "Already punched out today."})

        punch_in_dt = datetime.fromisoformat(existing.data[0]["punch_in_time"])
        hours_worked = round((now - punch_in_dt).seconds / 3600, 2)

        result = supabase.table("attendance_logs").update({
            "punch_out_time": now.isoformat(),
            "punch_out_lat": req.latitude,
            "punch_out_lon": req.longitude,
            "punch_out_dist_m": round(dist),
            "hours_worked": hours_worked,
        }).eq("id", existing.data[0]["id"]).execute()

    # ── Step 5: Return success ────────────────────────────────────────────────
    return {
        "success": True,
        "punch_type": req.punch_type,
        "time": now.strftime("%I:%M %p"),
        "distance_m": round(dist),
        "similarity": round(similarity, 4) if req.method in ("face", "dual") else None,
        "message": f"Punch-{'In' if req.punch_type=='in' else 'Out'} successful at {now.strftime('%I:%M %p')}",
    }


# ── 4. Get today's attendance summary (for dashboard) ─────────────────────────
@app.get("/attendance/today")
def today_summary():
    today = date.today().isoformat()
    logs = supabase.table("attendance_logs").select("*, employees(name, department)").eq("date", today).execute()
    total_emp = supabase.table("employees").select("id", count="exact").execute().count
    present = len([l for l in logs.data if l["status"] in ("present", "late")])
    late    = len([l for l in logs.data if l["status"] == "late"])
    return {
        "date": today,
        "total_employees": total_emp,
        "present": present,
        "absent": total_emp - present,
        "late": late,
        "logs": logs.data,
    }


# ── 5. Employee attendance report ─────────────────────────────────────────────
@app.get("/attendance/report/{employee_id}")
def employee_report(employee_id: str, month: Optional[str] = None):
    query = supabase.table("attendance_logs").select("*").eq("employee_id", employee_id)
    if month:   # format: "2026-03"
        query = query.gte("date", f"{month}-01").lte("date", f"{month}-31")
    result = query.order("date", desc=True).execute()
    return {"employee_id": employee_id, "records": result.data}
