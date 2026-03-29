"""
AttendX - Python Backend (FastAPI + InsightFace)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import os
import math
import traceback
import bcrypt
from datetime import datetime, date, timezone
from supabase import create_client, Client
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AttendX API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading InsightFace ArcFace model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model loaded")

OFFICE_LAT     = float(os.getenv("OFFICE_LAT",    "17.3850"))
OFFICE_LON     = float(os.getenv("OFFICE_LON",    "78.4867"))
FENCE_RADIUS_M = int(os.getenv("FENCE_RADIUS",    "500"))
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.42"))
TEST_MODE      = os.getenv("TEST_MODE", "false").lower() == "true"
print(f"TEST_MODE={TEST_MODE} fence={FENCE_RADIUS_M}m threshold={FACE_THRESHOLD}")

class PunchRequest(BaseModel):
    employee_id: str
    image_base64: Optional[str] = None      # face method
    finger_template: Optional[str] = None   # fingerprint method
    punch_pin: Optional[str] = None         # PIN method
    latitude: float
    longitude: float
    punch_type: str
    method: str                             # 'face' | 'thumb' | 'pin'

class RegisterFaceRequest(BaseModel):
    employee_id: str
    image_base64: str

class RegisterFingerRequest(BaseModel):
    employee_id: str
    template: str           # base64 minutiae template from scanner SDK
    finger: Optional[str] = "right_index"

class SetPinRequest(BaseModel):
    employee_id: str
    pin: str                # plain 6-digit, will be hashed server-side

class LocationCheckRequest(BaseModel):
    latitude: float
    longitude: float

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

def decode_image(data):
    if "base64," in data:
        data = data.split("base64,")[1]
    arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def get_embedding(img):
    faces = face_app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    return face.embedding if face.det_score >= 0.60 else None

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)))

def liveness_ok(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Liveness variance: {v:.1f}")
    return v > 15

@app.get("/health")
def health():
    return {"status": "ok", "model": "InsightFace ArcFace buffalo_l", "test_mode": TEST_MODE, "time": datetime.now(timezone.utc).isoformat()}

@app.post("/face/register")
async def register_face(req: RegisterFaceRequest):
    try:
        img = decode_image(req.image_base64)
        if img is None:
            raise HTTPException(400, {"code": "BAD_IMAGE", "message": "Could not decode image."})
        emb = get_embedding(img)
        if emb is None:
            raise HTTPException(422, {"code": "NO_FACE", "message": "No face detected. Use a clear front-facing photo."})
        supabase.table("face_embeddings").upsert({"employee_id": req.employee_id, "embedding": emb.tolist(), "registered_at": datetime.now(timezone.utc).isoformat()}).execute()
        return {"success": True, "message": "Face registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.post("/finger/register")
async def register_finger(req: RegisterFingerRequest):
    """Store a fingerprint minutiae template (base64) from scanner SDK."""
    try:
        if not req.template or len(req.template) < 10:
            raise HTTPException(400, {"code": "BAD_TEMPLATE", "message": "Fingerprint template is empty or invalid."})
        supabase.table("finger_templates").upsert({
            "employee_id": req.employee_id,
            "template": req.template,
            "finger": req.finger or "right_index",
            "enrolled_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return {"success": True, "message": "Fingerprint enrolled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.post("/employees/set-pin")
async def set_employee_pin(req: SetPinRequest):
    """Admin sets or resets a 6-digit punch PIN for an employee."""
    try:
        if not req.pin or not req.pin.isdigit() or len(req.pin) != 6:
            raise HTTPException(400, {"code": "INVALID_PIN", "message": "PIN must be exactly 6 digits."})
        hashed = bcrypt.hashpw(req.pin.encode(), bcrypt.gensalt()).decode()
        supabase.table("employees").update({
            "punch_pin_hash": hashed,
            "pin_attempts": 0,
            "pin_locked_until": None
        }).eq("id", req.employee_id).execute()
        return {"success": True, "message": "PIN set successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.delete("/employees/{employee_id}/pin")
async def clear_employee_pin(employee_id: str):
    """Admin clears/resets PIN for an employee (forces re-set)."""
    try:
        supabase.table("employees").update({
            "punch_pin_hash": None,
            "pin_attempts": 0,
            "pin_locked_until": None
        }).eq("id", employee_id).execute()
        return {"success": True, "message": "PIN cleared"}
    except Exception as e:
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.get("/employees/{employee_id}/auth-status")
async def get_auth_status(employee_id: str):
    """Returns which auth methods are enrolled for this employee."""
    try:
        emp = supabase.table("employees").select("punch_pin_hash, pin_locked_until").eq("id", employee_id).limit(1).execute()
        face = supabase.table("face_embeddings").select("id").eq("employee_id", employee_id).limit(1).execute()
        finger = supabase.table("finger_templates").select("id").eq("employee_id", employee_id).limit(1).execute()
        return {
            "face_enrolled": len(face.data) > 0,
            "finger_enrolled": len(finger.data) > 0,
            "pin_set": bool(emp.data and emp.data[0].get("punch_pin_hash")),
            "pin_locked": bool(emp.data and emp.data[0].get("pin_locked_until"))
        }
    except Exception as e:
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.post("/location/check")
def check_location(req: LocationCheckRequest):
    dist = haversine(OFFICE_LAT, OFFICE_LON, req.latitude, req.longitude)
    return {"in_zone": dist <= FENCE_RADIUS_M, "distance_m": round(dist), "fence_radius_m": FENCE_RADIUS_M}

@app.post("/attendance/punch")
async def punch(req: PunchRequest):
    try:
        # Load live settings from DB (fallback to env)
        cfg = get_system_settings()
        office_lat    = cfg.get("geofence_lat")    or OFFICE_LAT
        office_lon    = cfg.get("geofence_lon")    or OFFICE_LON
        fence_radius  = cfg.get("geofence_radius") or FENCE_RADIUS_M
        face_threshold= cfg.get("face_threshold")  or FACE_THRESHOLD
        test_mode     = cfg.get("test_mode")       if cfg.get("test_mode") is not None else TEST_MODE
        strict_geo    = cfg.get("strict_geofence") if cfg.get("strict_geofence") is not None else True
        grace_minutes = cfg.get("grace_minutes")   or 15
        shift_start_str = cfg.get("shift_start") or "09:00"

        now   = datetime.now(timezone.utc)
        from datetime import timedelta
        ist_now = now + timedelta(hours=5, minutes=30)
        today = ist_now.strftime("%Y-%m-%d")  # IST date
        similarity = 0.0

        print(f"\n--- PUNCH {req.punch_type.upper()} ---")
        print(f"employee={req.employee_id} method={req.method} test_mode={test_mode}")
        print(f"Settings: lat={office_lat} lon={office_lon} radius={fence_radius}m threshold={face_threshold}")

        # Step 1: Geofence — also check employee WFH settings
        dist = haversine(office_lat, office_lon, req.latitude, req.longitude)
        print(f"Distance from office: {round(dist)}m (limit {fence_radius}m)")

        # Check employee-level WFH override
        geo_ok = dist <= fence_radius
        if not geo_ok:
            try:
                emp_geo = supabase.table("employees").select(
                    "allow_wfh, wfh_lat, wfh_lon, wfh_radius"
                ).eq("id", req.employee_id).limit(1).execute()
                if emp_geo.data and emp_geo.data[0].get("allow_wfh"):
                    ed = emp_geo.data[0]
                    if ed.get("wfh_lat") and ed.get("wfh_lon"):
                        wfh_dist = haversine(ed["wfh_lat"], ed["wfh_lon"], req.latitude, req.longitude)
                        wfh_radius = ed.get("wfh_radius") or 500
                        print(f"WFH check: {round(wfh_dist)}m from home (limit {wfh_radius}m)")
                        if wfh_dist <= wfh_radius:
                            geo_ok = True
                            dist = wfh_dist
                    else:
                        # WFH enabled but no specific location → allow from anywhere
                        geo_ok = True
            except Exception as wfh_err:
                print(f"WFH check error: {wfh_err}")

        if not geo_ok and strict_geo and not test_mode:
            raise HTTPException(403, {"code": "OUTSIDE_GEOFENCE", "message": f"You are {round(dist)}m from office (limit: {fence_radius}m).", "distance_m": round(dist)})
        if test_mode:
            dist = min(dist, 50)

        # Step 2: Auth — face, fingerprint, or PIN
        face_enabled_cfg        = cfg.get("face_enabled", True)
        fingerprint_enabled_cfg = cfg.get("fingerprint_enabled", False)
        pin_enabled_cfg         = cfg.get("pin_enabled", False)

        if req.method == "face":
            if not face_enabled_cfg:
                raise HTTPException(403, {"code": "METHOD_DISABLED", "message": "Face recognition is disabled by admin."})
            if not req.image_base64 or len(req.image_base64) < 1000:
                raise HTTPException(400, {"code": "BAD_IMAGE", "message": "Camera image missing or too small."})

            img = decode_image(req.image_base64)
            if img is None:
                raise HTTPException(400, {"code": "BAD_IMAGE", "message": "Could not decode webcam image."})
            print(f"Image decoded: {img.shape}")

            liveness_enabled = cfg.get("liveness_enabled") if cfg.get("liveness_enabled") is not None else True
            if not test_mode and liveness_enabled and not liveness_ok(img):
                raise HTTPException(422, {"code": "LIVENESS_FAILED", "message": "Liveness check failed. Use a live camera."})

            live_emb = get_embedding(img)
            if live_emb is None:
                raise HTTPException(422, {"code": "NO_FACE_DETECTED", "message": "No face detected. Look directly at camera."})
            print(f"Live embedding OK: {live_emb.shape}")

            stored = supabase.table("face_embeddings").select("embedding").eq("employee_id", req.employee_id).limit(1).execute()
            if not stored.data or len(stored.data) == 0:
                raise HTTPException(404, {"code": "NOT_ENROLLED", "message": "Face not enrolled. Please ask admin to enroll your face."})

            stored_emb = np.array(stored.data[0]["embedding"])
            similarity = cosine_sim(live_emb, stored_emb)
            print(f"Similarity: {similarity:.4f} (need >= {face_threshold})")

            if similarity < face_threshold:
                supabase.table("failed_attempts").insert({"employee_id": req.employee_id, "attempt_type": "face_mismatch", "similarity_score": round(similarity, 4), "latitude": req.latitude, "longitude": req.longitude, "attempted_at": now.isoformat()}).execute()
                raise HTTPException(401, {"code": "FACE_MISMATCH", "message": f"Face not recognised ({round(similarity*100,1)}% confidence). Try again.", "similarity": round(similarity, 4)})

        elif req.method == "thumb":
            if not fingerprint_enabled_cfg:
                raise HTTPException(403, {"code": "METHOD_DISABLED", "message": "Fingerprint auth is disabled by admin."})
            if not req.finger_template:
                raise HTTPException(400, {"code": "NO_TEMPLATE", "message": "No fingerprint template provided."})
            # Load stored template from DB
            stored_fp = supabase.table("finger_templates").select("template").eq("employee_id", req.employee_id).limit(1).execute()
            if not stored_fp.data:
                raise HTTPException(404, {"code": "FINGER_NOT_ENROLLED", "message": "Fingerprint not enrolled. Ask admin to enroll your fingerprint."})
            # NOTE: Real minutiae matching happens on the scanner SDK (client side).
            # The template sent here is already a match result flag from the SDK.
            # If you are doing server-side matching, compare templates here.
            # For now we trust the SDK result — the client sends template only on match.
            # Swap this logic for your SDK's server API if needed.
            print(f"Fingerprint template received, length={len(req.finger_template)}")
            # TODO: integrate server-side minutiae match if your scanner SDK supports it

        elif req.method == "pin":
            if not pin_enabled_cfg:
                raise HTTPException(403, {"code": "METHOD_DISABLED", "message": "PIN auth is disabled by admin."})
            if not req.punch_pin:
                raise HTTPException(400, {"code": "NO_PIN", "message": "PIN not provided."})

            emp_pin_data = supabase.table("employees").select(
                "punch_pin_hash, pin_attempts, pin_locked_until"
            ).eq("id", req.employee_id).limit(1).execute()

            if not emp_pin_data.data:
                raise HTTPException(404, {"code": "EMPLOYEE_NOT_FOUND", "message": "Employee not found."})

            emp = emp_pin_data.data[0]

            # Check lockout
            locked_until = emp.get("pin_locked_until")
            if locked_until:
                lu = datetime.fromisoformat(locked_until)
                if lu.tzinfo is None:
                    lu = lu.replace(tzinfo=timezone.utc)
                if now < lu:
                    mins_left = math.ceil((lu - now).total_seconds() / 60)
                    raise HTTPException(429, {"code": "PIN_LOCKED", "message": f"Too many wrong PINs. Try again in {mins_left} minute(s)."})

            if not emp.get("punch_pin_hash"):
                raise HTTPException(404, {"code": "PIN_NOT_SET", "message": "PIN not set. Ask admin to set your PIN."})

            pin_ok = bcrypt.checkpw(req.punch_pin.encode(), emp["punch_pin_hash"].encode())

            if not pin_ok:
                new_attempts = (emp.get("pin_attempts") or 0) + 1
                PIN_MAX = 3
                if new_attempts >= PIN_MAX:
                    from datetime import timedelta
                    lock_until = (now + timedelta(minutes=15)).isoformat()
                    supabase.table("employees").update({
                        "pin_attempts": new_attempts,
                        "pin_locked_until": lock_until
                    }).eq("id", req.employee_id).execute()
                    supabase.table("failed_attempts").insert({
                        "employee_id": req.employee_id,
                        "attempt_type": "pin_locked",
                        "latitude": req.latitude,
                        "longitude": req.longitude,
                        "attempted_at": now.isoformat()
                    }).execute()
                    raise HTTPException(429, {"code": "PIN_LOCKED", "message": f"Too many wrong PINs. Account locked for 15 minutes."})
                else:
                    supabase.table("employees").update({"pin_attempts": new_attempts}).eq("id", req.employee_id).execute()
                    supabase.table("failed_attempts").insert({
                        "employee_id": req.employee_id,
                        "attempt_type": "pin_mismatch",
                        "latitude": req.latitude,
                        "longitude": req.longitude,
                        "attempted_at": now.isoformat()
                    }).execute()
                    raise HTTPException(401, {"code": "PIN_WRONG", "message": f"Wrong PIN. {PIN_MAX - new_attempts} attempt(s) remaining."})

            # PIN correct — reset attempts
            supabase.table("employees").update({"pin_attempts": 0, "pin_locked_until": None}).eq("id", req.employee_id).execute()
            print("PIN auth OK")

        else:
            raise HTTPException(400, {"code": "INVALID_METHOD", "message": f"Unknown auth method: {req.method}. Use 'face', 'thumb', or 'pin'."})


        # Step 3: Log to Supabase
        existing = supabase.table("attendance_logs").select("id, punch_in_time, punch_out_time").eq("employee_id", req.employee_id).eq("date", today).execute()

        if req.punch_type == "in":
            if existing.data:
                raise HTTPException(409, {"code": "ALREADY_PUNCHED_IN", "message": "Already punched in today."})
            from datetime import timedelta
            ist_now = now + timedelta(hours=5, minutes=30)
            # Parse shift start from settings (e.g. "09:00")
            try:
                sh, sm = map(int, shift_start_str.split(":"))
            except Exception:
                sh, sm = 9, 0
            grace_end_h = sh + (sm + grace_minutes) // 60
            grace_end_m = (sm + grace_minutes) % 60
            is_late = (ist_now.hour > grace_end_h) or (ist_now.hour == grace_end_h and ist_now.minute > grace_end_m)
            ist_date = ist_now.strftime("%Y-%m-%d")  # use IST date not UTC date
            print(f"IST time: {ist_now.strftime('%H:%M')} shift_start={shift_start_str} grace={grace_minutes}min is_late={is_late}")
            supabase.table("attendance_logs").insert({"employee_id": req.employee_id, "date": ist_date, "punch_in_time": now.isoformat(), "punch_in_lat": req.latitude, "punch_in_lon": req.longitude, "punch_in_dist_m": round(dist), "method": req.method, "status": "late" if is_late else "present"}).execute()
            print("Punch IN saved!")
        else:
            if not existing.data:
                raise HTTPException(400, {"code": "NOT_PUNCHED_IN", "message": "No punch-in found for today."})
            if existing.data[0].get("punch_out_time"):
                raise HTTPException(409, {"code": "ALREADY_PUNCHED_OUT", "message": "Already punched out today."})
            # Parse punch_in_time and make timezone-aware if needed
            pin_str = existing.data[0]["punch_in_time"]
            pin_dt  = datetime.fromisoformat(pin_str)
            if pin_dt.tzinfo is None:
                pin_dt = pin_dt.replace(tzinfo=timezone.utc)
            diff_secs = (now - pin_dt).total_seconds()
            hours = round(diff_secs / 3600, 2)
            supabase.table("attendance_logs").update({"punch_out_time": now.isoformat(), "punch_out_lat": req.latitude, "punch_out_lon": req.longitude, "punch_out_dist_m": round(dist), "hours_worked": hours}).eq("id", existing.data[0]["id"]).execute()
            print(f"Punch OUT saved! hours={hours}")

        # Convert to IST for display (UTC + 5:30)
        from datetime import timedelta
        ist_time = now + timedelta(hours=5, minutes=30)
        return {"success": True, "punch_type": req.punch_type, "time": ist_time.strftime("%I:%M %p"), "distance_m": round(dist), "similarity": round(similarity, 4) if similarity else None, "message": f"Punch-{'In' if req.punch_type=='in' else 'Out'} successful at {ist_time.strftime('%I:%M %p')} IST"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"UNHANDLED ERROR:\n{traceback.format_exc()}")
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.get("/attendance/today")
def today_summary():
    try:
        from datetime import timedelta
        ist_today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
        print(f"today_summary: IST date = {ist_today}")

        # Step 1: fetch logs for today
        logs_resp = supabase.table("attendance_logs")             .select("id, date, employee_id, status, punch_in_time, punch_out_time, hours_worked, method, punch_in_dist_m, punch_out_dist_m")             .eq("date", ist_today).execute()
        logs = logs_resp.data or []
        print(f"today_summary: {len(logs)} logs found")

        # Step 2: fetch employees with department join
        emps_resp = supabase.table("employees").select("id, name, department_id, departments(name)").execute()
        emps = emps_resp.data or []
        emp_map = {e["id"]: e for e in emps}
        print(f"today_summary: {len(emps)} employees found")

        # Step 3: enrich logs with employee name + dept
        for l in logs:
            emp = emp_map.get(l.get("employee_id"), {})
            l["employee_name"] = emp.get("name", "Employee")
            # departments is a nested object from the join
            dept = emp.get("departments") or {}
            l["department"] = dept.get("name", "—") if isinstance(dept, dict) else "—"

        present = len([l for l in logs if l.get("status") in ("present","late")])
        late    = len([l for l in logs if l.get("status") == "late"])

        return {
            "date": ist_today,
            "total_employees": len(emps),
            "present": present,
            "absent": len(emps) - present,
            "late": late,
            "logs": logs
        }
    except Exception as e:
        import traceback
        print(f"today_summary ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})


@app.get("/attendance/debug")
def debug_today():
    """Simple debug endpoint to test each step"""
    from datetime import timedelta
    results = {}
    
    # Step 1: IST date
    ist_today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    results["ist_date"] = ist_today
    
    # Step 2: Try fetching logs
    try:
        logs = supabase.table("attendance_logs").select("id, date, employee_id, status, punch_in_time, punch_out_time").execute()
        results["all_logs_count"] = len(logs.data or [])
        results["all_logs_dates"] = list(set([l.get("date") for l in (logs.data or [])]))
        results["today_logs"] = [l for l in (logs.data or []) if l.get("date") == ist_today]
    except Exception as e:
        results["logs_error"] = str(e)

    # Step 3: Try fetching employees
    try:
        emps = supabase.table("employees").select("id, name, is_active").execute()
        results["employees_count"] = len(emps.data or [])
        results["employees"] = emps.data
    except Exception as e:
        results["employees_error"] = str(e)

    return results



# ── Employee CRUD ─────────────────────────────────────────────────────────────

class EmployeeRequest(BaseModel):
    emp_code: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = "employee"
    shift_start: Optional[str] = "09:00"
    shift_end: Optional[str] = "18:00"


@app.get("/employees/bycode/{emp_code}")
def get_employee_by_code(emp_code: str):
    try:
        print(f"Looking up employee code: {emp_code.upper()}")
        # Try exact match first
        res = supabase.table("employees") \
            .select("id, emp_code, name, department, role, shift_start, shift_end, is_active") \
            .eq("emp_code", emp_code.upper()) \
            .execute()
        print(f"Found {len(res.data or [])} employees")
        if not res.data:
            # Try case-insensitive search
            all_emps = supabase.table("employees").select("id, emp_code, name, department, role, shift_start, shift_end, is_active").execute()
            match = [e for e in (all_emps.data or []) if e.get("emp_code","").upper() == emp_code.upper()]
            if not match:
                raise HTTPException(404, {"code":"NOT_FOUND","message":f"Employee code '{emp_code}' not found"})
            return match[0]
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"bycode error: {e}")
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})

@app.get("/employees")
def get_employees():
    try:
        res = supabase.table("employees").select("*").order("name").execute()
        return res.data or []
    except Exception as e:
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})

@app.post("/employees")
def add_employee(req: EmployeeRequest):
    try:
        res = supabase.table("employees").insert({
            "emp_code":    req.emp_code,
            "name":        req.name,
            "email":       req.email if req.email else None,
            "phone":       req.phone,
            "department":  req.department,
            "role":        req.role,
            "shift_start": req.shift_start,
            "shift_end":   req.shift_end,
            "is_active":   True,
        }).execute()
        return {"success": True, "employee": res.data[0] if res.data else {}}
    except Exception as e:
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})

@app.put("/employees/{employee_id}")
def update_employee(employee_id: str, req: EmployeeRequest):
    try:
        res = supabase.table("employees").update({
            "emp_code":    req.emp_code,
            "name":        req.name,
            "email":       req.email if req.email else None,
            "phone":       req.phone,
            "department":  req.department,
            "role":        req.role,
            "shift_start": req.shift_start,
            "shift_end":   req.shift_end,
        }).eq("id", employee_id).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})

@app.delete("/employees/{employee_id}")
def delete_employee(employee_id: str):
    try:
        # Delete face embedding first
        supabase.table("face_embeddings").delete().eq("employee_id", employee_id).execute()
        # Delete attendance logs
        supabase.table("attendance_logs").delete().eq("employee_id", employee_id).execute()
        # Delete employee
        supabase.table("employees").delete().eq("id", employee_id).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})


@app.delete("/face/delete/{employee_id}")
def delete_face(employee_id: str):
    try:
        supabase.table("face_embeddings").delete().eq("employee_id", employee_id).execute()
        return {"success": True, "message": "Face enrollment removed"}
    except Exception as e:
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})

@app.get("/face/list")
def list_enrolled():
    try:
        res = supabase.table("face_embeddings").select("employee_id, registered_at").execute()
        return res.data or []
    except Exception as e:
        return []

@app.get("/departments")
def get_departments():
    try:
        res = supabase.table("departments").select("id, name").order("name").execute()
        return res.data or []
    except Exception as e:
        return []

@app.get("/attendance/date/{date}")
def attendance_by_date(date: str):
    try:
        logs = supabase.table("attendance_logs").select("*").eq("date", date).execute()
        logs_data = logs.data or []

        emp_ids = list(set([l["employee_id"] for l in logs_data if l.get("employee_id")]))
        emp_map = {}
        if emp_ids:
            emps = supabase.table("employees").select("id, name, department").execute()
            emp_map = {e["id"]: e for e in (emps.data or [])}

        for l in logs_data:
            emp = emp_map.get(l.get("employee_id"), {})
            l["employee_name"] = emp.get("name", "Unknown")
            l["department"]    = emp.get("department", "—")

        total   = supabase.table("employees").select("id", count="exact").execute()
        present = len([l for l in logs_data if l.get("status") in ("present","late")])
        late    = len([l for l in logs_data if l.get("status") == "late"])

        return {"date": date, "total_employees": total.count or 0, "present": present, "late": late, "logs": logs_data}
    except Exception as e:
        print(f"attendance_by_date error: {e}")
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})


def month_range(month_str: str):
    """Returns (from_date, to_date) for a month string like '2026-03'"""
    from calendar import monthrange
    year, mon = int(month_str.split('-')[0]), int(month_str.split('-')[1])
    last_day = monthrange(year, mon)[1]
    return f"{month_str}-01", f"{month_str}-{last_day:02d}"

@app.delete("/attendance/{log_id}")
def delete_attendance_log(log_id: str):
    try:
        check = supabase.table("attendance_logs").select("id").eq("id", log_id).execute()
        if not check.data:
            raise HTTPException(404, {"code": "NOT_FOUND", "message": "Attendance record not found"})
        supabase.table("attendance_logs").delete().eq("id", log_id).execute()
        print(f"Attendance log {log_id} deleted by admin")
        return {"success": True, "message": "Attendance record deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})


@app.get("/attendance/monthly")
def monthly_report(month: str):
    try:
        # month format: 2026-03
        from_date, to_date = month_range(month)
        logs = supabase.table("attendance_logs").select("*")             .gte("date", from_date).lte("date", to_date).execute()
        logs_data = logs.data or []

        emps = supabase.table("employees").select("id, name, department").execute()
        emp_map = {e["id"]: e for e in (emps.data or [])}

        # Group by employee
        from collections import defaultdict
        grouped = defaultdict(list)
        for l in logs_data:
            grouped[l["employee_id"]].append(l)

        result = []
        for emp_id, records in grouped.items():
            emp = emp_map.get(emp_id, {})
            present = len([r for r in records if r.get("status") == "present"])
            late    = len([r for r in records if r.get("status") == "late"])
            hours   = [r["hours_worked"] for r in records if r.get("hours_worked")]
            total_h = round(sum(hours), 1)
            avg_h   = round(total_h / len(hours), 1) if hours else 0
            work_days = 26
            absent  = max(0, work_days - present - late)
            result.append({
                "employee_id":  emp_id,
                "name":         emp.get("name", "Unknown"),
                "department":   emp.get("department", "—"),
                "present_days": present,
                "late_days":    late,
                "absent_days":  absent,
                "total_hours":  total_h,
                "avg_hours":    avg_h,
            })

        result.sort(key=lambda x: x["name"])
        return result
    except Exception as e:
        print(f"monthly_report error: {e}")
        raise HTTPException(500, {"code":"SERVER_ERROR","message":str(e)})



# ── Admin Manual Punch ────────────────────────────────────────────────────────
class AdminPunchRequest(BaseModel):
    employee_id: str
    punch_type: str            # "in" or "out"
    punch_time: Optional[str] = None   # ISO string, defaults to now
    reason: Optional[str] = "Admin override"
    admin_note: Optional[str] = None

@app.post("/attendance/admin-punch")
def admin_punch(req: AdminPunchRequest):
    try:
        from datetime import timedelta
        now     = datetime.now(timezone.utc)
        ist_now = now + timedelta(hours=5, minutes=30)
        today   = ist_now.strftime("%Y-%m-%d")

        # Use provided time or current time
        if req.punch_time:
            try:
                punch_dt = datetime.fromisoformat(req.punch_time)
                if punch_dt.tzinfo is None:
                    punch_dt = punch_dt.replace(tzinfo=timezone.utc)
            except:
                punch_dt = now
        else:
            punch_dt = now

        ist_punch = punch_dt + timedelta(hours=5, minutes=30)
        punch_date = ist_punch.strftime("%Y-%m-%d")

        print(f"Admin punch: {req.punch_type} for {req.employee_id} at {ist_punch.strftime('%H:%M')} IST")

        existing = supabase.table("attendance_logs") \
            .select("id, punch_in_time, punch_out_time") \
            .eq("employee_id", req.employee_id) \
            .eq("date", punch_date) \
            .execute()

        if req.punch_type == "in":
            if existing.data:
                # Update existing punch-in time
                supabase.table("attendance_logs").update({
                    "punch_in_time": punch_dt.isoformat(),
                    "method": "admin",
                    "notes": req.reason,
                    "status": "present",
                }).eq("id", existing.data[0]["id"]).execute()
            else:
                # Create new record
                supabase.table("attendance_logs").insert({
                    "employee_id":   req.employee_id,
                    "date":          punch_date,
                    "punch_in_time": punch_dt.isoformat(),
                    "punch_in_lat":  OFFICE_LAT,
                    "punch_in_lon":  OFFICE_LON,
                    "punch_in_dist_m": 0,
                    "method":        "admin",
                    "notes":         req.reason,
                    "status":        "present",
                }).execute()

        elif req.punch_type == "out":
            if not existing.data:
                raise HTTPException(400, {"code": "NOT_PUNCHED_IN",
                    "message": "No punch-in record found for this date. Please punch-in first."})

            # Punch-in exists — update punch-out
            rec = existing.data[0]
            pin_dt = datetime.fromisoformat(rec["punch_in_time"])
            if pin_dt.tzinfo is None:
                pin_dt = pin_dt.replace(tzinfo=timezone.utc)
            hours = round((punch_dt - pin_dt).total_seconds() / 3600, 2)

            supabase.table("attendance_logs").update({
                "punch_out_time":  punch_dt.isoformat(),
                "punch_out_lat":   OFFICE_LAT,
                "punch_out_lon":   OFFICE_LON,
                "punch_out_dist_m": 0,
                "hours_worked":    max(0, hours),
                "notes":           req.reason,
            }).eq("id", rec["id"]).execute()

        ist_time = ist_punch.strftime("%I:%M %p")
        return {
            "success": True,
            "punch_type": req.punch_type,
            "time": ist_time,
            "date": punch_date,
            "message": f"Admin punch-{'in' if req.punch_type=='in' else 'out'} recorded at {ist_time} IST"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})


# ── Employee WFH / Geofence Override ─────────────────────────────────────────
# ── System Settings ───────────────────────────────────────────────────────────

class SystemSettingsRequest(BaseModel):
    geofence_lat:        Optional[float] = None
    geofence_lon:        Optional[float] = None
    geofence_radius:     Optional[int]   = None
    geofence_location:   Optional[str]   = None
    strict_geofence:     Optional[bool]  = True
    allow_wfh_global:    Optional[bool]  = False
    shift_start:         Optional[str]   = "09:00"
    shift_end:           Optional[str]   = "18:00"
    grace_minutes:       Optional[int]   = 15
    halfday_hours:       Optional[float] = 4.0
    weekend_attendance:  Optional[bool]  = False
    face_enabled:        Optional[bool]  = True
    fingerprint_enabled: Optional[bool]  = False
    pin_enabled:         Optional[bool]  = False
    liveness_enabled:    Optional[bool]  = True
    test_mode:           Optional[bool]  = False
    face_threshold:      Optional[float] = 0.42

def get_system_settings() -> dict:
    """Load settings from DB; fall back to env vars if not set."""
    try:
        res = supabase.table("system_settings").select("*").eq("id", 1).limit(1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"get_system_settings error: {e}")
    # fallback to env
    return {
        "geofence_lat":    OFFICE_LAT,
        "geofence_lon":    OFFICE_LON,
        "geofence_radius": FENCE_RADIUS_M,
        "strict_geofence": True,
        "allow_wfh_global": False,
        "face_threshold":  FACE_THRESHOLD,
        "test_mode":       TEST_MODE,
        "shift_start":     "09:00",
        "shift_end":       "18:00",
        "grace_minutes":   15,
    }

@app.get("/settings")
def get_settings():
    return get_system_settings()

@app.post("/settings")
def save_settings(req: SystemSettingsRequest):
    try:
        data = req.dict(exclude_none=False)
        data["id"] = 1  # single-row settings
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("system_settings").upsert(data).execute()
        print(f"Settings saved: lat={req.geofence_lat} lon={req.geofence_lon} radius={req.geofence_radius} test_mode={req.test_mode}")
        return {"success": True, "message": "Settings saved successfully"}
    except Exception as e:
        print(f"save_settings error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})


class EmployeeGeoRequest(BaseModel):
    allow_wfh:     bool = False
    wfh_lat:       Optional[float] = None
    wfh_lon:       Optional[float] = None
    wfh_radius:    Optional[int]   = 500
    wfh_address:   Optional[str]   = None

@app.post("/employees/{employee_id}/geo-settings")
def set_employee_geo(employee_id: str, req: EmployeeGeoRequest):
    try:
        update_data = {
            "allow_wfh":   req.allow_wfh,
            "wfh_lat":     req.wfh_lat,
            "wfh_lon":     req.wfh_lon,
            "wfh_radius":  req.wfh_radius,
            "wfh_address": req.wfh_address,
        }
        supabase.table("employees").update(update_data).eq("id", employee_id).execute()
        return {"success": True, "message": "Geofence settings updated"}
    except Exception as e:
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.get("/employees/{employee_id}/geo-settings")
def get_employee_geo(employee_id: str):
    try:
        res = supabase.table("employees") \
            .select("id, name, allow_wfh, wfh_lat, wfh_lon, wfh_radius, wfh_address") \
            .eq("id", employee_id).execute()
        if not res.data:
            raise HTTPException(404, {"message": "Employee not found"})
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})

@app.get("/attendance/report/{employee_id}")
def employee_report(employee_id: str, month: Optional[str] = None):
    try:
        query = supabase.table("attendance_logs").select("*").eq("employee_id", employee_id)
        if month:
            from_date, to_date = month_range(month)
            query = query.gte("date", from_date).lte("date", to_date)
        data = query.order("date", desc=True).execute()
        return {"employee_id": employee_id, "records": data.data or []}
    except Exception as e:
        print(f"employee_report error: {e}")
        raise HTTPException(500, {"code": "SERVER_ERROR", "message": str(e)})