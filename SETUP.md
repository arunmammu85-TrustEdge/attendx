# AttendX — Complete Setup Guide
## 100% Free Stack · 100 Employees · ₹0/month

---

## Stack Overview

| Layer             | Tool                        | Cost   |
|-------------------|-----------------------------|--------|
| Face Recognition  | InsightFace ArcFace         | Free   |
| Fingerprint       | WebAuthn (browser built-in) | Free   |
| Geofencing        | Browser GPS + Haversine     | Free   |
| Database          | Supabase Free Tier          | Free   |
| Backend API       | FastAPI + Python            | Free   |
| Hosting (backend) | Render.com Free Tier        | Free   |
| Web Frontend      | HTML/JS (static)            | Free   |
| Mobile App        | React Native + Expo Go      | Free   |

**Total monthly cost: ₹0**

---

## Step 1 — Supabase Setup (5 mins)

1. Go to https://supabase.com → Create free account
2. New Project → name it `attendx`
3. Go to **SQL Editor** → paste contents of `supabase/schema.sql` → Run
4. Go to **Settings → API** → copy:
   - `Project URL` → this is your `SUPABASE_URL`
   - `service_role` secret key → this is your `SUPABASE_SERVICE_KEY`
5. Go to **Database → Replication** → enable realtime for `attendance_logs`

---

## Step 2 — Backend Setup (Local)

```bash
# Clone / create project folder
cd attendx/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# InsightFace will auto-download the ArcFace model (~300MB) on first run

# Configure environment
cp .env.example .env
# Edit .env with your Supabase URL, key, and office coordinates

# Run locally
uvicorn main:app --reload --port 8000

# Test it
curl http://localhost:8000/health
```

---

## Step 3 — Deploy Backend FREE on Render.com

1. Push your backend folder to a GitHub repo
2. Go to https://render.com → New → **Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Environment**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables (from your .env file)
6. Deploy → you get a free URL like `https://attendx-api.onrender.com`

> ⚠️ Render free tier sleeps after 15 mins of inactivity.
> Use https://cron-job.org to ping `/health` every 10 mins to keep it awake (free).

---

## Step 4 — Connect Frontend to Backend

In your web app (attendance-system.html), update the API base URL:

```javascript
const API_BASE = 'https://attendx-api.onrender.com';  // your Render URL

// Example punch call
async function doPunch(type) {
  const snapshot = captureWebcam();   // base64 from <video> element
  const location = await getGPS();

  const response = await fetch(`${API_BASE}/attendance/punch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      employee_id: currentEmployee.id,
      image_base64: snapshot,
      latitude: location.lat,
      longitude: location.lon,
      punch_type: type,       // 'in' or 'out'
      method: currentMethod,  // 'face', 'thumb', 'dual'
    })
  });

  const result = await response.json();
  if (result.success) showToast('✅', result.message);
  else showToast('🚫', result.message);
}
```

---

## Step 5 — Enroll Employee Faces (Onboarding)

For each employee, call the register endpoint once:

```bash
curl -X POST https://attendx-api.onrender.com/face/register \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "uuid-of-employee",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

Or build a simple admin onboarding page with a webcam capture button.

---

## Step 6 — Mobile App (React Native)

```bash
# Install Expo CLI
npm install -g expo-cli

# Create app
npx create-expo-app AttendX
cd AttendX

# Install required packages
npx expo install expo-camera expo-location expo-local-authentication

# Run on your phone
npx expo start
# Scan QR code with Expo Go app (iOS/Android)
```

Key mobile packages:
- `expo-camera` → webcam capture for face scan
- `expo-location` → GPS coordinates
- `expo-local-authentication` → device fingerprint/Face ID

---

## Accuracy Reference

| Model          | Accuracy  | Speed    | RAM   |
|----------------|-----------|----------|-------|
| face-api.js    | 92–95%    | 300ms    | 100MB |
| DeepFace       | 96–98%    | 400ms    | 200MB |
| **InsightFace ArcFace** | **99%+** | **150ms** | **512MB** |
| AWS Rekognition| 99.4%     | 100ms    | Cloud |
| Azure Face API | 99.2%     | 120ms    | Cloud |

InsightFace matches paid cloud services in accuracy. ✅

---

## Scaling Guide

| Employees | Supabase Plan | Render Plan  | Monthly Cost |
|-----------|---------------|--------------|--------------|
| < 100     | Free          | Free         | ₹0           |
| 100–300   | Free          | Free/Starter | ₹0–₹800      |
| 300–500   | Pro ($25)     | Starter $7   | ~₹2,700      |
| 500–1000  | Pro ($25)     | Standard $25 | ~₹4,200      |
| 1000+     | Pro ($25)     | Custom       | ~₹6,000+     |

---

## Security Checklist

- [x] Face embeddings stored encrypted in Supabase (never sent to 3rd party)
- [x] RLS policies — employees can only see their own records
- [x] Liveness detection (anti-spoofing)
- [x] Geofence enforced server-side (can't be bypassed from frontend)
- [x] Failed attempt logging for audit trail
- [x] HTTPS everywhere (Render + Supabase both enforce TLS)
- [ ] Add rate limiting (max 5 punch attempts per minute per employee)
- [ ] Add admin 2FA for dashboard access
- [ ] Periodic face re-enrollment reminder (every 6 months)
