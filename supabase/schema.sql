-- ============================================================
-- AttendX — Supabase Schema
-- Run this in Supabase SQL Editor (free tier)
-- ============================================================

-- ── 1. Departments ────────────────────────────────────────
CREATE TABLE departments (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL UNIQUE,
  created_at  TIMESTAMPTZ DEFAULT now()
);

INSERT INTO departments (name) VALUES
  ('Engineering'), ('HR'), ('Finance'), ('Marketing'), ('Operations');


-- ── 2. Employees ──────────────────────────────────────────
CREATE TABLE employees (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  emp_code      TEXT NOT NULL UNIQUE,          -- e.g. EMP001
  name          TEXT NOT NULL,
  email         TEXT NOT NULL UNIQUE,
  phone         TEXT,
  department_id UUID REFERENCES departments(id),
  role          TEXT DEFAULT 'employee',        -- 'employee' | 'manager' | 'admin'
  shift_start   TIME DEFAULT '09:00',
  shift_end     TIME DEFAULT '18:00',
  grace_minutes INT  DEFAULT 15,               -- late threshold in minutes
  is_active     BOOLEAN DEFAULT true,
  joined_at     DATE DEFAULT CURRENT_DATE,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- Index for fast lookups
CREATE INDEX idx_employees_emp_code ON employees(emp_code);
CREATE INDEX idx_employees_department ON employees(department_id);


-- ── 3. Face Embeddings ────────────────────────────────────
-- Stores the 512-dim ArcFace embedding per employee
-- Data never leaves your Supabase — fully private
CREATE TABLE face_embeddings (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id   UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
  embedding     FLOAT8[] NOT NULL,             -- 512-element ArcFace vector
  registered_at TIMESTAMPTZ DEFAULT now(),
  registered_by TEXT,                          -- admin who enrolled this face

  CONSTRAINT uq_face_per_employee UNIQUE (employee_id)
);

-- RLS: Only backend service role can read/write embeddings (never expose to frontend)
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role only" ON face_embeddings
  USING (auth.role() = 'service_role');


-- ── 4. Attendance Logs ────────────────────────────────────
CREATE TABLE attendance_logs (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id       UUID NOT NULL REFERENCES employees(id),
  date              DATE NOT NULL DEFAULT CURRENT_DATE,

  -- Punch In
  punch_in_time     TIMESTAMPTZ,
  punch_in_lat      FLOAT8,
  punch_in_lon      FLOAT8,
  punch_in_dist_m   INT,                       -- metres from office at punch-in

  -- Punch Out
  punch_out_time    TIMESTAMPTZ,
  punch_out_lat     FLOAT8,
  punch_out_lon     FLOAT8,
  punch_out_dist_m  INT,                       -- metres from office at punch-out

  -- Summary
  hours_worked      FLOAT4,                    -- calculated on punch-out
  method            TEXT DEFAULT 'face',       -- 'face' | 'thumb' | 'dual'
  status            TEXT DEFAULT 'present',    -- 'present' | 'late' | 'absent' | 'half_day' | 'wfh'
  notes             TEXT,
  created_at        TIMESTAMPTZ DEFAULT now(),

  CONSTRAINT uq_attendance_per_day UNIQUE (employee_id, date)
);

CREATE INDEX idx_attendance_date        ON attendance_logs(date);
CREATE INDEX idx_attendance_employee    ON attendance_logs(employee_id);
CREATE INDEX idx_attendance_status      ON attendance_logs(status);


-- ── 5. Failed Attempts Log ────────────────────────────────
-- Tracks failed punch attempts for security auditing
CREATE TABLE failed_attempts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id     UUID REFERENCES employees(id),
  attempt_type    TEXT,    -- 'face_mismatch' | 'outside_geofence' | 'liveness_failed'
  similarity_score FLOAT4,
  latitude        FLOAT8,
  longitude       FLOAT8,
  attempted_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_failed_attempts_employee ON failed_attempts(employee_id);
CREATE INDEX idx_failed_attempts_type     ON failed_attempts(attempt_type);


-- ── 6. Leave Requests ─────────────────────────────────────
CREATE TABLE leave_requests (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id   UUID NOT NULL REFERENCES employees(id),
  leave_type    TEXT NOT NULL,   -- 'sick' | 'casual' | 'earned' | 'wfh'
  from_date     DATE NOT NULL,
  to_date       DATE NOT NULL,
  reason        TEXT,
  status        TEXT DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
  approved_by   UUID REFERENCES employees(id),
  applied_at    TIMESTAMPTZ DEFAULT now()
);


-- ── 7. Shift Overrides ────────────────────────────────────
-- For employees with different shift timings on specific days
CREATE TABLE shift_overrides (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id   UUID NOT NULL REFERENCES employees(id),
  override_date DATE NOT NULL,
  shift_start   TIME,
  shift_end     TIME,
  reason        TEXT,
  created_at    TIMESTAMPTZ DEFAULT now(),

  CONSTRAINT uq_shift_override UNIQUE (employee_id, override_date)
);


-- ── 8. Useful Views ───────────────────────────────────────

-- Today's attendance summary view
CREATE VIEW today_attendance AS
SELECT
  e.emp_code,
  e.name,
  d.name          AS department,
  a.punch_in_time,
  a.punch_out_time,
  a.hours_worked,
  a.method,
  a.status,
  a.punch_in_dist_m,
  a.punch_out_dist_m
FROM employees e
LEFT JOIN attendance_logs a ON a.employee_id = e.id AND a.date = CURRENT_DATE
LEFT JOIN departments d ON d.id = e.department_id
WHERE e.is_active = true
ORDER BY a.punch_in_time ASC NULLS LAST;


-- Monthly summary per employee
CREATE VIEW monthly_summary AS
SELECT
  e.emp_code,
  e.name,
  d.name                                            AS department,
  DATE_TRUNC('month', a.date)                       AS month,
  COUNT(*) FILTER (WHERE a.status = 'present')      AS present_days,
  COUNT(*) FILTER (WHERE a.status = 'late')         AS late_days,
  COUNT(*) FILTER (WHERE a.status = 'absent')       AS absent_days,
  COUNT(*) FILTER (WHERE a.status = 'wfh')          AS wfh_days,
  ROUND(AVG(a.hours_worked)::NUMERIC, 2)            AS avg_hours_per_day,
  ROUND(SUM(a.hours_worked)::NUMERIC, 2)            AS total_hours
FROM employees e
LEFT JOIN attendance_logs a ON a.employee_id = e.id
LEFT JOIN departments d ON d.id = e.department_id
WHERE e.is_active = true
GROUP BY e.emp_code, e.name, d.name, DATE_TRUNC('month', a.date)
ORDER BY month DESC, e.name;


-- ── 9. Row Level Security (RLS) ───────────────────────────
-- Employees can only read their own records
ALTER TABLE attendance_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

-- Admins/managers see everything, employees see only their own
CREATE POLICY "Employees see own records" ON attendance_logs
  FOR SELECT USING (
    employee_id = (
      SELECT id FROM employees WHERE email = auth.jwt()->>'email'
    )
    OR EXISTS (
      SELECT 1 FROM employees
      WHERE email = auth.jwt()->>'email'
      AND role IN ('admin', 'manager')
    )
  );

-- ── 10. Realtime (enable for live dashboard) ──────────────
-- Run in Supabase dashboard: Database > Replication
-- Enable realtime for: attendance_logs, failed_attempts

-- ── Sample Data (optional — remove in production) ─────────
INSERT INTO employees (emp_code, name, email, phone, department_id, role) VALUES
  ('EMP001', 'Priya Sharma',  'priya@company.com',  '9876543210', (SELECT id FROM departments WHERE name='Engineering'), 'employee'),
  ('EMP002', 'Arjun Mehta',   'arjun@company.com',  '9876543211', (SELECT id FROM departments WHERE name='Finance'),     'employee'),
  ('EMP003', 'Ravi Kumar',    'ravi@company.com',   '9876543212', (SELECT id FROM departments WHERE name='Engineering'), 'employee'),
  ('EMP004', 'Meera Nair',    'meera@company.com',  '9876543213', (SELECT id FROM departments WHERE name='Marketing'),   'employee'),
  ('EMP005', 'Sanjay Patel',  'sanjay@company.com', '9876543214', (SELECT id FROM departments WHERE name='HR'),          'manager'),
  ('ADM001', 'Admin User',    'admin@company.com',  '9876543200', (SELECT id FROM departments WHERE name='HR'),          'admin');
