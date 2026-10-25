"""
FraudShield — Flask Backend
Connects frontend to ML model + SQLite database
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import sqlite3, hashlib, os, json, time
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = "fraudshield_secret_2026"
CORS(app)

DB_PATH = "fraudshield.db"
MODEL_PATH = "model/fraud_model.pkl"

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            home_location TEXT DEFAULT 'Bengaluru,IN',
            avg_monthly_spend REAL DEFAULT 800,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            merchant TEXT,
            amount REAL,
            location TEXT,
            time_of_day TEXT,
            v14 REAL DEFAULT 0,
            v17 REAL DEFAULT 0,
            v1  REAL DEFAULT 0,
            v3  REAL DEFAULT 0,
            risk_score REAL,
            status TEXT,         -- 'fraud', 'warning', 'safe'
            triggers TEXT,       -- JSON array of trigger names
            distance_km REAL,
            predicted_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            transaction_id INTEGER,
            level TEXT,          -- 'critical', 'warning'
            message TEXT,
            triggers TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            dismissed INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# City coordinates for geo-distance calculation
CITY_COORDS = {
    "Bengaluru,IN":   (12.97, 77.59),
    "Mumbai,IN":      (19.08, 72.88),
    "Delhi,IN":       (28.61, 77.21),
    "London,UK":      (51.51, -0.13),
    "New York,US":    (40.71, -74.01),
    "Dubai,UAE":      (25.20, 55.27),
    "Tokyo,JP":       (35.68, 139.69),
    "Sydney,AU":      (-33.87, 151.21),
    "Moscow,RU":      (55.75, 37.62),
    "Lagos,NG":       (6.52, 3.38),
}

HIGH_RISK_CITIES = {"Moscow,RU", "Lagos,NG"}

def haversine(c1, c2):
    """Distance in km between two (lat,lon) pairs."""
    R = 6371
    lat1, lon1 = np.radians(c1)
    lat2, lon2 = np.radians(c2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_model():
    """Load the trained sklearn model if available."""
    if os.path.exists(MODEL_PATH):
        import joblib
        return joblib.load(MODEL_PATH)
    return None


def score_transaction(data, user_home, thresholds):
    """
    Score a transaction. Returns dict with risk_score, status, triggers.
    Falls back to rule-based if ML model not available.
    """
    amount   = float(data.get("amount", 0))
    location = data.get("location", "")
    v14      = float(data.get("v14", 0))
    v17      = float(data.get("v17", 0))
    v1       = float(data.get("v1", 0))
    v3       = float(data.get("v3", 0))

    # Thresholds (use user settings or defaults)
    th_warn      = thresholds.get("amount_warn", 500)
    th_alert     = thresholds.get("amount_alert", 1000)
    th_svm_warn  = thresholds.get("svm_warn", 40)
    th_svm_alert = thresholds.get("svm_alert", 70)
    th_geo_warn  = thresholds.get("geo_warn", 500)
    th_geo_alert = thresholds.get("geo_alert", 2000)

    triggers = []
    risk_score = 0.0

    # ── 1. Geo distance ──────────────────────────────
    home_coords = CITY_COORDS.get(user_home, (12.97, 77.59))
    txn_coords  = CITY_COORDS.get(location, home_coords)
    distance_km = haversine(home_coords, txn_coords)

    if location in HIGH_RISK_CITIES:
        risk_score += 40
        triggers.append("geo")
    elif distance_km > th_geo_alert:
        risk_score += 30
        triggers.append("geo")
    elif distance_km > th_geo_warn:
        risk_score += 15
        triggers.append("geo")

    # ── 2. Amount ────────────────────────────────────
    if amount > th_alert:
        risk_score += 30
        triggers.append("amt")
    elif amount > th_warn:
        risk_score += 15
        triggers.append("amt")

    # ── 3. Behavioural (V-features / SVM proxy) ──────
    model = load_model()
    if model:
        try:
            features = np.array([[v14, v17, v1, v3, amount]])
            prob = model.predict_proba(features)[0][1] * 100
        except Exception:
            prob = _rule_svm_score(v14, v17, v1, v3)
    else:
        prob = _rule_svm_score(v14, v17, v1, v3)

    if prob > th_svm_alert:
        risk_score += 40
        triggers.append("svm")
    elif prob > th_svm_warn:
        risk_score += 20
        triggers.append("svm")

    risk_score = min(risk_score, 100)

    # ── 4. Classify ──────────────────────────────────
    if risk_score >= 60 or "svm" in triggers and prob > th_svm_alert:
        status = "fraud"
    elif risk_score >= 25:
        status = "warning"
    else:
        status = "safe"

    return {
        "risk_score":  round(risk_score, 1),
        "svm_prob":    round(prob, 1),
        "distance_km": round(distance_km, 1),
        "status":      status,
        "triggers":    triggers,
    }


def _rule_svm_score(v14, v17, v1, v3):
    """Rule-based behavioural score (0–100) when model unavailable."""
    score = 0
    if v14 <= -5:  score += 40
    elif v14 <= -2: score += 20
    if v17 <= -5:  score += 40
    elif v17 <= -2: score += 20
    if v1  <= -3:  score += 10
    if v3  <= -3:  score += 10
    return min(score, 100)


# ─────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    d = request.json
    username = d.get("username", "").strip()
    email    = d.get("email", "").strip()
    password = d.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (username, email, hash_password(password))
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        session["user_id"]  = user["id"]
        session["username"] = user["username"]
        return jsonify({"success": True, "username": username})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 409
    finally:
        conn.close()


@app.route("/api/login", methods=["POST"])
def login():
    d = request.json
    username = d.get("username", "").strip()
    password = d.get("password", "")

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    return jsonify({
        "success":       True,
        "username":      user["username"],
        "home_location": user["home_location"],
        "avg_spend":     user["avg_monthly_spend"],
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/me")
def me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not logged in"}), 401
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id":            user["id"],
        "username":      user["username"],
        "email":         user["email"],
        "home_location": user["home_location"],
        "avg_spend":     user["avg_monthly_spend"],
    })


@app.route("/api/profile", methods=["PUT"])
def update_profile():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not logged in"}), 401
    d = request.json
    conn = get_db()
    conn.execute(
        "UPDATE users SET home_location=?, avg_monthly_spend=? WHERE id=?",
        (d.get("home_location"), d.get("avg_spend"), uid)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ─────────────────────────────────────────
# TRANSACTION / FRAUD DETECTION ROUTES
# ─────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze a single transaction and store result."""
    uid = session.get("user_id")
    d   = request.json

    # Get user profile for home location
    home_location = "Bengaluru,IN"
    if uid:
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        if user:
            home_location = user["home_location"]

    thresholds = d.get("thresholds", {})
    result = score_transaction(d, home_location, thresholds)

    # Persist to DB
    if uid:
        conn = get_db()
        cur = conn.execute(
            """INSERT INTO transactions
               (user_id, merchant, amount, location, time_of_day,
                v14, v17, v1, v3, risk_score, status, triggers, distance_km)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid,
                d.get("merchant", "Unknown"),
                float(d.get("amount", 0)),
                d.get("location", ""),
                d.get("time_of_day", ""),
                float(d.get("v14", 0)),
                float(d.get("v17", 0)),
                float(d.get("v1", 0)),
                float(d.get("v3", 0)),
                result["risk_score"],
                result["status"],
                json.dumps(result["triggers"]),
                result["distance_km"],
            )
        )
        txn_id = cur.lastrowid

        # Log alert if needed
        if result["status"] in ("fraud", "warning"):
            level   = "critical" if result["status"] == "fraud" else "warning"
            reasons = ", ".join(t.upper() for t in result["triggers"]) or "GENERAL"
            msg = (f"{d.get('merchant','Unknown')} · ${float(d.get('amount',0)):.2f} "
                   f"· {d.get('location','')} — triggers: {reasons}")
            conn.execute(
                "INSERT INTO alerts (user_id, transaction_id, level, message, triggers) VALUES (?,?,?,?,?)",
                (uid, txn_id, level, msg, json.dumps(result["triggers"]))
            )

        conn.commit()
        conn.close()

    return jsonify({**result, "merchant": d.get("merchant"), "amount": d.get("amount")})


@app.route("/api/transactions")
def get_transactions():
    uid = session.get("user_id")
    if not uid:
        return jsonify([])
    limit = request.args.get("limit", 50, type=int)
    conn  = get_db()
    rows  = conn.execute(
        "SELECT * FROM transactions WHERE user_id=? ORDER BY predicted_at DESC LIMIT ?",
        (uid, limit)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts")
def get_alerts():
    uid = session.get("user_id")
    if not uid:
        return jsonify([])
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC LIMIT 100",
        (uid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts/<int:alert_id>/dismiss", methods=["POST"])
def dismiss_alert(alert_id):
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not logged in"}), 401
    conn = get_db()
    conn.execute("UPDATE alerts SET dismissed=1 WHERE id=? AND user_id=?", (alert_id, uid))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/stats")
def get_stats():
    """Summary stats for the dashboard counters."""
    uid = session.get("user_id")
    if not uid:
        return jsonify({"total": 0, "fraud": 0, "warning": 0, "safe": 0})
    conn = get_db()
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='fraud'   THEN 1 ELSE 0 END) as fraud,
            SUM(CASE WHEN status='warning' THEN 1 ELSE 0 END) as warning,
            SUM(CASE WHEN status='safe'    THEN 1 ELSE 0 END) as safe
        FROM transactions WHERE user_id=?
    """, (uid,)).fetchone()
    conn.close()
    return jsonify(dict(row))


# ─────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html") if os.path.exists("static/index.html") else "FraudShield API running"

@app.route("/dashboard")
def dashboard():
    return app.send_static_file("app.html") if os.path.exists("static/app.html") else "Dashboard"


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    init_db()
    print("✅ FraudShield backend running → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
