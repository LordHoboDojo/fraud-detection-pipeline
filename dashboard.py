"""
Fraud Detection Dashboard
Flask server running on EMR master — serves the UI and proxies data from Redshift.

Run: python3 dashboard.py
Visit: http://<emr-public-ip>:5000
"""

import subprocess, threading, time, json
from datetime import datetime, timezone
from flask import Flask, jsonify
import redshift_connector

app = Flask(__name__, static_folder="static")

REDSHIFT = dict(
    host="fraud-pipeline-redshift.cj6sggo2xcoh.us-east-2.redshift.amazonaws.com",
    port=5439, database="frauddb", user="fraudadmin", password="FraudPipeline2024!",
)
PRODUCER_CMD = [
    "python3", "/home/hadoop/kafka_producer.py",
    "--rate", "50", "--topic", "fraud-transactions", "--limit", "100",
    "--ensure-fraud", "15",
]

# ── simulation state ──────────────────────────────────────────────────────────
_sim_lock   = threading.Lock()
_sim_status = {"running": False, "sent": 0, "fraud_sent": 0, "error": None,
               "start_ts": None, "end_ts": None, "completed": False}


def _redshift():
    return redshift_connector.connect(**REDSHIFT, timeout=10)


def _row_to_dict(cols, r):
    rec = dict(zip(cols, r))
    rec["transaction_timestamp"] = str(rec.get("transaction_timestamp", ""))
    rec["processed_at"]          = str(rec.get("processed_at", ""))
    rec["transaction_amt"]       = float(rec.get("transaction_amt", 0))
    rec["fraud_probability"]     = round(float(rec.get("fraud_probability", 0)), 4)
    rec["transaction_id"]        = int(rec.get("transaction_id", 0))
    rec["is_fraud_actual"]       = int(rec.get("is_fraud_actual", 0))
    return rec


# ── API: overall stats ────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    try:
        conn = _redshift()
        cur  = conn.cursor()

        cur.execute("""
            SELECT
                COUNT(*)                                                AS total,
                SUM(CASE WHEN fraud_alert='FRAUD' THEN 1 ELSE 0 END)   AS fraud_total,
                AVG(transaction_amt)                                    AS avg_amt,
                AVG(fraud_probability)                                  AS avg_prob
            FROM fraud_predictions
        """)
        row = cur.fetchone()
        total, fraud_total, avg_amt, avg_prob = (
            int(row[0] or 0), int(row[1] or 0),
            float(row[2] or 0), float(row[3] or 0),
        )

        cur.execute("""
            SELECT transaction_id, transaction_timestamp, transaction_amt,
                   fraud_probability, fraud_alert, is_fraud_actual, processed_at
            FROM fraud_predictions
            ORDER BY processed_at DESC, transaction_id DESC
            LIMIT 50
        """)
        cols   = [d[0] for d in cur.description]
        recent = [_row_to_dict(cols, r) for r in cur.fetchall()]

        cur.execute("""
            SELECT
                FLOOR(DATEDIFF(second, '2026-01-01', processed_at) / 30) AS bucket,
                COUNT(*) AS cnt,
                SUM(CASE WHEN fraud_alert='FRAUD' THEN 1 ELSE 0 END)     AS frauds
            FROM fraud_predictions
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT 20
        """)
        timeline = [
            {"bucket": int(r[0]), "total": int(r[1]), "fraud": int(r[2])}
            for r in cur.fetchall()
        ]

        conn.close()
        return jsonify({
            "total": total,
            "fraud_total": fraud_total,
            "fraud_rate": round(fraud_total / total * 100, 2) if total else 0,
            "avg_amt": round(avg_amt, 2),
            "avg_prob": round(avg_prob, 4),
            "recent": recent,
            "timeline": list(reversed(timeline)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: trigger simulation ───────────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def simulate():
    with _sim_lock:
        if _sim_status["running"]:
            return jsonify({"error": "Simulation already running"}), 409
        _sim_status.update(running=True, sent=0, fraud_sent=0, error=None,
                           start_ts=datetime.now(timezone.utc).isoformat(),
                           end_ts=None, completed=False)

    def _run():
        try:
            proc = subprocess.Popen(
                PRODUCER_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("Sent"):
                    parts = line.replace(",", "").split()
                    with _sim_lock:
                        _sim_status["sent"]       = int(parts[1].split("/")[0])
                        _sim_status["fraud_sent"] = int(parts[-1].rstrip(")"))
                elif line.startswith("Done"):
                    # "Done. Sent 100 transactions (15 fraud) to 'fraud-transactions'"
                    parts = line.replace(",", "").split()
                    with _sim_lock:
                        _sim_status["sent"]       = int(parts[2])
                        _sim_status["fraud_sent"] = int(parts[4].lstrip("("))
            proc.wait()
            with _sim_lock:
                _sim_status["end_ts"]    = datetime.now(timezone.utc).isoformat()
                _sim_status["completed"] = True
        except Exception as e:
            with _sim_lock:
                _sim_status["error"] = str(e)
        finally:
            with _sim_lock:
                _sim_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/simulate/status")
def sim_status():
    with _sim_lock:
        return jsonify(dict(_sim_status))


# ── API: results of the last simulation batch ─────────────────────────────────

@app.route("/api/simulate/results")
def sim_results():
    with _sim_lock:
        start_ts  = _sim_status.get("start_ts")
        end_ts    = _sim_status.get("end_ts")
        completed = _sim_status.get("completed", False)

    if not start_ts:
        return jsonify({"error": "No simulation has been run yet"}), 404
    if not completed:
        return jsonify({"error": "Simulation still running"}), 202

    try:
        conn = _redshift()
        cur  = conn.cursor()

        # Give Spark up to a few seconds of lag — fetch rows processed after the
        # simulation started (Spark may still be writing for a few seconds after
        # Kafka producer finishes, so we query with a generous window).
        cur.execute("""
            SELECT transaction_id, transaction_timestamp, transaction_amt,
                   fraud_probability, fraud_alert, is_fraud_actual, processed_at
            FROM fraud_predictions
            WHERE processed_at >= CONVERT_TIMEZONE('UTC', %s::timestamp)
            ORDER BY processed_at ASC, transaction_id ASC
            LIMIT 200
        """, (start_ts,))
        cols = [d[0] for d in cur.description]
        rows = [_row_to_dict(cols, r) for r in cur.fetchall()]

        # Summary metrics for this batch
        total   = len(rows)
        frauds  = sum(1 for r in rows if r["fraud_alert"] == "FRAUD")
        legits  = total - frauds
        tp = sum(1 for r in rows if r["fraud_alert"] == "FRAUD" and r["is_fraud_actual"] == 1)
        fp = sum(1 for r in rows if r["fraud_alert"] == "FRAUD" and r["is_fraud_actual"] == 0)
        fn = sum(1 for r in rows if r["fraud_alert"] == "LEGIT" and r["is_fraud_actual"] == 1)
        tn = sum(1 for r in rows if r["fraud_alert"] == "LEGIT" and r["is_fraud_actual"] == 0)
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
        recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0

        conn.close()
        return jsonify({
            "total": total,
            "fraud_alerts": frauds,
            "legit": legits,
            "fraud_rate": round(frauds / total * 100, 1) if total else 0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Serve pages ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/results")
def results():
    return app.send_static_file("results.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
