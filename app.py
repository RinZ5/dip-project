"""Civet Behavior Detector: YOLO finds the civet, tray and food each second; rules label the behavior.
Run: python app.py, then open http://127.0.0.1:5001
"""
import csv, datetime as dt, json, re, subprocess, threading, uuid
from pathlib import Path

import cv2, imageio_ffmpeg, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, abort, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# --- settings
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/civet_yolo.pt"
CIVET, TRAY, FOOD = 0, 1, 2
CIVET_CONF, TRAY_CONF, FOOD_CONF = 0.25, 0.25, 0.35
if MODEL.with_suffix(".json").exists():        # thresholds saved with the model
    cfg = json.loads(MODEL.with_suffix(".json").read_text())
    CIVET_CONF, TRAY_CONF, FOOD_CONF = cfg["civet_conf"], cfg["tray_conf"], cfg["food_conf"]
W, H = 640, 360
MOVE_T = 0.08              # body lengths per second
TRAY_PAD = 0.25            # margin around the tray
MIN_REST_S = 10            # shortest resting run, in seconds
EAT_BRIDGE_S = 2           # short moves inside a meal stay eating
FOOD_GONE_S = 10           # food must be unseen this long to count as gone
BOUT_GAP_S = 10            # short gaps don't split a bout
VIDEO_SPEED = 4            # tracked video playback speed

BEHAVIORS = ["moving", "resting", "eating", "unknown"]
COLORS = {"moving": "#C47F2A", "resting": "#3F7A5C", "eating": "#6F5A94", "unknown": "#8E9893"}
ACTIVITY = {"moving": "active", "eating": "active", "resting": "inactive"}


# --- video
# read one frame per second
def frames(video):
    """One frame per second: (second, frame)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(video), "-an",
           "-vf", f"fps=1,scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    t = 0
    while len(buf := p.stdout.read(W * H * 3)) == W * H * 3:
        yield t, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        t += 1
    p.wait()


# recording start time, from the file name
def video_start(name):
    """Start time from the file name, e.g. A03_20260405000000 or 2569... (Thai year)."""
    name = Path(name).stem
    if m := re.search(r"(20\d{12})", name):
        return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if m := re.search(r"(25\d{2})(\d{10})", name):
        return dt.datetime.strptime(f"{int(m.group(1)) - 543}{m.group(2)}", "%Y%m%d%H%M%S")
    return None


# --- detection and behavior rules
# most confident box of one class
def best_box(boxes, cls, conf):
    b = boxes[(boxes.cls == cls) & (boxes.conf >= conf)]
    return b.xyxy[b.conf.argmax()].tolist() if len(b) else None


# do two boxes overlap?
def touches(a, b, pad=0.0):
    pw, ph = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    return a[0] < b[2] + pw and a[2] > b[0] - pw and a[1] < b[3] + ph and a[3] > b[1] - ph


def body_movement(box, prev):
    """Box centre movement in body lengths."""
    diag = np.hypot(box[2] - box[0], box[3] - box[1])
    return float(np.hypot((box[0] + box[2] - prev[0] - prev[2]) / 2, (box[1] + box[3] - prev[1] - prev[3]) / 2) / diag)


# the behavior rules
def behavior(movement, at_tray, tray_has_food):
    if movement > MOVE_T:
        return "moving"
    return "eating" if at_tray and tray_has_food else "resting"


# label every second of a video
def analyse(video, model, progress=None):
    rows, prev = [], None
    tray, food, no_food_s = None, False, 0   # remembered while the civet covers the tray
    for t, f in frames(video):
        b = model(f, conf=min(CIVET_CONF, TRAY_CONF, FOOD_CONF), verbose=False)[0].boxes
        civet = best_box(b, CIVET, CIVET_CONF)
        seen_tray = best_box(b, TRAY, TRAY_CONF)
        if seen_tray:
            tray = seen_tray
            food_now = any(touches(x, tray) for x in b[(b.cls == FOOD) & (b.conf >= FOOD_CONF)].xyxy.tolist())
            no_food_s = 0 if food_now else no_food_s + 1
            food = food_now or (food and no_food_s < FOOD_GONE_S)
        row = dict(second=t, label="absent", movement="", at_tray=False, food=food, civet_box=civet, tray_box=tray)
        if civet:
            row["at_tray"] = tray is not None and touches(civet, tray, TRAY_PAD)
            if prev:
                row["movement"] = round(body_movement(civet, prev), 3)
                row["label"] = behavior(row["movement"], row["at_tray"], food)
            else:
                row["label"] = "unknown"     # first second of a visit
        prev = civet
        rows.append(row)
        if progress and t % 30 == 0:
            progress(t)
    return clean_up(rows)


def runs(rows):
    """Runs of the same label: (start, end, label)."""
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["label"] == rows[i]["label"]:
            j += 1
        out.append((i, j, rows[i]["label"]))
        i = j
    return out


# smooth the labels over time
def clean_up(rows):
    # short moves inside a meal
    for i, j, lab in runs(rows):
        if lab == "moving" and j - i <= EAT_BRIDGE_S and 0 < i and j < len(rows) \
                and rows[i - 1]["label"] == "eating" and rows[j]["label"] == "eating":
            for r in rows[i:j]: r["label"] = "eating"
    # too-short rests
    for i, j, lab in runs(rows):
        if lab == "resting" and j - i < MIN_REST_S:
            for r in rows[i:j]: r["label"] = "unknown"
    return rows


# --- observation sheet
# seconds -> clock time
def clock(sec, start):
    return (start + dt.timedelta(seconds=sec)).strftime("%H:%M:%S") if start else f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def spans(rows, key):
    """Runs of the same key, bridging short gaps."""
    out, gap = [], 0
    for r in rows:
        k = key(r)
        if k is None:
            gap += 1
            continue
        if out and out[-1]["state"] == k and gap <= BOUT_GAP_S:
            out[-1]["end"] = r["second"] + 1
        else:
            out.append(dict(state=k, start=r["second"], end=r["second"] + 1))
        gap = 0
    return out


# active/inactive bouts for the observation sheet
def bouts(rows, start=None):
    return [dict(no=i + 1, state=b["state"], start=clock(b["start"], start), stop=clock(b["end"], start),
                 start_s=b["start"], end_s=b["end"], minutes=round((b["end"] - b["start"]) / 60, 2))
            for i, b in enumerate(spans(rows, lambda r: ACTIVITY.get(r["label"])))]


# per-hour table
def hourly(rows, start=None):
    base = start or dt.datetime(2000, 1, 1)
    hour = lambda s: (base + dt.timedelta(seconds=s)).replace(minute=0, second=0)
    h = {}
    for r in rows:
        x = h.setdefault(hour(r["second"]), dict(active=0, inactive=0, visible=0, visits=0, feeding=0))
        if r["label"] in ACTIVITY: x[ACTIVITY[r["label"]]] += 1
        x["visible"] += r["label"] != "absent"
    for v in spans(rows, lambda r: None if r["label"] == "absent" else "visit"): h[hour(v["start"])]["visits"] += 1
    for e in spans(rows, lambda r: "eat" if r["label"] == "eating" else None): h[hour(e["start"])]["feeding"] += 1
    name = (lambda t: f"{t:%H:%M}-{t + dt.timedelta(hours=1):%H:%M}") if start else (lambda t: f"hour {int((t - base).total_seconds() // 3600) + 1}")
    return [dict(time=name(t), active_min=round(x["active"] / 60, 1), inactive_min=round(x["inactive"] / 60, 1),
                 visible_min=round(x["visible"] / 60, 1), visits=x["visits"], feeding=x["feeding"],
                 drinking="n/d", urine="n/d", feces="n/d") for t, x in sorted(h.items())]


# runs of the same behavior, in seconds
def intervals(rows):
    return [dict(start=rows[i]["second"], end=rows[j - 1]["second"] + 1, duration=j - i, label=lab) for i, j, lab in runs(rows)]


# seconds and share of each behavior
def summary(rows):
    n = {s: sum(r["label"] == s for r in rows) for s in BEHAVIORS + ["absent"]}
    visible = sum(n[s] for s in BEHAVIORS) or 1
    return {s: dict(seconds=n[s], pct=round(100 * n[s] / visible, 1)) for s in BEHAVIORS + ["absent"]}


# --- output files
# write CSVs, charts and the tracked video
def save_outputs(rows, video, out, start):
    out.mkdir(parents=True, exist_ok=True)
    tables = {"per_second.csv": [{k: v for k, v in r.items() if not k.endswith("_box")} for r in rows],
              "intervals.csv": intervals(rows), "bouts.csv": bouts(rows, start), "hourly.csv": hourly(rows, start)}
    for name, data in tables.items():
        with open(out / name, "w", newline="") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    # timeline
    fig, ax = plt.subplots(figsize=(14, 2.2))
    for lab, col in COLORS.items():
        ax.broken_barh([(i["start"] / 60, i["duration"] / 60) for i in intervals(rows) if i["label"] == lab], (0, 1), color=col, label=lab)
    ax.set_xlim(0, max(len(rows), 1) / 60); ax.set_yticks([]); ax.set_xlabel("minutes (blank = civet not visible)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=False)
    plt.tight_layout(); plt.savefig(out / "timeline.png", dpi=110); plt.close(fig)
    # time budget
    n = {b: sum(r["label"] == b for r in rows) for b in BEHAVIORS}
    n = {k: v for k, v in n.items() if v}
    fig, ax = plt.subplots(figsize=(5, 4))
    if n:
        ax.pie(n.values(), labels=[f"{k}\n{v / 60:.1f} min" for k, v in n.items()], colors=[COLORS[k] for k in n],
               autopct="%1.0f%%", startangle=90, counterclock=False, wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.set_title("Time budget (civet visible)"); plt.tight_layout(); plt.savefig(out / "time_budget.png", dpi=120); plt.close(fig)
    # tracked video
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(VIDEO_SPEED), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "tracked.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bgr = lambda hexcol: tuple(int(hexcol[i:i + 2], 16) for i in (5, 3, 1))
    for (t, f), r in zip(frames(video), rows):
        f = f.copy(); col = bgr(COLORS.get(r["label"], "#ffffff"))
        if r["tray_box"]:
            x1, y1, x2, y2 = map(int, r["tray_box"]); c = (0, 200, 255) if r["food"] else (200, 200, 200)
            cv2.rectangle(f, (x1, y1), (x2, y2), c, 1); cv2.putText(f, "tray+food" if r["food"] else "tray", (x1, y2 + 14), 0, 0.45, c, 1)
        if r["civet_box"]:
            x1, y1, x2, y2 = map(int, r["civet_box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 2); cv2.putText(f, r["label"], (x1, max(14, y1 - 5)), 0, 0.55, col, 2)
        cv2.rectangle(f, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(f, f"{clock(t, None)}  {r['label']}", (6, 17), 0, 0.55, col, 2)
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()


# --- web app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3
UPLOADS, RESULTS = ROOT / "uploads", ROOT / "results"
jobs = {}
lock = threading.Lock()        # one video at a time
model = None

app.jinja_env.filters["hms"] = lambda s: clock(int(s), None)


@app.context_processor
def page_globals():
    return dict(colors=COLORS, behaviors=BEHAVIORS, min_rest=MIN_REST_S, speed=VIDEO_SPEED)


# pages
@app.get("/")
def index():
    return render_template("index.html", jobs=list(jobs.items())[::-1])


@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename: abort(400)
    jid = uuid.uuid4().hex[:8]
    UPLOADS.mkdir(exist_ok=True)
    path = UPLOADS / f"{jid}_{secure_filename(f.filename)}"; f.save(path)
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 0
    jobs[jid] = dict(name=f.filename, status="queued", done=0, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps else 0)
    threading.Thread(target=run_job, args=(jid, path), daemon=True).start()
    return redirect(url_for("job", jid=jid))


# analyse an uploaded video in the background
def run_job(jid, path):
    global model
    j = jobs[jid]
    with lock:
        try:
            j["status"] = "analysing"
            model = model or YOLO(str(MODEL))
            rows = analyse(path, model, progress=lambda t: j.update(done=t))
            start = video_start(j["name"])
            j["status"] = "rendering tracked video"
            save_outputs(rows, path, RESULTS / jid, start)
            j.update(status="done", done=len(rows), total=len(rows), summary=summary(rows), start=start,
                     intervals=[i for i in intervals(rows) if i["label"] != "absent"], bouts=bouts(rows, start), hourly=hourly(rows, start))
        except Exception as e:
            j["status"] = f"failed: {e}"


@app.get("/job/<jid>")
def job(jid):
    j = jobs.get(jid) or abort(404)
    return render_template("job.html" if j["status"] == "done" else "progress.html", jid=jid, j=j)


@app.get("/job/<jid>/status")
def status(jid):
    j = jobs.get(jid) or abort(404)
    return jsonify(status=j["status"], done=j["done"], total=j["total"])


@app.get("/job/<jid>/<name>")
def file(jid, name):
    return send_from_directory(RESULTS / secure_filename(jid), secure_filename(name))


if __name__ == "__main__":
    import os
    app.run(port=int(os.environ.get("PORT", 5001)))
