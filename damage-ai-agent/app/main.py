# app/main.py
# ============================================================
# DAMAGE LABELING STUDIO
# YOLO + LLM AUTONOMOUS AGENT + HUMAN-IN-THE-LOOP + RTSP/VIDEO + RL
# ============================================================

import sys
from pathlib import Path
import base64
import json
import tempfile
import time

import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

# Optional CV for RTSP / video
try:
    import cv2
except Exception:
    cv2 = None

# ------------------------------------------------------------
# Fix import path
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ------------------------------------------------------------
# Internal imports (your existing modules)
# ------------------------------------------------------------
from app.model import detect_damage
from app.agent import agent_decision
from app.agent_core import autonomous_agent
from app.auto_accept import auto_accept_save
from app.feedback import save_class_feedback, log_error
from app.retrain_manager import should_retrain, trigger_retrain

# RL & memory imports (robust)
try:
    from app.rl_memory import log_rl_step, read_all as read_rl_log
except Exception:
    # log_rl_step is required; read_rl_log optional
    from app.rl_memory import log_rl_step
    read_rl_log = None

try:
    from app.agent_memory import record_confirmation, record_correction
except Exception:
    # Provide safe no-op fallbacks if agent_memory isn't implemented yet
    def record_confirmation(*a, **k):
        return None

    def record_correction(*a, **k):
        return None

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
UPLOAD_DIR = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\incoming")
DATASET_IMG = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\dataset/images/train")
DATASET_LBL = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\dataset/labels/train")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMG.mkdir(parents=True, exist_ok=True)
DATASET_LBL.mkdir(parents=True, exist_ok=True)

CLASSES = ["dent", "hole", "rust", "not_damaged"]

# ------------------------------------------------------------
# Helpers: RTSP capture & video frame extraction
# ------------------------------------------------------------
def capture_from_rtsp(rtsp_url, timeout_seconds=6):
    """Capture a single frame from RTSP and save to UPLOAD_DIR, return Path or None."""
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(rtsp_url)
    start = time.time()
    frame = None
    while time.time() - start < timeout_seconds:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.15)
            continue
        frame = img
        break
    cap.release()
    if frame is None:
        return None
    tmp_path = UPLOAD_DIR / f"rtsp_capture_{int(time.time())}.jpg"
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = frame
    Image.fromarray(img_rgb).save(tmp_path)
    return tmp_path

def extract_frame_from_video(uploaded_video_file, sec=1.0):
    """Save uploaded video to temp, extract a frame at sec seconds, return Path or None."""
    if cv2 is None:
        return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video_file.read())
    tfile.flush()
    path = tfile.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_MSEC, int(sec * 1000))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    frame_path = UPLOAD_DIR / f"video_frame_{int(time.time())}.jpg"
    Image.fromarray(frame).save(frame_path)
    return frame_path

# ------------------------------------------------------------
# Streamlit page layout & input mode controls (top)
# ------------------------------------------------------------
st.set_page_config(page_title="Damage Labeling Studio", layout="wide")
st.title("🧠 Damage Labeling Studio")
st.caption("YOLO Pre-labeling + Autonomous AI Agent + Human-in-the-loop Learning + RL")

st.sidebar.header("Input & Controls")
input_mode = st.sidebar.radio("Input source", ["Upload Image", "RTSP Camera", "Upload Video"])
st.sidebar.markdown("---")

# Simple system control buttons (placeholders / UI)
st.sidebar.subheader("Inspection Controls")
if st.sidebar.button("▶ Start Inspection"):
    st.sidebar.info("Inspection started (single-frame capture mode).")
if st.sidebar.button("⏸ Pause Inspection"):
    st.sidebar.info("Inspection paused.")
if st.sidebar.button("🚩 Flag for Supervisor"):
    st.sidebar.warning("Current item flagged for supervisor review.")

# Health / status indicators
st.sidebar.subheader("System Status")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    st.metric("YOLO", "Online")
with col_b:
    st.metric("Vision LLM", "Online")
col_c, col_d = st.sidebar.columns(2)
with col_c:
    st.metric("Thinking LLM", "Online")
with col_d:
    st.metric("Auto-Accept Rate", "Improving")

st.sidebar.markdown("---")
st.sidebar.caption("RTSP: provide rtsp://user:pass@ip:port/stream\nVideo: extract a single frame for annotation")

# ------------------------------------------------------------
# Input handling (upload / rtsp / video) — produce image_path
# ------------------------------------------------------------
image_path = None

if input_mode == "Upload Image":
    uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_path = UPLOAD_DIR / uploaded.name
        with open(image_path, "wb") as f:
            f.write(uploaded.read())

elif input_mode == "RTSP Camera":
    st.subheader("RTSP Camera Capture")
    rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream")
    if st.button("📸 Capture Frame from RTSP"):
        if not rtsp_url:
            st.error("Enter RTSP URL first")
        else:
            with st.spinner("Capturing frame..."):
                p = capture_from_rtsp(rtsp_url)
                if p:
                    image_path = p
                    st.success(f"Captured frame: {p.name}")
                else:
                    st.error("Failed to capture frame (check RTSP URL and network)")

elif input_mode == "Upload Video":
    st.subheader("Upload Video (we'll extract a frame)")
    video_file = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if video_file:
        sec = st.number_input("Frame time (seconds)", min_value=0.0, value=1.0, step=0.5)
        if st.button("🖼 Extract Frame"):
            with st.spinner("Extracting frame..."):
                p = extract_frame_from_video(video_file, sec=sec)
                if p:
                    image_path = p
                    st.success(f"Frame extracted: {p.name}")
                else:
                    st.error("Failed to extract frame")

# If no image path yet, stop and wait
if image_path is None:
    st.info("Provide an input source (upload, RTSP capture, or video extract) to begin.")
    st.stop()

# ------------------------------------------------------------
# At this point image_path points to the image to process
# ------------------------------------------------------------
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    st.error(f"Unable to open image: {e}")
    st.stop()

W, H = image.size
# safely read image bytes for base64
with open(image_path, "rb") as f:
    _img_bytes = f.read()
img_b64 = base64.b64encode(_img_bytes).decode()

# ------------------------------------------------------------
# YOLO inference
# ------------------------------------------------------------
try:
    yolo_preds, yolo_boxes = detect_damage(str(image_path))
except Exception as e:
    st.error(f"YOLO inference failed: {e}")
    yolo_preds, yolo_boxes = {}, []

# small wrapper to safely call agent
try:
    agent_thought = autonomous_agent(
        image_path=str(image_path),
        yolo_preds=yolo_preds,
        yolo_boxes=yolo_boxes
    )
except Exception as e:
    # safe fallback
    agent_thought = {
        "action": "ASK_HUMAN",
        "reason": f"Agent error: {str(e)}",
        "confidence": "low",
        "damage_type": "unknown"
    }

# Also get a compact yolo decision for UI
try:
    yolo_decision = agent_decision(yolo_preds)
except Exception:
    yolo_decision = {"label": "unknown", "confidence": 0.0}

# ------------------------------------------------------------
# AUTO_ACCEPT (skip UI if confident)
# ------------------------------------------------------------
if agent_thought.get("action") == "AUTO_ACCEPT":
    try:
        auto_accept_save(
            image_path=str(image_path),
            yolo_boxes=yolo_boxes,
            classes=CLASSES,
            W=W,
            H=H,
            dataset_img=DATASET_IMG,
            dataset_lbl=DATASET_LBL
        )
        st.success("🤖 AUTO_ACCEPT: Labels saved automatically")
        # log & feedback
        save_class_feedback(
            str(image_path),
            user_label="auto_accepted",
            model_label=yolo_decision.get("label", "unknown"),
            confidence=yolo_decision.get("confidence", 0.0)
        )
        log_error(str(image_path), yolo_decision.get("label", "unknown"), "auto_accepted")
        if should_retrain():
            trigger_retrain()
            st.info("🔁 Background retraining triggered")
    except Exception as e:
        st.error(f"Auto-accept save failed: {e}")
    st.stop()

# ------------------------------------------------------------
# Convert YOLO boxes → editor init annotations
# ------------------------------------------------------------
init_annotations = []
for b in yolo_boxes:
    if b.get("label") not in CLASSES:
        continue
    x1, y1, x2, y2 = b.get("bbox", [0, 0, 0, 0])
    init_annotations.append({
        "type": "rect",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "class_id": CLASSES.index(b["label"]),
        "class_name": b["label"]
    })

# ------------------------------------------------------------
# Annotation editor HTML (Label Studio–like)
# ------------------------------------------------------------
editor_html = r"""
<style>
#wrap {position: relative; display: inline-block; border: 1px solid #444; max-width:100%;}
#bg {position:absolute; left:0; top:0;}
canvas {position:absolute; left:0; top:0; z-index:2;}
.toolbar button, .toolbar select {margin-right:10px; margin-bottom:6px;}
.toolbar {margin-bottom:10px;}
</style>

<div class="toolbar">
    <button onclick="setMode('rect')">Rectangle</button>
    <button onclick="setMode('poly')">Polygon</button>
    <button onclick="setMode('select')">Select/Edit</button>
    <button onclick="undo()">Undo</button>
    <button onclick="clearAll()">Clear</button>
    <select id="cls_select"></select>
    <button onclick="copyJSON()">Copy JSON</button>
</div>

<div id="wrap">
    <img id="bg">
    <canvas id="layer"></canvas>
</div>

<script>
let IMG_B64 = "IMG_B64_REPLACE";
let W = IMG_W_REPLACE;
let H = IMG_H_REPLACE;
let INIT = INIT_REPLACE;
let CLASS_LIST = CLASSLIST_REPLACE;

let bg = document.getElementById("bg");
bg.src = "data:image/png;base64," + IMG_B64;
bg.width = W; bg.height = H;

let wrap = document.getElementById("wrap");
wrap.style.width = W + "px";
wrap.style.height = H + "px";

let canvas = document.getElementById("layer");
canvas.width = W; canvas.height = H;
let ctx = canvas.getContext("2d");

let clsSel = document.getElementById("cls_select");
CLASS_LIST.forEach((c,i)=>{
    let o=document.createElement("option");
    o.value=i;
    o.text=c;
    clsSel.appendChild(o);
});

let annotations = INIT || [];
let mode="select", down=false, sx=0, sy=0, poly=[], selected=-1;

function setMode(m){ mode=m; selected=-1; drawAll(); }

function drawAll(){
    ctx.clearRect(0,0,W,H);
    annotations.forEach((a,i)=>{
        ctx.strokeStyle=i===selected?"blue":"red";
        ctx.lineWidth=2;
        if(a.type==="rect"){
            let [x1,y1,x2,y2]=a.bbox;
            ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        } else if(a.type==="poly"){
            ctx.beginPath();
            ctx.moveTo(a.points[0].x, a.points[0].y);
            a.points.forEach(p=>ctx.lineTo(p.x,p.y));
            ctx.closePath();
            ctx.stroke();
        }
        ctx.fillStyle="white";
        try {
            ctx.fillText(a.class_name, a.bbox ? a.bbox[0] : a.points[0].x, (a.bbox ? a.bbox[1] : a.points[0].y)-5);
        } catch(e) {}
    });
}

canvas.onmousedown=e=>{down=true; sx=e.offsetX; sy=e.offsetY;}
canvas.onmouseup=e=>{
    if(mode==="rect"){
        let x1=Math.min(sx,e.offsetX), y1=Math.min(sy,e.offsetY);
        let x2=Math.max(sx,e.offsetX), y2=Math.max(sy,e.offsetY);
        let cls=parseInt(clsSel.value);
        annotations.push({
            type:"rect",
            bbox:[x1,y1,x2,y2],
            class_id:cls,
            class_name:CLASS_LIST[cls]
        });
        drawAll();
    } else if(mode==="poly"){
        // poly handled on click/dblclick in simple mode
    }
    down=false;
}

canvas.onclick=e=>{
    if(mode==="poly"){
        poly.push({x:e.offsetX,y:e.offsetY});
        drawAll();
    }
}

canvas.ondblclick=e=>{
    if(mode==="poly" && poly.length>=3){
        let cls=parseInt(clsSel.value);
        annotations.push({type:"poly", points:[...poly], class_id:cls, class_name:CLASS_LIST[cls]});
        poly=[];
        drawAll();
    }
}

function undo(){ annotations.pop(); drawAll(); }
function clearAll(){ annotations=[]; poly=[]; drawAll(); }

function copyJSON(){
    navigator.clipboard.writeText(JSON.stringify(annotations,null,2));
    alert("Copied annotations JSON");
}

drawAll();
</script>
"""

editor_html = (
    editor_html
    .replace("IMG_B64_REPLACE", img_b64)
    .replace("IMG_W_REPLACE", str(W))
    .replace("IMG_H_REPLACE", str(H))
    .replace("INIT_REPLACE", json.dumps(init_annotations))
    .replace("CLASSLIST_REPLACE", json.dumps(CLASSES))
)

# ------------------------------------------------------------
# Layout: left = editor, right = controls & info
# ------------------------------------------------------------
left, right = st.columns([3.5, 1.5])

with left:
    components.html(editor_html, height=min(H + 140, 1200))

with right:
    st.subheader("📋 YOLO Decision")
    st.json(yolo_decision)

    st.subheader("🤖 Autonomous Agent")
    st.json(agent_thought)

    st.divider()

    st.subheader("🛠 Controls")
    if st.button("🚨 Mark for Review"):
        st.warning("Marked for supervisor review")

    if st.button("📁 Download Image"):
        # prepare download link
        with open(image_path, "rb") as f:
            b = f.read()
            href = f"data:application/octet-stream;base64,{base64.b64encode(b).decode()}"
            st.markdown(f"[Download image]({href})")

    st.divider()

    # -----------------------
    # NEW: RL feedback buttons (reward / penalize)
    # -----------------------
    st.subheader("🎯 Agent Feedback (RL)")
    if st.button("❌ Wrong Detection (Penalize Agent)", use_container_width=True):
        # negative reward (safe logging)
        try:
            log_rl_step(
                state={"yolo": yolo_preds, "agent": agent_thought},
                action=agent_thought.get("action"),
                reward=-2.0,
                info={"reason": "human_marked_wrong", "image": str(image_path)}
            )
        except Exception as e:
            st.error(f"RL log failed: {e}")

        # record in memory + feedback
        try:
            record_correction(agent_thought.get("damage_type", "unknown"))
        except Exception:
            pass

        try:
            save_class_feedback(
                str(image_path),
                user_label="wrong_detection",
                model_label=yolo_decision.get("label", "unknown"),
                confidence=yolo_decision.get("confidence", 0.0)
            )
        except Exception:
            pass

        try:
            log_error(str(image_path), yolo_decision.get("label", "unknown"), "wrong_detection")
        except Exception:
            pass

        st.error("❌ Agent penalized (reward = -2.0)")

    if st.button("✅ Correct Detection (Reward Agent)", use_container_width=True):
        # positive reward (safe logging)
        try:
            log_rl_step(
                state={"yolo": yolo_preds, "agent": agent_thought},
                action=agent_thought.get("action"),
                reward=+1.0,
                info={"reason": "human_confirmed", "image": str(image_path)}
            )
        except Exception as e:
            st.error(f"RL log failed: {e}")

        try:
            record_confirmation(agent_thought.get("damage_type", "unknown"))
        except Exception:
            pass

        st.success("✅ Agent rewarded (reward = +1.0)")

    st.divider()

    st.subheader("📥 Paste Annotation JSON")
    ann_json = st.text_area("Paste copied JSON here", height=260, placeholder='Click "Copy JSON" → paste here')

    if st.button("💾 Save to Training Dataset", use_container_width=True):
        if not ann_json.strip():
            st.warning("No annotation JSON provided")
        else:
            try:
                anns = json.loads(ann_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                anns = None

            if anns:
                # save image
                try:
                    DATASET_IMG.joinpath(image_path.name).write_bytes(open(image_path, "rb").read())
                except Exception as e:
                    st.error(f"Failed to save image to dataset: {e}")

                # convert boxes → YOLO txt
                label_file = DATASET_LBL / f"{image_path.stem}.txt"
                try:
                    with open(label_file, "w") as f:
                        for a in anns:
                            if a.get("type") != "rect":
                                continue
                            cls = int(a.get("class_id", 0))
                            x1,y1,x2,y2 = a["bbox"]
                            xc = ((x1+x2)/2)/W
                            yc = ((y1+y2)/2)/H
                            bw = (x2-x1)/W
                            bh = (y2-y1)/H
                            f.write(f"{cls} {xc} {yc} {bw} {bh}\n")
                except Exception as e:
                    st.error(f"Failed to write label file: {e}")

                # feedback logs
                try:
                    save_class_feedback(
                        str(image_path),
                        user_label="human_corrected",
                        model_label=yolo_decision.get("label", "unknown"),
                        confidence=yolo_decision.get("confidence", 0.0)
                    )
                except Exception:
                    pass

                try:
                    log_error(str(image_path), yolo_decision.get("label", "unknown"), "human_corrected")
                except Exception:
                    pass

                try:
                    if should_retrain():
                        trigger_retrain()
                        st.info("🔁 Background retraining triggered")
                except Exception:
                    pass

                st.success("✅ Image + labels saved")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("<hr><p style='text-align:center;color:gray'>Autonomous Damage AI Agent — RTSP & Video enabled — RL integrated</p>", unsafe_allow_html=True)
