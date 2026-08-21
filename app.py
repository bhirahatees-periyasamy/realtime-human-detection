import collections
import math
import os
import time
import warnings

import cv2
import numpy as np
import streamlit as st
import supervision as sv

warnings.filterwarnings("ignore", category=FutureWarning)

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
MODEL_WEIGHTS = "yolo26m.pt"  # latest Ultralytics YOLO architecture; "m" tier benchmarked best recall/stability vs "s"/"l" on this video
PERSON_CLASS_ID = 0  # COCO class 0 = "person"
CONF_THRESHOLD = 0.3  # lower than the 0.4 first used: real people in this footage (small/partially cut off at frame edges) scored as low as ~0.3
INFERENCE_IMGSZ = 960  # larger than the 640 default: catches smaller/partially-occluded people in this top-down camera angle
FPS_WINDOW = 30
# TrackTrack (CVPR 2025) fuses motion + ReID appearance as a *weighted* cost, unlike BoT-SORT's
# hard appearance-similarity gate. Benchmarked against this video: BoT-SORT+ReID and plain
# ByteTrack both gave ~100-120 "unique" IDs (people fragmented into many short-lived IDs across
# occlusion); TrackTrack+ReID cut that to ~47-49 because the weighted fusion degrades gracefully
# even though this steep top-down angle gives a fairly weak/noisy appearance signal.
# yolo26n-reid.onnx (nano) chosen over the medium ReID checkpoint: same accuracy, ~2x faster
# (8.2fps vs 3.6fps) -- still below the ~13fps source rate, so playback runs slower than
# real-time, but the app already handles that gracefully (no backlog, just slower pacing).
TRACKER_CFG = dict(
    track_high_thresh=0.6,
    track_low_thresh=0.25,
    # Lowered from the 0.7 default: it was higher than CONF_THRESHOLD, so a real, stably-detected
    # person sitting at ~0.67 confidence (verified via crop -- a genuine person, not noise) could
    # never get a track at all, since new_track_thresh gates *starting* a track independently of
    # the detector's own confidence floor. Tested down to 0.3 (matching CONF_THRESHOLD exactly)
    # with zero new spurious tracks on this video, so 0.5 has margin to spare.
    new_track_thresh=0.5,
    track_buffer=90,
    match_thresh=0.7,
    lost_match_thr=0.0,
    iou_weight=0.5,
    reid_weight=0.5,
    conf_weight=0.1,
    angle_weight=0.05,
    penalty_p=0.2,
    penalty_q=0.4,
    reduce_step=0.05,
    tai_thr=0.55,
    min_track_len=3,
    gmc_method="sparseOptFlow",
    with_reid=True,
    model="yolo26n-reid.onnx",
)
# Even with TrackTrack+ReID, a person fully occluded/undetected for a couple of seconds can come
# back under a brand-new tracker ID. Tuning the tracker's own lost-track/matching knobs made this
# *worse*, not better, so identity merging is instead done here: when a never-before-seen tracker
# ID appears near where a just-vanished ID was last seen, it's treated as the same person
# continuing. Distance tolerance is graduated by how long they've been gone: a single missed
# detection (<=SHORT_GAP_FRAMES) gets a generous radius (real per-frame movement measured on this
# video is under ~30px at the 99.5th percentile, but detector jitter can occasionally exceed that);
# beyond that, a *tight* radius is required. This was validated against this video by checking
# actual crops at each merge point: a flat 80px cap at longer gaps produced 2 real errors (two
# different customers near the checkout counter mistaken for one person); tightening long-gap
# matches to 50px removes both without losing any of the correct short-gap merges. Widening the
# time window instead of tightening distance was also tried and made things worse -- this specific
# checkout-counter spot is revisited by many different customers throughout the video, so a longer
# window just means more opportunities to match the wrong person standing in the same place.
REID_MERGE_GAP_FRAMES = 90  # matches track_buffer: how long a vanished person stays eligible for re-matching at all
REID_SHORT_GAP_FRAMES = 5  # a gap this short is almost certainly a single missed detection, not a different person
REID_SHORT_GAP_DIST_PX = 80  # position tolerance for those short gaps
REID_LONG_GAP_DIST_PX = 50  # tighter tolerance once enough time has passed that a different person could occupy the spot
# TODO: replace VIDEO_PATH with st.file_uploader for custom video sources.


@st.cache_resource
def load_model():
    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(MODEL_WEIGHTS)
    return model, device


@st.cache_resource
def load_annotators():
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.TRACK, text_position=sv.Position.TOP_LEFT
    )
    return box_annotator, label_annotator


def format_timestamp(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def resolve_person_id(raw_tracker_id, centroid, frame_index):
    """Map a raw tracker ID to a stable person ID, merging it into a recently-vanished
    person's ID if it appears nearby soon after they were last seen."""
    id_remap = st.session_state.id_remap
    if raw_tracker_id in id_remap:
        return id_remap[raw_tracker_id]

    last_seen = st.session_state.last_seen
    best_person_id, best_dist = None, None
    for person_id, info in last_seen.items():
        gap = frame_index - info["frame"]
        if not (0 < gap <= REID_MERGE_GAP_FRAMES):
            continue
        dist_cap = REID_SHORT_GAP_DIST_PX if gap <= REID_SHORT_GAP_FRAMES else REID_LONG_GAP_DIST_PX
        dist = math.hypot(centroid[0] - info["centroid"][0], centroid[1] - info["centroid"][1])
        if dist < dist_cap and (best_dist is None or dist < best_dist):
            best_dist, best_person_id = dist, person_id

    person_id = best_person_id if best_person_id is not None else raw_tracker_id
    id_remap[raw_tracker_id] = person_id
    return person_id


def build_person_log_df(active_ids):
    import pandas as pd

    rows = [
        {
            "Person ID": tracker_id,
            "In Time": format_timestamp(times["in"]),
            "Out Time": format_timestamp(times["out"]),
            "Duration (s)": round(times["out"] - times["in"], 1),
            "Status": "Active" if tracker_id in active_ids else "Left",
        }
        for tracker_id, times in sorted(st.session_state.person_log.items())
    ]
    return pd.DataFrame(rows)


def init_state():
    defaults = {
        "running": False,
        "cap": None,
        "tracker": None,
        "unique_ids": set(),
        "current_count": 0,
        "fps_estimate": 0.0,
        "frame_times": collections.deque(maxlen=FPS_WINDOW),
        "frame_index": 0,
        "video_ended": False,
        "last_frame_rgb": None,
        "person_log": {},  # person_id -> {"in": seconds, "out": seconds}
        "id_remap": {},  # raw tracker_id -> resolved person_id
        "last_seen": {},  # person_id -> {"centroid": (x, y), "frame": frame_index}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_capture():
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(VIDEO_PATH)


def new_tracker():
    from ultralytics.trackers.track_tracker import TRACKTRACK
    from ultralytics.utils import IterableSimpleNamespace

    cfg = IterableSimpleNamespace(**TRACKER_CFG, device=device)
    st.session_state.tracker = TRACKTRACK(args=cfg)


def start_processing():
    open_capture()
    if st.session_state.tracker is None:
        new_tracker()
    st.session_state.video_ended = False
    st.session_state.running = True


def reset_all():
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = cv2.VideoCapture(VIDEO_PATH)
    new_tracker()
    st.session_state.unique_ids = set()
    st.session_state.current_count = 0
    st.session_state.fps_estimate = 0.0
    st.session_state.frame_times.clear()
    st.session_state.frame_index = 0
    st.session_state.video_ended = False
    st.session_state.last_frame_rgb = None
    st.session_state.person_log = {}
    st.session_state.id_remap = {}
    st.session_state.last_seen = {}


st.set_page_config(page_title="Person Detection & Counting", layout="wide")
init_state()
model, device = load_model()
box_annotator, label_annotator = load_annotators()

st.title("Real-Time Person Detection, Tracking & Counting")

col1, col2, col3 = st.columns(3)
start_clicked = col1.button(
    "Start", width="stretch", disabled=st.session_state.running
)
stop_clicked = col2.button(
    "Stop", width="stretch", disabled=not st.session_state.running
)
reset_clicked = col3.button("Reset", width="stretch")

if start_clicked:
    start_processing()
if stop_clicked:
    st.session_state.running = False
if reset_clicked:
    reset_all()

video_col, stats_col = st.columns([1, 1])
video_ph = video_col.empty()
stats_col.subheader("Stats")
m1 = stats_col.empty()
m2 = stats_col.empty()
m3 = stats_col.empty()

VIDEO_DISPLAY_WIDTH = 640

if st.session_state.last_frame_rgb is not None:
    video_ph.image(
        st.session_state.last_frame_rgb, channels="RGB", width=VIDEO_DISPLAY_WIDTH
    )
else:
    video_ph.info("Click Start to begin processing.")

m1.metric("Current People", st.session_state.current_count)
m2.metric("Unique People Seen", len(st.session_state.unique_ids))
m3.metric("Processing FPS", f"{st.session_state.fps_estimate:.1f}")

if st.session_state.video_ended:
    st.success(
        f"Video finished. Processed {st.session_state.frame_index} frames — "
        f"{len(st.session_state.unique_ids)} unique people detected."
    )

st.subheader("Person In/Out Log")
log_ph = st.empty()
log_ph.dataframe(build_person_log_df(set()), width="stretch", hide_index=True)

if st.session_state.running:
    cap = st.session_state.cap
    tracker = st.session_state.tracker
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 13.0
    target_dt = 1.0 / source_fps

    while st.session_state.running:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            st.session_state.running = False
            st.session_state.video_ended = True
            break

        results = model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONF_THRESHOLD,
            imgsz=INFERENCE_IMGSZ,
            device=device,
            verbose=False,
        )[0]
        tracks = tracker.update(results.boxes.cpu().numpy(), frame)
        centroids = [((row[0] + row[2]) / 2, (row[1] + row[3]) / 2) for row in tracks]
        person_ids = np.array(
            [
                resolve_person_id(int(row[4]), centroid, st.session_state.frame_index)
                for row, centroid in zip(tracks, centroids)
            ],
            dtype=int,
        )
        for person_id, centroid in zip(person_ids, centroids):
            st.session_state.last_seen[int(person_id)] = {
                "centroid": centroid,
                "frame": st.session_state.frame_index,
            }

        detections = sv.Detections(
            xyxy=tracks[:, 0:4] if len(tracks) else np.empty((0, 4)),
            confidence=tracks[:, 5] if len(tracks) else np.empty(0),
            class_id=(tracks[:, 6].astype(int) if len(tracks) else np.empty(0, dtype=int)),
            tracker_id=person_ids,
        )

        current_ids = {int(tid) for tid in detections.tracker_id}
        st.session_state.current_count = len(detections)
        st.session_state.unique_ids.update(current_ids)
        st.session_state.frame_index += 1

        video_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / source_fps
        for tracker_id in current_ids:
            if tracker_id not in st.session_state.person_log:
                st.session_state.person_log[tracker_id] = {
                    "in": video_time,
                    "out": video_time,
                }
            else:
                st.session_state.person_log[tracker_id]["out"] = video_time

        labels = [
            f"#{int(tid)} {conf:.2f}"
            for tid, conf in zip(detections.tracker_id, detections.confidence)
        ]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame_rgb = annotated_rgb
        video_ph.image(annotated_rgb, channels="RGB", width=VIDEO_DISPLAY_WIDTH)

        now = time.time()
        st.session_state.frame_times.append(now)
        if len(st.session_state.frame_times) >= 2:
            span = st.session_state.frame_times[-1] - st.session_state.frame_times[0]
            st.session_state.fps_estimate = (
                (len(st.session_state.frame_times) - 1) / span if span > 0 else 0.0
            )

        m1.metric("Current People", st.session_state.current_count)
        m2.metric("Unique People Seen", len(st.session_state.unique_ids))
        m3.metric("Processing FPS", f"{st.session_state.fps_estimate:.1f}")
        log_ph.dataframe(build_person_log_df(current_ids), width="stretch", hide_index=True)

        elapsed = time.time() - loop_start
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if st.session_state.video_ended:
        st.success(
            f"Video finished. Processed {st.session_state.frame_index} frames — "
            f"{len(st.session_state.unique_ids)} unique people detected."
        )
