"""
8_Offline_Analysis.py
======================
Streamlit dashboard page for Offline Video Analysis and Session Replay.
Allows users to upload recorded rehab videos, analyze them using the biomechanics pipeline,
interactively scrub through frames, click timeline events, compare sessions, and export reports.
"""

from __future__ import annotations

import sys
# Mask tensorflow to prevent import errors with incompatible protobuf versions in the environment
sys.modules["tensorflow"] = None

import os
import json
import time
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import plotly.express as px

from configs.config_loader import load_config
from dashboard.db import get_connection
from dashboard.patient_manager import PatientManager
from dashboard.session_manager import SessionManager
from dashboard.offline_report_generator import SingleSessionReportGenerator
from inference.offline_analyzer import OfflineVideoAnalyzer, get_video_metadata
from inference.exercise_library import SUPPORTED_EXERCISES
from utils.gpu_utils import get_gpu_info

st.set_page_config(page_title="Offline Video Analysis", layout="wide", page_icon="📹")

cfg = load_config()
db_path = cfg.dashboard.database_path
patient_manager = PatientManager(db_path)
session_manager = SessionManager(db_path)

st.title("📹 Offline Video Analysis & Session Replay")
st.caption("Upload pre-recorded patient exercise videos, run biomechanical assessments, and review results on a frame-by-frame timeline.")

# Load active patients
patients = patient_manager.list_all()
patient_options = {p.full_name: p.patient_id for p in patients}

if not patient_options:
    st.warning("No patients found in the database. Please register a patient first in Patient Management.")
    st.stop()

# Initialize session state keys
if "offline_selected_frame" not in st.session_state:
    st.session_state.offline_selected_frame = 0
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload & Analyze"

# ------------------------------------------------------------------
# Sidebar Info
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("System Configuration")
    gpu = get_gpu_info(cfg.gpu.device_index)
    st.write(f"**GPU Acceleration:** {'🟢 Enabled' if gpu.available else '🔴 Disabled (CPU Fallback)'}")
    if gpu.available:
        st.caption(f"Device: {gpu.device_name}")
    st.write(f"**Target FPS:** {cfg.camera.fps_target} Hz")
    st.write(f"**Standard Resolution:** {cfg.camera.width}x{cfg.camera.height}")
    st.divider()
    st.markdown("### Supported Video Formats\nMP4, AVI, MOV, MKV, WebM")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Session Replay & Frame Review", "Session Comparison"])

# ------------------------------------------------------------------
# Tab 1: Upload & Analyze
# ------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Parameters")
        selected_patient_name = st.selectbox("Select Patient", list(patient_options.keys()))
        patient_id = patient_options[selected_patient_name]
        
        exercise_labels = ["Auto-Detect Exercise"] + [name.replace("_", " ").title() for name in SUPPORTED_EXERCISES]
        selected_exercise_label = st.selectbox("Exercise Mode", exercise_labels)
        
        expected_exercise_key = None
        if selected_exercise_label != "Auto-Detect Exercise":
            expected_exercise_key = selected_exercise_label.lower().replace(" ", "_")
            
        st.divider()
        st.info("💡 **Auto-Detect** uses the spatial-temporal classifier to identify which of the 10 supported exercises is performed in the video.")
        
    with col2:
        st.subheader("Video File Import")
        uploaded_file = st.file_uploader(
            "Drag & Drop or browse for a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Videos shot on mobile phones, tablets, or laptops are accepted."
        )
        
        if uploaded_file is not None:
            # Save uploaded file to temp file
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            temp_video_path = temp_dir / uploaded_file.name
            
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Perform metadata extraction & validation
            metadata = get_video_metadata(str(temp_video_path))
            
            if not metadata["is_valid"]:
                st.error(f"❌ Video Validation Error: {metadata.get('error_msg')}")
                try:
                    os.remove(temp_video_path)
                except:
                    pass
            else:
                st.success("✅ Video integrity validation checks passed successfully.")
                
                # Display Metadata Grid
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
                m_col2.metric("Frame Rate", f"{metadata['fps']:.2f} FPS")
                m_col3.metric("Duration", f"{metadata['duration_seconds']:.1f} s")
                
                m_col4, m_col5, m_col6 = st.columns(3)
                m_col4.metric("Orientation", metadata["orientation"].capitalize())
                m_col5.metric("File Size", f"{metadata['file_size_bytes'] / (1024*1024):.2f} MB")
                m_col6.metric("Metadata Rotation", f"{metadata['rotation_degrees']}°")
                
                # Start button
                if st.button("Start Biomechanical Analysis", type="primary", use_container_width=True):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    def progress_cb(frac, msg):
                        progress_bar.progress(frac)
                        status_text.text(msg)
                        
                    try:
                        analyzer = OfflineVideoAnalyzer(db_path)
                        session_id = analyzer.analyze(
                            str(temp_video_path),
                            patient_id,
                            expected_exercise_key=expected_exercise_key,
                            progress_callback=progress_cb
                        )
                        st.balloons()
                        st.success(f"Analysis completed successfully! Session ID: {session_id} has been logged.")
                        st.session_state.new_session_id = session_id
                        st.session_state.offline_selected_frame = 0
                        # Auto navigate to Replay tab
                        st.info("👉 Switch to the **Session Replay & Frame Review** tab to view results.")
                    except Exception as exc:
                        st.error(f"An error occurred during analysis: {exc}")
                        logger.error("Offline analysis error: %s", exc, exc_info=True)
                    finally:
                        try:
                            os.remove(temp_video_path)
                        except:
                            pass

# ------------------------------------------------------------------
# Tab 2: Session Replay & Frame Review
# ------------------------------------------------------------------
with tab2:
    st.subheader("Select Rehabilitation Session")
    
    # Load session history for selected patient
    # Re-fetch patient ID from selected dropdown (we can sync them via st.session_state or simple selectbox)
    rep_patient_name = st.selectbox("Select Patient Profile", list(patient_options.keys()), key="rep_patient_select")
    rep_patient_id = patient_options[rep_patient_name]
    
    sessions = session_manager.list_sessions(patient_id=rep_patient_id)
    
    if not sessions:
        st.info("No recorded sessions found for this patient. Run an offline analysis or live session first.")
    else:
        # Build select option string
        session_opts = {}
        for s in sessions:
            lbl = f"ID: {s.session_id} | {s.exercise_name} | {s.started_at[:16].replace('T', ' ')} | Quality: {int((s.mean_quality or 0)*100)}%"
            session_opts[lbl] = s.session_id
            
        selected_session_lbl = st.selectbox("Choose Session to Replay", list(session_opts.keys()))
        session_id = session_opts[selected_session_lbl]
        
        # Load full session details
        detail = session_manager.get_session_detail(session_id)
        session = detail["session"]
        frames = detail["frames"]
        reps = detail["reps"]
        events = detail["events"]
        
        st.divider()
        
        # Summary Metrics Grid
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Overall Quality Index", f"{int((session['mean_quality'] or 0)*100)}%")
        s_col2.metric("Total Completed Reps", session["total_reps"] or 0)
        s_col3.metric("Successful / Failed", f"{detail['successful_repetitions']} / {detail['incorrect_repetitions']}")
        s_col4.metric("Session Duration", f"{session['duration_seconds'] or 0:.0f} s")
        
        st.divider()
        
        # Video & Frame scrubbing columns
        p_col1, p_col2 = st.columns([3, 2])
        
        with p_col1:
            st.subheader("Annotated Video Playback")
            processed_path = session.get("processed_video_path")
            if processed_path and Path(processed_path).exists():
                st.video(processed_path)
            else:
                st.warning("Processed annotated video file not found on disk.")
                
            st.divider()
            
            # Interactive Timeline
            st.subheader("Timeline Events")
            if not events and not reps:
                st.caption("No rep or error events recorded for this session.")
            else:
                # Generate Timeline Dataframe
                timeline_rows = []
                for rep in reps:
                    offset = rep["timestamp"] - session["started_at"] if "started_at" in session and session["started_at"] else 0.0
                    if isinstance(offset, str):
                        # fallback if started_at is timestamp string
                        offset = rep["timestamp"]
                    timeline_rows.append({
                        "Time Offset (s)": round(offset, 2),
                        "Event": f"Rep {rep['rep_number']}",
                        "Details": "Completed" if rep["completed"] else "Incomplete",
                        "Severity": "Completed" if rep["completed"] else "Failed",
                        "frame_idx": int(rep["rep_number"] * 25) # rough approximation for jump fallback
                    })
                    
                for idx, ev in enumerate(events):
                    if ev["event_type"] == "error":
                        offset = ev["timestamp"] - session["started_at"] if "started_at" in session and session["started_at"] else 0.0
                        if isinstance(offset, str):
                            offset = ev["timestamp"]
                        timeline_rows.append({
                            "Time Offset (s)": round(offset, 2),
                            "Event": ev["error_type"].replace("_", " ").title(),
                            "Details": f"Severity: {ev['severity'].capitalize()}",
                            "Severity": ev["severity"].capitalize(),
                            "frame_idx": int(offset * cfg.camera.fps_target)
                        })
                
                df_timeline = pd.DataFrame(timeline_rows)
                df_timeline = df_timeline.sort_values(by="Time Offset (s)").reset_index(drop=True)
                
                # Render timeline using Plotly scatter plot
                fig = px.scatter(
                    df_timeline,
                    x="Time Offset (s)",
                    y="Event",
                    color="Severity",
                    hover_data=["Details"],
                    title="Chronological Session Event Timeline (Click on events below to review frames)",
                    color_discrete_map={"Completed": "#2ecc71", "Failed": "#e74c3c", "High": "#9b59b6", "Moderate": "#e67e22", "Minor": "#3498db", "Critical": "#c0392b", "Warning": "#f1c40f", "Info": "#34495e"}
                )
                fig.update_traces(marker=dict(size=14, symbol="diamond"))
                st.plotly_chart(fig, use_container_width=True)
                
                # Selection box for Timeline Jumping
                st.write("🔍 **Event Inspector**: Jump directly to an event's frame")
                event_labels = [f"{row['Time Offset (s)']}s - {row['Event']} ({row['Details']})" for idx, row in df_timeline.iterrows()]
                selected_event_lbl = st.selectbox("Select Event to Jump To", event_labels)
                
                if selected_event_lbl:
                    sel_idx = event_labels.index(selected_event_lbl)
                    chosen_row = df_timeline.iloc[sel_idx]
                    jump_seconds = chosen_row["Time Offset (s)"]
                    target_frame_idx = int(jump_seconds * cfg.camera.fps_target)
                    target_frame_idx = max(0, min(target_frame_idx, len(frames) - 1))
                    
                    if st.button("Jump to Frame"):
                        st.session_state.offline_selected_frame = target_frame_idx
                        st.success(f"Scrubbed to Frame {target_frame_idx} (Time: {jump_seconds}s)")
                        st.rerun()

        with p_col2:
            st.subheader("Interactive Frame Review")
            
            # Frame Scrub Slider
            total_f = len(frames)
            slider_max = max(0, total_f - 1)
            
            # Ensure index stays in bounds
            st.session_state.offline_selected_frame = min(st.session_state.offline_selected_frame, slider_max)
            
            selected_f_idx = st.slider(
                "Select Frame Index",
                0,
                slider_max,
                value=int(st.session_state.offline_selected_frame),
                key="frame_slider"
            )
            st.session_state.offline_selected_frame = selected_f_idx
            
            # Playback Controls Columns
            c_play1, c_play2, c_play3 = st.columns(3)
            with c_play1:
                if st.button("⏪ Prev Frame", disabled=selected_f_idx <= 0):
                    st.session_state.offline_selected_frame -= 1
                    st.rerun()
            with c_play2:
                # Reset playhead
                if st.button("🔄 Reset Head"):
                    st.session_state.offline_selected_frame = 0
                    st.rerun()
            with c_play3:
                if st.button("Next Frame ⏩", disabled=selected_f_idx >= slider_max):
                    st.session_state.offline_selected_frame += 1
                    st.rerun()
                    
            c_jump1, c_jump2 = st.columns(2)
            with c_jump1:
                # Jump to Next Rep Completion
                if st.button("🏃 Next Rep"):
                    current_rep_num = frames[selected_f_idx].get("rep_count", 0)
                    next_rep_idx = None
                    for idx in range(selected_f_idx + 1, len(frames)):
                        if frames[idx].get("rep_count", 0) > current_rep_num:
                            next_rep_idx = idx
                            break
                    if next_rep_idx is not None:
                        st.session_state.offline_selected_frame = next_rep_idx
                        st.success(f"Jumped to Rep {frames[next_rep_idx]['rep_count']}")
                        st.rerun()
                    else:
                        st.info("No further repetitions recorded in session.")
                        
            with c_jump2:
                # Jump to Next Active Error Frame
                if st.button("⚠️ Next Error"):
                    next_err_idx = None
                    # We can find error events matching timeline offsets
                    for idx in range(selected_f_idx + 1, len(frames)):
                        # Look for frames containing non-empty errors in log list or check active_events
                        # Check database session_events for timestamp matching
                        frame_time = frames[idx].get("session_elapsed_seconds", 0.0)
                        has_error = False
                        for ev in events:
                            if ev["event_type"] == "error":
                                offset = ev["timestamp"] - session["started_at"] if "started_at" in session and session["started_at"] else 0.0
                                if abs(offset - frame_time) < 0.5:
                                    has_error = True
                                    break
                        if has_error:
                            next_err_idx = idx
                            break
                    if next_err_idx is not None:
                        st.session_state.offline_selected_frame = next_err_idx
                        st.warning(f"Jumped to Error Frame at {frames[next_err_idx]['session_elapsed_seconds']:.2f}s")
                        st.rerun()
                    else:
                        st.info("No further errors found in session.")
            
            # Read and render single frame snapshot
            if processed_path and Path(processed_path).exists():
                snap_cap = cv2.VideoCapture(processed_path)
                snap_cap.set(cv2.CAP_PROP_POS_FRAMES, selected_f_idx)
                ret_snap, snap_frame = snap_cap.read()
                if ret_snap:
                    st.image(snap_frame, channels="BGR", caption=f"Frame Snapshot: {selected_f_idx} (Time: {frames[selected_f_idx].get('session_elapsed_seconds', 0.0):.2f}s)", use_container_width=True)
                snap_cap.release()
            
            # Display Selected Frame Metrics
            st.markdown("#### Frame Metrics Details")
            curr_frame = frames[selected_f_idx]
            
            m_col1, m_col2 = st.columns(2)
            m_col1.write(f"**Exercise Name:** {curr_frame.get('exercise_name') or '-'}")
            m_col1.write(f"**Movement Phase:** {(curr_frame.get('phase') or '-').replace('_', ' ').capitalize()}")
            m_col1.write(f"**Rep Count:** {curr_frame.get('rep_count') or 0}")
            
            m_col2.write(f"**Movement Quality:** {int((curr_frame.get('movement_quality') or 0)*100)}%")
            m_col2.write(f"**Model Confidence:** {int((curr_frame.get('confidence') or 0)*100)}%")
            m_col2.write(f"**ROM Angle:** {curr_frame.get('rom_deg') or 0.0:.1f}°")
            
            with st.expander("Joint Angles List"):
                st.json(curr_frame["joint_angles"])
                
            # Report Generation
            st.divider()
            st.subheader("Generate Clinical Export Report")
            r_format = st.selectbox("Choose Format", ["PDF", "Excel", "CSV", "JSON"])
            if st.button("Generate Session Report", type="primary"):
                try:
                    rep_generator = SingleSessionReportGenerator(db_path)
                    report_path = rep_generator.generate(session_id, r_format)
                    
                    with open(report_path, "rb") as file_fh:
                        st.download_button(
                            label=f"📥 Download {r_format} Report",
                            data=file_fh.read(),
                            file_name=report_path.name,
                            mime="application/octet-stream"
                        )
                    st.success("Report successfully generated!")
                except Exception as exc:
                    st.error(f"Failed to generate report: {exc}")

# ------------------------------------------------------------------
# Tab 3: Session Comparison
# ------------------------------------------------------------------
with tab3:
    st.subheader("Rehabilitation Session Comparison")
    
    comp_patient_name = st.selectbox("Select Patient for Comparison", list(patient_options.keys()), key="comp_patient_select")
    comp_patient_id = patient_options[comp_patient_name]
    
    patient_sessions = session_manager.list_sessions(patient_id=comp_patient_id)
    
    if len(patient_sessions) < 2:
        st.info("A minimum of two sessions are required to run comparisons.")
    else:
        comp_opts = {}
        for s in patient_sessions:
            lbl = f"ID: {s.session_id} | {s.exercise_name} | {s.started_at[:16].replace('T', ' ')}"
            comp_opts[lbl] = s.session_id
            
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            session_a_lbl = st.selectbox("Primary Session (e.g. Uploaded Session)", list(comp_opts.keys()), index=0)
            session_a_id = comp_opts[session_a_lbl]
        with c_col2:
            session_b_lbl = st.selectbox("Comparison Baseline (e.g. Previous/Live Session)", list(comp_opts.keys()), index=1)
            session_b_id = comp_opts[session_b_lbl]
            
        if session_a_id == session_b_id:
            st.warning("Please choose two different sessions to run the comparison.")
        else:
            # Query details
            detail_a = session_manager.get_session_detail(session_a_id)
            detail_b = session_manager.get_session_detail(session_b_id)
            
            scores_a = SingleSessionReportGenerator(db_path).metrics_calculator.compute_for_session(detail_a["frames"], detail_a["events"], detail_a["reps"])
            scores_b = SingleSessionReportGenerator(db_path).metrics_calculator.compute_for_session(detail_b["frames"], detail_b["events"], detail_b["reps"])
            
            # Display side-by-side comparison
            comp_data = []
            metrics = [
                ("Exercise Quality Score", scores_a.exercise_quality_score, scores_b.exercise_quality_score, "%"),
                ("Recovery index", scores_a.recovery_index, scores_b.recovery_index, ""),
                ("Shoulder Mobility", scores_a.shoulder_mobility_score, scores_b.shoulder_mobility_score, "%"),
                ("Elbow Mobility", scores_a.elbow_mobility_score, scores_b.elbow_mobility_score, "%"),
                ("Movement Stability", scores_a.movement_stability, scores_b.movement_stability, "%"),
                ("Movement Smoothness", scores_a.movement_smoothness, scores_b.movement_smoothness, "%"),
                ("Symmetry Index", scores_a.symmetry_score, scores_b.symmetry_score, "%"),
                ("Compensation Score", scores_a.compensation_score, scores_b.compensation_score, "%"),
                ("Total Completed Repetitions", float(detail_a["session"]["total_reps"] or 0), float(detail_b["session"]["total_reps"] or 0), " reps"),
                ("Success Rate", float(detail_a["successful_repetitions"]) / max(1, detail_a["session"]["total_reps"] or 1)*100.0,
                 float(detail_b["successful_repetitions"]) / max(1, detail_b["session"]["total_reps"] or 1)*100.0, "%"),
            ]
            
            for name, val_a, val_b, unit in metrics:
                delta = val_a - val_b
                delta_str = f"{delta:+.1f}{unit}" if delta != 0 else "-"
                verdict = "🟢 Improvement" if delta > 0.5 else ("🔴 Regression" if delta < -0.5 else "⚪ Stable")
                
                # Check for reverse logic on compensation (lower is better, but here we compare scores, so higher compensation_score is better)
                comp_data.append({
                    "Biomechanics Metric": name,
                    "Session A": f"{val_a:.1f}{unit}",
                    "Session B (Baseline)": f"{val_b:.1f}{unit}",
                    "Change": delta_str,
                    "Verdict": verdict
                })
                
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Side-by-side ROM trajectory comparison
            st.subheader("Joint Angle ROM Trajectory Comparison")
            
            # Plotly overlay trajectory if primary joint angles match
            frames_a = detail_a["frames"]
            frames_b = detail_b["frames"]
            
            if frames_a and frames_b:
                angle_keys_a = set().union(*(f.get("joint_angles", {}).keys() for f in frames_a))
                angle_keys_b = set().union(*(f.get("joint_angles", {}).keys() for f in frames_b))
                common_angles = angle_keys_a.intersection(angle_keys_b)
                
                if common_angles:
                    sel_angle = st.selectbox("Select Angle to Compare Plot", list(common_angles))
                    
                    df_plot_a = pd.DataFrame([
                        {"Time Offset (s)": f.get("session_elapsed_seconds", 0.0), "Angle (deg)": f.get("joint_angles", {}).get(sel_angle, 0.0), "Session": "Session A"}
                        for f in frames_a
                    ])
                    df_plot_b = pd.DataFrame([
                        {"Time Offset (s)": f.get("session_elapsed_seconds", 0.0), "Angle (deg)": f.get("joint_angles", {}).get(sel_angle, 0.0), "Session": "Session B"}
                        for f in frames_b
                    ])
                    
                    df_combined = pd.concat([df_plot_a, df_plot_b]).reset_index(drop=True)
                    
                    fig_comp = px.line(
                        df_combined,
                        x="Time Offset (s)",
                        y="Angle (deg)",
                        color="Session",
                        title=f"Joint Trajectory Overlay: {sel_angle.replace('_', ' ').title()}"
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.caption("No matching joint angles found between these sessions to overlay.")
