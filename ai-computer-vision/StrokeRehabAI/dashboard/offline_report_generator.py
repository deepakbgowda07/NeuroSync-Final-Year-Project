"""
offline_report_generator.py
===========================
Generates single-session rehabilitation reports for offline processed videos.
Formats: PDF (with Matplotlib charts), Excel, CSV, and JSON.
"""

from __future__ import annotations

import sys
# Mask tensorflow to prevent import errors with incompatible protobuf versions in the environment
sys.modules["tensorflow"] = None

import csv
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from dashboard.db import get_connection
from dashboard.patient_manager import PatientManager
from dashboard.session_manager import SessionManager
from dashboard.clinical_metrics import ClinicalMetricsCalculator
from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class SingleSessionReportGenerator:
    """Generates detailed reports for a single offline video session in PDF, Excel, CSV, and JSON."""

    def __init__(self, db_path: str, output_dir: str = "outputs/evaluation_reports/clinical_reports"):
        self.db_path = db_path
        self.output_dir = ensure_dir(output_dir)
        self.patient_manager = PatientManager(db_path)
        self.session_manager = SessionManager(db_path)
        self.metrics_calculator = ClinicalMetricsCalculator()

    def _slugify(self, text: str) -> str:
        import re
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
        return slug or "patient"

    def build_report_data(self, session_id: int) -> Dict:
        """Assembles the metrics, timeline, and recommendations for a single session."""
        detail = self.session_manager.get_session_detail(session_id)
        session = detail["session"]
        patient_id = session["patient_id"]
        
        patient = self.patient_manager.get(patient_id)
        if not patient:
            raise ValueError(f"Patient not found with ID {patient_id}")
            
        scores = self.metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])
        
        # Determine primary joint name & target angles
        exercise_name = session["exercise_name"]
        primary_joint = "elbow" if "elbow" in exercise_name.lower() or "forearm" in exercise_name.lower() else "shoulder"
        
        # Compile timeline events
        timeline_events = []
        # Add reps
        for rep in detail["reps"]:
            status_text = "Correct" if rep["completed"] else "Incomplete"
            timeline_events.append({
                "time_offset_sec": rep["timestamp"] - session["started_at"] if "started_at" in session and session["started_at"] else 0.0,
                "event_type": f"Rep {rep['rep_number']}",
                "detail": status_text,
                "severity": "info" if rep["completed"] else "warning"
            })
            
        # Add errors/compensations
        for event in detail["events"]:
            if event["event_type"] == "error":
                time_offset = event["timestamp"] - session["started_at"] if "started_at" in session and session["started_at"] else 0.0
                timeline_events.append({
                    "time_offset_sec": time_offset,
                    "event_type": "Error: " + event["error_type"].replace("_", " ").title(),
                    "detail": f"Severity: {event['severity']}",
                    "severity": event["severity"]
                })
                
        # Sort timeline by time offset
        timeline_events.sort(key=lambda x: x["time_offset_sec"])
        
        # Build recommendations
        recommendations = []
        if scores.exercise_quality_score < 70.0:
            recommendations.append({
                "category": "quality",
                "priority": "high",
                "text": "Focus on slower, more controlled movement. Exercise quality score is below target."
            })
        if scores.compensation_score < 70.0:
            recommendations.append({
                "category": "compensation",
                "priority": "high",
                "text": "Reduce trunk leaning or shoulder hiking. Consider stabilizing your base while performing reps."
            })
        if scores.elbow_mobility_score < 60.0 and primary_joint == "elbow":
            recommendations.append({
                "category": "rom",
                "priority": "medium",
                "text": "Work on extending your range of motion to hit targeted flexion/extension peaks."
            })
        if scores.shoulder_mobility_score < 60.0 and primary_joint == "shoulder":
            recommendations.append({
                "category": "rom",
                "priority": "medium",
                "text": "Work on extending your shoulder abduction/flexion range of motion peaks."
            })
        if scores.movement_smoothness < 60.0:
            recommendations.append({
                "category": "speed",
                "priority": "medium",
                "text": "Slow down your execution speed to reduce jerky motions and improve stability."
            })
            
        if not recommendations:
            recommendations.append({
                "category": "general",
                "priority": "low",
                "text": "Excellent control and range of motion. Maintain this steady form."
            })
            
        return {
            "generated_at": datetime.now().isoformat(),
            "patient_information": {
                "patient_id": patient.patient_id,
                "full_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "affected_side": patient.affected_side,
                "diagnosis": patient.diagnosis,
                "physiotherapist": patient.physiotherapist,
            },
            "session_information": {
                "session_id": session_id,
                "date": session["started_at"],
                "duration_seconds": session["duration_seconds"],
                "exercise_name": exercise_name,
                "total_reps": session["total_reps"],
                "successful_reps": detail["successful_repetitions"],
                "incorrect_reps": detail["incorrect_repetitions"],
                "average_score": (session["mean_quality"] or 0.0) * 100.0,
                "mean_confidence": (session["mean_confidence"] or 0.0) * 100.0,
                "original_video_path": session.get("original_video_path"),
                "processed_video_path": session.get("processed_video_path"),
            },
            "recovery_scores": scores.to_dict(),
            "timeline": timeline_events,
            "recommendations": recommendations,
            "raw_detail": detail  # kept for internal chart plotting
        }

    def generate(self, session_id: int, report_format: str) -> Path:
        data = self.build_report_data(session_id)
        patient_name_slug = self._slugify(data["patient_information"]["full_name"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"session_{session_id}_{patient_name_slug}_report_{timestamp}"

        format_dispatch = {
            "pdf": self._generate_pdf,
            "excel": self._generate_excel,
            "csv": self._generate_csv,
            "json": self._generate_json,
        }
        builder = format_dispatch.get(report_format.lower())
        if builder is None:
            raise ValueError(f"Unknown report format '{report_format}'. Options: {list(format_dispatch.keys())}")

        output_path = builder(data, base_name)
        self._log_report(data["patient_information"]["patient_id"], session_id, report_format, output_path)
        logger.info("Generated %s session report: %s", report_format, output_path)
        return output_path

    def _log_report(self, patient_id: int, session_id: int, report_format: str, output_path: Path) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO generated_reports (patient_id, session_id, report_type, file_path) VALUES (?, ?, ?, ?)",
                (patient_id, session_id, report_format.lower(), str(output_path)),
            )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def _generate_json(self, data: Dict, base_name: str) -> Path:
        output_path = self.output_dir / f"{base_name}.json"
        # Remove raw_detail before serialization
        serializable_data = {k: v for k, v in data.items() if k != "raw_detail"}
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serializable_data, fh, indent=2, default=str)
        return output_path

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def _generate_csv(self, data: Dict, base_name: str) -> Path:
        output_path = self.output_dir / f"{base_name}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            
            writer.writerow(["Session Report - Patient Information"])
            for key, value in data["patient_information"].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            writer.writerow(["Session Details"])
            for key, value in data["session_information"].items():
                if key != "raw_detail":
                    writer.writerow([key, value])
            writer.writerow([])
            
            writer.writerow(["Recovery Scores"])
            for key, value in data["recovery_scores"].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            writer.writerow(["Timeline Events"])
            writer.writerow(["Offset (s)", "Event Type", "Details", "Severity"])
            for event in data["timeline"]:
                writer.writerow([f"{event['time_offset_sec']:.1f}", event["event_type"], event["detail"], event["severity"]])
            writer.writerow([])
            
            writer.writerow(["Recommendations"])
            for rec in data["recommendations"]:
                writer.writerow([rec["priority"].upper(), rec["category"], rec["text"]])
                
        return output_path

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    def _generate_excel(self, data: Dict, base_name: str) -> Path:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        
        output_path = self.output_dir / f"{base_name}.xlsx"
        wb = Workbook()
        
        # Summary Sheet
        summary_sheet = wb.active
        summary_sheet.title = "Session Summary"
        
        summary_sheet.cell(row=1, column=1, value="Patient Information").font = Font(bold=True, size=12)
        r = 2
        for k, v in data["patient_information"].items():
            summary_sheet.cell(row=r, column=1, value=str(k).replace("_", " ").title())
            summary_sheet.cell(row=r, column=2, value=v)
            r += 1
            
        r += 1
        summary_sheet.cell(row=r, column=1, value="Session Details").font = Font(bold=True, size=12)
        r += 1
        for k, v in data["session_information"].items():
            if k != "raw_detail":
                summary_sheet.cell(row=r, column=1, value=str(k).replace("_", " ").title())
                summary_sheet.cell(row=r, column=2, value=v)
                r += 1
                
        r += 1
        summary_sheet.cell(row=r, column=1, value="Recovery Scores").font = Font(bold=True, size=12)
        r += 1
        for k, v in data["recovery_scores"].items():
            summary_sheet.cell(row=r, column=1, value=str(k).replace("_", " ").title())
            summary_sheet.cell(row=r, column=2, value=v)
            r += 1
            
        # Timeline Sheet
        timeline_sheet = wb.create_sheet("Timeline")
        headers = ["Offset (s)", "Event Type", "Details", "Severity"]
        timeline_sheet.append(headers)
        for cell in timeline_sheet[1]:
            cell.font = Font(bold=True)
            
        for event in data["timeline"]:
            timeline_sheet.append([
                round(event["time_offset_sec"], 1),
                event["event_type"],
                event["detail"],
                event["severity"]
            ])
            
        # Recommendations Sheet
        recs_sheet = wb.create_sheet("Recommendations")
        recs_sheet.append(["Priority", "Category", "Recommendation"])
        for cell in recs_sheet[1]:
            cell.font = Font(bold=True)
        for rec in data["recommendations"]:
            recs_sheet.append([rec["priority"].upper(), rec["category"], rec["text"]])
            
        wb.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # PDF (ReportLab + Matplotlib Charts)
    # ------------------------------------------------------------------
    def _generate_pdf(self, data: Dict, base_name: str) -> Path:
        from reportlab.lib.colors import HexColor, grey
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        
        output_path = self.output_dir / f"{base_name}.pdf"
        detail = data["raw_detail"]
        
        # 1. Generate Matplotlib Charts
        tmp_charts = []
        try:
            # Chart 1: Joint Angle Trajectory
            frames = detail["frames"]
            if frames:
                # Find primary angle name
                angle_keys = set()
                for f in frames:
                    angle_keys.update(f.get("joint_angles", {}).keys())
                
                # Default to elbow or shoulder depending on what exists
                angle_name = None
                for key in ["left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"]:
                    if key in angle_keys:
                        angle_name = key
                        break
                if not angle_name and angle_keys:
                    angle_name = list(angle_keys)[0]
                    
                if angle_name:
                    times = [f.get("session_elapsed_seconds", 0.0) for f in frames]
                    angles = [f.get("joint_angles", {}).get(angle_name, 0.0) for f in frames]
                    
                    plt.figure(figsize=(6, 3))
                    plt.plot(times, angles, label=angle_name.replace("_", " ").title(), color="#16a085")
                    plt.title("Joint Angle Trajectory Over Time")
                    plt.xlabel("Time (seconds)")
                    plt.ylabel("Angle (degrees)")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.legend()
                    plt.tight_layout()
                    
                    fd, path1 = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    plt.savefig(path1, dpi=150)
                    plt.close()
                    tmp_charts.append(path1)
                    
            # Chart 2: ROM peak analysis per repetition
            reps = detail["reps"]
            if reps:
                rep_nums = [r["rep_number"] for r in reps]
                completed_vals = [r["completed"] for r in reps]
                
                plt.figure(figsize=(6, 2.5))
                colors = ["#2ecc71" if c else "#e74c3c" for c in completed_vals]
                plt.bar(rep_nums, [1.0 if c else 0.5 for c in completed_vals], color=colors, width=0.5)
                plt.title("Repetition Completion Peak Results")
                plt.xlabel("Repetition Number")
                plt.ylabel("Completion (1.0 = Success)")
                plt.yticks([0.0, 0.5, 1.0], ["Failed", "Incomplete", "Success"])
                plt.grid(axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout()
                
                fd, path2 = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                plt.savefig(path2, dpi=150)
                plt.close()
                tmp_charts.append(path2)
        except Exception as exc:
            logger.error("Failed to generate report charts: %s", exc)
            
        # 2. Build PDF Document
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("StrokeRehabAI — Single Session Analysis Report", styles["Title"]))
        story.append(Paragraph(f"Generated on: {data['generated_at'][:19].replace('T', ' ')}", styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))
        
        # Patient Info Table
        story.append(Paragraph("Patient Information", styles["Heading2"]))
        patient_data = [
            ["Patient Name", data["patient_information"]["full_name"], "Gender / Age", f"{data['patient_information']['gender']} / {data['patient_information']['age']}"],
            ["Diagnosis", data["patient_information"]["diagnosis"], "Affected Side", data["patient_information"]["affected_side"]],
            ["Physiotherapist", data["patient_information"]["physiotherapist"], "Patient ID", str(data["patient_information"]["patient_id"])]
        ]
        t_pat = Table(patient_data, colWidths=[1.25*inch, 2.25*inch, 1.25*inch, 1.75*inch])
        t_pat.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f2f4f4")),
            ("BACKGROUND", (2, 0), (2, -1), HexColor("#f2f4f4")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t_pat)
        story.append(Spacer(1, 0.15 * inch))
        
        # Session Info Table
        story.append(Paragraph("Session Information", styles["Heading2"]))
        sess_info = data["session_information"]
        sess_data = [
            ["Exercise Name", sess_info["exercise_name"], "Duration", f"{sess_info['duration_seconds'] or 0:.0f} seconds"],
            ["Total Repetitions", str(sess_info["total_reps"]), "Successful / Incorrect", f"{sess_info['successful_reps']} / {sess_info['incorrect_reps']}"],
            ["Average Quality Score", f"{sess_info['average_score']:.1f}%", "Mean Pipeline Confidence", f"{sess_info['mean_confidence']:.1f}%"]
        ]
        t_sess = Table(sess_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1.5*inch])
        t_sess.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f8f9f9")),
            ("BACKGROUND", (2, 0), (2, -1), HexColor("#f8f9f9")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t_sess)
        story.append(Spacer(1, 0.15 * inch))
        
        # Clinical Recovery Scores
        story.append(Paragraph("Biomechanics Recovery Scores", styles["Heading2"]))
        scores = data["recovery_scores"]
        scores_data = [
            ["Shoulder Mobility", f"{scores['shoulder_mobility_score']:.1f}%", "Elbow Mobility", f"{scores['elbow_mobility_score']:.1f}%"],
            ["Movement Stability", f"{scores['movement_stability']:.1f}%", "Movement Smoothness", f"{scores['movement_smoothness']:.1f}%"],
            ["Symmetry Score", f"{scores['symmetry_score']:.1f}%", "Compensation Score", f"{scores['compensation_score']:.1f}%"],
            ["Exercise Quality Index", f"{scores['exercise_quality_score']:.1f}%", "Recovery Index", f"{scores['recovery_index']:.1f}%"]
        ]
        t_scores = Table(scores_data, colWidths=[1.5*inch, 1.75*inch, 1.5*inch, 1.75*inch])
        t_scores.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#e8f8f5")),
            ("BACKGROUND", (2, 0), (2, -1), HexColor("#e8f8f5")),
            ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t_scores)
        story.append(Spacer(1, 0.2 * inch))
        
        # Append charts
        if tmp_charts:
            story.append(Paragraph("Biomechanical Trajectory Visualizations", styles["Heading2"]))
            for img_path in tmp_charts:
                story.append(Image(img_path, width=6*inch, height=2.8*inch))
                story.append(Spacer(1, 0.1 * inch))
            story.append(Spacer(1, 0.1 * inch))
            
        # Clinical Recommendations
        story.append(Paragraph("Clinical Recommendations", styles["Heading2"]))
        for rec in data["recommendations"]:
            p = Paragraph(f"<b>[{rec['priority'].upper()}]</b> {rec['text']}", styles["Normal"])
            story.append(p)
            story.append(Spacer(1, 0.05 * inch))
        story.append(Spacer(1, 0.15 * inch))
        
        # Timeline
        story.append(Paragraph("Session Timeline Events", styles["Heading2"]))
        timeline_rows = [["Time Offset", "Event Type", "Detail", "Severity"]]
        for event in data["timeline"][:30]:  # limit to top 30 events to fit page boundaries
            timeline_rows.append([
                f"{event['time_offset_sec']:.1f}s",
                event["event_type"],
                event["detail"],
                event["severity"].upper()
            ])
            
        t_time = Table(timeline_rows, colWidths=[1.25*inch, 2.25*inch, 2*inch, 1*inch])
        t_time.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, grey),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t_time)
        
        # Build Document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        doc.build(story)
        
        # 3. Clean up charts
        for img_path in tmp_charts:
            try:
                os.remove(img_path)
            except Exception:
                pass
                
        return output_path
