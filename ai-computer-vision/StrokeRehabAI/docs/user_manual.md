# User Manual

This manual is for physiotherapists and doctors using StrokeRehabAI
day-to-day. For setup, see `installation_guide.md`; for technical
details, see `architecture.md` and `developer_guide.md`.

## 1. Starting the platform

```powershell
streamlit run dashboard/app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.
The **Home** page shows a clinic-wide overview: active patients,
sessions logged, and any active alerts across all patients.

## 2. Registering a patient

Go to **Patients → Register / Edit**. Fill in the patient's name,
date of birth, gender, affected side, diagnosis, physiotherapist, and
treatment start date. Click **Register Patient**.

To find a patient later, use **Patients → Patient Directory** and
search by name, diagnosis, or physiotherapist.

## 3. Assigning a rehabilitation plan

Go to **Patients → Assign Plan**, select the patient, choose which of
the 10 supported exercises to include, set sessions-per-week, and
click **Assign Plan**. Assigning a new plan automatically deactivates
any previous plan for that patient (history is preserved).

## 4. Running a live session

1. Go to **Live Session**, select the patient, and click **Start
   Session**.
2. A camera window opens. The patient stands where their upper body is
   visible to the webcam.
3. **Calibration** runs automatically first (about 12 seconds) — the
   patient stands still in a neutral pose. If calibration fails
   ("not stable enough"), it's usually because the patient moved or
   part of their upper body was out of frame; it will prompt again.
4. The patient performs exercises. The system automatically detects
   which exercise and which camera view (front/side) they're in — no
   manual selection needed.
5. Live feedback appears on screen: skeleton overlay (green = correct,
   red = a flagged joint), a ghost/target pose, correction arrows,
   rep counter, movement quality, and natural-language feedback
   messages.
6. Press **q** in the camera window, or click **Stop Session** in the
   dashboard, to end.
7. Return to the Live Session page for a live-updating summary, or
   check **Exercise History** afterward for the full record.

## 5. Reviewing recovery progress

Go to **Recovery Analytics**, select the patient. You'll see:
- Recovery score, trend, and improvement percentage
- Nine clinical 0-100 scores for the most recent session (radar chart)
- Active alerts (recovery decrease, compensation increase, missed
  sessions, etc.)
- Auto-generated recommendations, each tied to a specific measured
  condition
- Progress graphs: recovery score over time, ROM improvement, exercise
  accuracy, joint angle trends, compensation frequency, session
  duration, exercise compliance, weekly/monthly progress
- Session-to-session, week-to-week, or month-to-month comparison

## 6. Reviewing session history

Go to **Exercise History**. Filter by patient, browse the session
table, and select any session to **replay** it frame-by-frame
(joint-angle timeline, phase, quality, and any errors/compensation
events logged during that session) — without needing the patient
present or the camera running again.

## 7. Generating a report

Go to **Reports**, select the patient, and click **Generate** for the
format you need (PDF, Excel, CSV, or JSON). Every report includes
patient information, exercise summary, recovery metrics, joint/ROM/
compensation analysis, recommendations, and a session summary.
Previously generated reports for that patient are listed below.

## 8. Understanding alerts

Alerts appear on the Home page (across all patients) and the Recovery
Analytics page (per patient):
- 🔴 **Critical** — e.g. a significant recovery-score drop
- 🟠 **Warning** — e.g. rising compensation, dropping accuracy/ROM,
  skipped sessions
- 🔵 **Info** — e.g. an unusually short session

Alerts are computed live from stored session data — nothing is
pre-scheduled; they always reflect the most recent sessions.

## 9. Demonstration mode (no patient present)

To show the platform without a real patient:

```powershell
python main.py demo --generate-data
```

This creates a synthetic demo patient with several sessions showing a
realistic gradual-improvement trend. Open **Recovery Analytics** or
**Reports** and select the new "Demo Patient" entry.

To demo the live inference engine with a sample video instead of a
webcam:

```powershell
python main.py demo --mode video --source path/to/sample_video.mp4
```

## 10. Important limitations

- This is a movement-quality assessment tool, **not** a diagnostic
  device. Recovery estimates and recommendations are derived from
  measured movement data only — they are not medical diagnoses or
  treatment decisions. Clinical judgment remains with the
  physiotherapist/doctor.
- Forearm pronation/supination and shoulder rotation angles are
  engineering approximations (MediaPipe Pose provides only sparse hand
  landmarks) — see `docs/inference_guide.md` for detail.
- All scoring thresholds are principled defaults pending clinician
  validation against real outcome data (flagged throughout the code
  and docs as `TODO`).
