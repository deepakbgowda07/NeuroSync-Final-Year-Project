"""Dashboard package: Streamlit clinical UI for StrokeRehabAI —
patient management, live session monitoring, recovery analytics,
clinical scoring, alerts, recommendations, and multi-format reporting."""

from .db import init_db, get_connection
from .patient_manager import PatientManager, Patient
from .session_manager import SessionManager, SessionSummary
from .clinical_metrics import ClinicalMetricsCalculator, ClinicalScores
from .analytics import RecoveryAnalyticsEngine, RecoveryAnalytics
from .recommendation_engine import RecommendationEngine, Recommendation
from .alert_system import AlertSystem, Alert
from .report_generator import ReportGenerator, ReportDataBuilder
from .personalization_engine import PersonalizationEngine, PersonalizedPlan
from .recovery_tracking import RecoveryTracker, RecoveryTrackingReport

__all__ = [
    "init_db",
    "get_connection",
    "PatientManager",
    "Patient",
    "SessionManager",
    "SessionSummary",
    "ClinicalMetricsCalculator",
    "ClinicalScores",
    "RecoveryAnalyticsEngine",
    "RecoveryAnalytics",
    "RecommendationEngine",
    "Recommendation",
    "AlertSystem",
    "Alert",
    "ReportGenerator",
    "ReportDataBuilder",
    "PersonalizationEngine",
    "PersonalizedPlan",
    "RecoveryTracker",
    "RecoveryTrackingReport",
]
