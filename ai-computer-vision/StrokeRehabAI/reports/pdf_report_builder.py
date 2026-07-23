"""
pdf_report_builder.py
=======================
Builds a clinician-facing PDF summary of a patient's session history
and recovery trend.

TODO (next development phase): implement using reportlab or a similar
library once report content/branding requirements are finalized. This
is currently a documented stub so the dashboard's Reports page has a
clear extension point.
"""

from __future__ import annotations

from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class PDFReportBuilder:
    def __init__(self, title: str = "StrokeRehabAI Patient Report"):
        self.title = title

    def build(self, patient_info: Dict, session_summaries: List[Dict], output_path: str) -> str:
        """Render a PDF report to `output_path`.

        TODO: implement with reportlab (canvas + tables) mirroring the
        black-and-white minimal styling used elsewhere in this project's
        documentation generation. Raises NotImplementedError until then.
        """
        raise NotImplementedError(
            "PDF report generation is planned for the next development phase. "
            "Use reports/session_report.py's JSON/CSV export in the meantime."
        )
