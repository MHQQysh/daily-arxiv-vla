import re
from typing import Dict, Optional


DRIVING_CONTEXT = (
    '(all:"autonomous driving" OR all:"self-driving" OR all:"automated driving" OR '
    '(all:"autonomous vehicle" AND (all:"road" OR all:"traffic" OR all:"driving")) OR '
    'all:"driving scene" OR all:"road vehicle")'
)

TOPIC_QUERIES: Dict[str, str] = {
    "overview": (
        'all:"autonomous driving" OR all:"self-driving" OR all:"automated driving" OR '
        '(all:"autonomous vehicle" AND (all:"road" OR all:"traffic" OR all:"driving"))'
    ),
    "perception": DRIVING_CONTEXT + " AND " + (
        '(all:"perception" OR all:"BEV" OR all:"3D object detection" OR all:"object tracking" OR '
        'all:"lane detection" OR all:"occupancy prediction" OR all:"sensor fusion")'
    ),
    "localization_mapping": DRIVING_CONTEXT + " AND " + (
        '(all:"localization" OR all:"mapping" OR all:"SLAM" OR all:"visual odometry" OR all:"HD map")'
    ),
    "prediction": DRIVING_CONTEXT + " AND " + (
        '(all:"motion forecasting" OR all:"trajectory forecasting" OR all:"motion prediction" OR '
        'all:"trajectory prediction" OR all:"behavior prediction")'
    ),
    "planning_decision": DRIVING_CONTEXT + " AND " + (
        '(all:"motion planning" OR all:"trajectory planning" OR all:"decision making" OR all:"driving policy")'
    ),
    "control": DRIVING_CONTEXT + " AND " + (
        '(all:"vehicle control" OR all:"model predictive control" OR all:"path tracking")'
    ),
    "end_to_end_foundation": (
        '(all:"end-to-end autonomous driving") OR (all:"end-to-end driving") OR '
        '(all:"foundation model" AND ' + DRIVING_CONTEXT + ') OR '
        '(all:"vision language model" AND ' + DRIVING_CONTEXT + ') OR '
        '(all:"world model" AND ' + DRIVING_CONTEXT + ')'
    ),
}

ALLOWED_PRIMARY_CATEGORIES = {
    "cs.CV", "cs.RO", "cs.AI", "cs.LG", "cs.MM", "eess.SY", "eess.IV"
}

STRONG_DRIVING_TERMS = (
    "autonomous driving", "self-driving", "self driving", "automated driving",
    "end-to-end driving", "end to end driving", "road vehicle",
)
DRIVING_CONTEXT_TERMS = ("driving", "road", "traffic", "automotive")
TASK_TERMS = (
    "perception", "bev", "object detection", "tracking", "lane", "occupancy",
    "sensor fusion", "localization", "mapping", "slam", "odometry", "hd map",
    "motion forecasting", "trajectory forecasting", "motion prediction",
    "trajectory prediction", "behavior prediction", "planning",
    "decision making", "driving policy", "vehicle control", "path tracking",
    "foundation model", "vision language model", "world model",
)

STRONG_DRIVING_PATTERNS = tuple(
    re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
    for term in STRONG_DRIVING_TERMS
)
DRIVING_CONTEXT_PATTERNS = tuple(
    re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
    for term in DRIVING_CONTEXT_TERMS
)
TASK_PATTERNS = tuple(
    re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
    for term in TASK_TERMS
)


def get_topic_queries(query_override: Optional[str] = None) -> Dict[str, str]:
    if query_override and query_override.strip():
        return {"custom": query_override.strip()}
    return dict(TOPIC_QUERIES)


def is_relevant_paper(title: str, summary: str) -> bool:
    text = f"{title} {summary}"
    if any(pattern.search(text) for pattern in STRONG_DRIVING_PATTERNS):
        return True
    has_context = any(pattern.search(text) for pattern in DRIVING_CONTEXT_PATTERNS)
    has_task = any(pattern.search(text) for pattern in TASK_PATTERNS)
    return has_context and has_task
