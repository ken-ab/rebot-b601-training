"""Positive completion evidence for this blue-tissue/black-frame task.

These are image recognition settings, never motion/fault thresholds. Missing
or uncertain visual evidence leaves the existing episode timeout unchanged.
The pi0.5 checkpoint has no trained 'done' output.
"""

from collections import deque
from copy import copy
from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).with_name("b601-task-reference.jpg")
GOAL = np.array([[224, 176], [424, 159], [436, 291], [208, 301]], np.float32)


def align_goal(reference, initial):
    """Fit the printed board before enable, using a crop away from the tissue start."""
    x, y = 70, 90
    template, current = [cv2.cvtColor(image[y:430, x:444], cv2.COLOR_RGB2GRAY)
                         for image in (reference, initial)]
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 120, 1e-5)
    try:
        # Start from a translation so the affine fit does not latch onto a
        # nearby printed line when the board has shifted by several pixels.
        _, seed = cv2.findTransformECC(template, current, np.eye(2, 3, dtype=np.float32),
                                       cv2.MOTION_TRANSLATION, criteria, None, 7)
        score, warp = cv2.findTransformECC(
            template, current, seed, cv2.MOTION_AFFINE, criteria, None, 7)
    except cv2.error:
        return None, None
    if not np.isfinite(score) or not np.isfinite(warp).all():
        return None, None
    # Reject low-confidence or implausible board matches. This only
    # gates the visual success detector, never the allowed motion range.
    scale = np.linalg.svd(warp[:, :2], compute_uv=False)
    shift = np.array([x, y], dtype=np.float32)
    corners = cv2.transform((GOAL - shift)[None], warp)[0] + shift
    if (score < .80 or np.linalg.det(warp[:, :2]) <= 0 or min(scale) < .85
            or max(scale) > 1.15 or np.max(np.abs(corners - GOAL)) > 60):
        return None, float(score)
    return corners, float(score)


class PlacementDetector:
    def __init__(self, reference=REFERENCE, *, initial_rgb=None):
        image = cv2.imdecode(np.frombuffer(Path(reference).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot read completion reference")
        saved = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 480))
        self.registration_score = None
        if initial_rgb is None:
            self.reference = saved
            self.reference_source = "saved_reference"
            self.corners = GOAL
        else:
            self.reference = cv2.resize(initial_rgb, (640, 480))
            self.reference_source = "episode_start"
            self.corners, self.registration_score = align_goal(saved, self.reference)
        self.goal_mask = np.zeros((480, 640), np.uint8)
        # The old outline remains available for diagnostics if registration
        # failed. Only verified registration can end a new real episode.
        outline = GOAL if self.corners is None else self.corners
        cv2.fillConvexPoly(self.goal_mask, np.rint(outline).astype(np.int32), 255)
        # Static objects or shadows outside the placement area cannot be
        # evidence that the tool is still over the deposited tissue.
        self.tool_region = cv2.dilate(self.goal_mask, np.ones((41, 41), np.uint8)) > 0
        self.seen_outside = False
        self.history = deque(maxlen=100)

    def new_window(self):
        """Keep the board match but discard evidence from the old control epoch."""
        detector = copy(self)
        detector.history = deque(maxlen=self.history.maxlen)
        return detector

    def update(self, rgb, seconds):
        image = cv2.resize(np.asarray(rgb), (640, 480))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        blue = cv2.inRange(hsv, (100, 80, 35), (140, 255, 255))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
        result = {"success": False, "seen_outside": self.seen_outside,
                  "reference_source": self.reference_source,
                  "goal_registration_score": self.registration_score,
                  "goal_registered": self.corners is not None}
        if count <= 1:
            self.history.clear()
            return {**result, "evidence": "blue_object_not_visible"}
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 800:
            self.history.clear()
            return {**result, "evidence": "blue_object_uncertain", "blue_pixels": area}
        pixels = labels == label
        fraction = float(np.count_nonzero(pixels & (self.goal_mask > 0)) / area)
        center = centers[label]
        if fraction < 0.2:
            self.seen_outside = True
        # Look for the tool around the goal. Exclude the tissue's own box,
        # then count its black housing/green fingers in the surrounding image.
        x, y, w, h = stats[label, :4]
        tool = ((hsv[:, :, 2] < 90) & (hsv[:, :, 1] < 120)) | (
            (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 100))
        region = self.tool_region.copy()
        region[max(0, y-10):y+h+10, max(0, x-10):x+w+10] = False
        # Existing board printing is background, not the gripper.
        changed = np.max(np.abs(image.astype(np.int16) - self.reference.astype(np.int16)), axis=2) > 45
        tool_count = int(np.count_nonzero(tool & region & changed))
        eligible = (self.corners is not None and self.seen_outside
                    and fraction >= 0.95 and tool_count < 250)
        if eligible:
            if self.history and np.max(np.abs(center - self.history[0][1])) > 5:
                self.history.clear()
            self.history.append((float(seconds), center))
        else:
            self.history.clear()
        stable = (len(self.history) >= 2 and seconds - self.history[0][0] >= 0.75
                  and np.max(np.ptp(np.array([v for _, v in self.history]), axis=0)) <= 5)
        return {**result, "success": bool(eligible and stable), "seen_outside": self.seen_outside,
                "blue_pixels": area, "fraction_in_goal": round(fraction, 4),
                "center_px": center.tolist(), "tool_pixels": tool_count,
                "stable": bool(stable), "evidence": ("goal_registration_unconfirmed" if self.corners is None
                    else "placed_and_tool_withdrawn" if stable else "observing")}
