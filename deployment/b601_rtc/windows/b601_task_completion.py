"""Positive completion evidence for this blue-tissue/black-frame task.

These are image recognition settings, never motion/fault thresholds. Missing
or uncertain visual evidence leaves the existing episode timeout unchanged.
The pi0.5 checkpoint has no trained 'done' output.
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).with_name("b601-task-reference.jpg")
GOAL = np.array([[224, 176], [424, 159], [436, 291], [208, 301]], np.float32)


class PlacementDetector:
    def __init__(self, reference=REFERENCE):
        image = cv2.imdecode(np.frombuffer(Path(reference).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot read completion reference")
        self.reference = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.goal_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(self.goal_mask, GOAL.astype(np.int32), 255)
        self.seen_outside = False
        self.history = deque(maxlen=100)

    def update(self, rgb, seconds):
        image = cv2.resize(np.asarray(rgb), (640, 480))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        blue = cv2.inRange(hsv, (100, 80, 35), (140, 255, 255))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
        result = {"success": False, "seen_outside": self.seen_outside}
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
        region = np.zeros((480, 640), bool)
        region[100:350, 150:500] = True
        region[max(0, y-10):y+h+10, max(0, x-10):x+w+10] = False
        # Existing board printing is background, not the gripper.
        changed = np.max(np.abs(image.astype(np.int16) - self.reference.astype(np.int16)), axis=2) > 45
        tool_count = int(np.count_nonzero(tool & region & changed))
        eligible = self.seen_outside and fraction >= 0.95 and tool_count < 250
        if eligible:
            if self.history and np.max(np.abs(center - self.history[0][1])) > 5:
                self.history.clear()
            self.history.append((float(seconds), center))
        else:
            self.history.clear()
        stable = (len(self.history) >= 2 and seconds - self.history[0][0] >= 0.75
                  and np.max(np.ptp(np.array([v for _, v in self.history]), axis=0)) <= 5)
        return {"success": bool(eligible and stable), "seen_outside": self.seen_outside,
                "blue_pixels": area, "fraction_in_goal": round(fraction, 4),
                "center_px": center.tolist(), "tool_pixels": tool_count,
                "stable": bool(stable), "evidence": "placed_and_tool_withdrawn" if stable else "observing"}
