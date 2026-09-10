from pathlib import Path

from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

POTHOLE_MODEL_PATH = BASE_DIR / "models" / "pothole_v1.pt"
OBSTACLE_MODEL_PATH = BASE_DIR / "yolo11n.pt"


# =========================================================
# LOAD MODELS ONCE
# =========================================================

pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
obstacle_model = YOLO(str(OBSTACLE_MODEL_PATH))


# =========================================================
# DETECTION SETTINGS
# =========================================================

POTHOLE_CONFIDENCE = 0.60

OBSTACLE_CONFIDENCE = 0.40


# =========================================================
# RELATIVE DISTANCE ESTIMATION
# =========================================================
#
# This uses the apparent size of an object in the image.
#
# It is NOT a true measurement in meters.
#
# True physical distance requires depth estimation,
# stereo camera, LiDAR, calibration, etc.
# =========================================================

def estimate_distance(
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height
):

    box_height = max(1, y2 - y1)

    height_ratio = box_height / frame_height

    if height_ratio >= 0.55:
        return "Very Close"

    if height_ratio >= 0.35:
        return "Close"

    if height_ratio >= 0.18:
        return "Medium"

    return "Far"


# =========================================================
# OBJECT DIRECTION
# =========================================================

def calculate_direction(
    x1,
    x2,
    frame_width
):

    center_x = (x1 + x2) / 2

    ratio = center_x / frame_width

    if ratio < 0.35:
        return "Left"

    if ratio > 0.65:
        return "Right"

    return "Center"


# =========================================================
# FORWARD PATH GEOMETRY
# =========================================================
#
# The forward path is a trapezoid:
#
#             CENTER
#             |    |
#            /      \
#           /        \
#          /__________\
#
# Objects inside this region have higher priority.
#
# The path becomes wider toward the bottom because
# nearby objects occupy more image width.
# =========================================================

def point_inside_forward_path(
    x,
    y,
    frame_width,
    frame_height
):

    # Forward path vertical region
    path_top_y = frame_height * 0.35
    path_bottom_y = frame_height

    if y < path_top_y:
        return False

    # Normalized vertical position
    t = (
        y - path_top_y
    ) / (
        path_bottom_y - path_top_y
    )

    # Path boundaries
    #
    # At the top:
    #   42% - 58%
    #
    # At the bottom:
    #   10% - 90%

    left_ratio = (
        0.42 - (0.32 * t)
    )

    right_ratio = (
        0.58 + (0.32 * t)
    )

    left_boundary = (
        frame_width * left_ratio
    )

    right_boundary = (
        frame_width * right_ratio
    )

    return (
        left_boundary <= x <= right_boundary
    )


# =========================================================
# CHECK WHETHER BOUNDING BOX INTERSECTS FORWARD PATH
# =========================================================

def is_in_forward_path(
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height
):

    # Use the bottom-center of the object.
    #
    # This is more useful for navigation because
    # the bottom of the object represents where it
    # meets the road/ground.

    bottom_center_x = (
        x1 + x2
    ) / 2

    bottom_center_y = y2

    return point_inside_forward_path(
        bottom_center_x,
        bottom_center_y,
        frame_width,
        frame_height
    )


# =========================================================
# PRIORITY CALCULATION
# =========================================================

def calculate_priority(
    detection
):

    distance = detection["distance"]

    in_path = detection["in_forward_path"]

    if in_path and distance == "Very Close":
        return "CRITICAL"

    if in_path and distance == "Close":
        return "HIGH"

    if in_path:
        return "HIGH"

    if distance == "Very Close":
        return "MEDIUM"

    return "LOW"


# =========================================================
# CREATE DETECTION OBJECT
# =========================================================

def create_detection(
    class_name,
    object_type,
    confidence,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height
):

    direction = calculate_direction(
        x1,
        x2,
        frame_width
    )

    distance = estimate_distance(
        x1,
        y1,
        x2,
        y2,
        frame_width,
        frame_height
    )

    in_forward_path = is_in_forward_path(
        x1,
        y1,
        x2,
        y2,
        frame_width,
        frame_height
    )

    detection = {
        "class": class_name,
        "type": object_type,
        "confidence": round(
            confidence,
            3
        ),

        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,

        "direction": direction,

        "distance": distance,

        "in_forward_path": in_forward_path,

        "priority": "LOW"
    }

    detection["priority"] = calculate_priority(
        detection
    )

    return detection


# =========================================================
# NAVIGATION DECISION
# =========================================================

def calculate_navigation(
    detections,
    frame_width,
    frame_height
):

    if not detections:

        return {
            "status": "CLEAR",
            "instruction": "PATH CLEAR",
            "reason": "No objects detected",
            "priority": "NONE",
            "target": None
        }


    # -----------------------------------------------------
    # Separate objects by position
    # -----------------------------------------------------

    forward_objects = [
        d for d in detections
        if d["in_forward_path"]
    ]


    left_objects = [
        d for d in detections
        if d["direction"] == "Left"
    ]


    right_objects = [
        d for d in detections
        if d["direction"] == "Right"
    ]


    center_objects = [
        d for d in detections
        if d["direction"] == "Center"
    ]


    # -----------------------------------------------------
    # Sort by priority and apparent size
    # -----------------------------------------------------

    priority_value = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }


    forward_objects.sort(
        key=lambda d: (
            priority_value[d["priority"]],
            (d["y2"] - d["y1"]) *
            (d["x2"] - d["x1"])
        ),
        reverse=True
    )


    # =====================================================
    # CASE 1
    # CRITICAL OBJECT DIRECTLY AHEAD
    # =====================================================

    critical_forward = [
        d for d in forward_objects
        if d["priority"] == "CRITICAL"
    ]


    if critical_forward:

        target = critical_forward[0]

        left_blocked = any(
            d["distance"] in [
                "Very Close",
                "Close"
            ]
            for d in left_objects
        )

        right_blocked = any(
            d["distance"] in [
                "Very Close",
                "Close"
            ]
            for d in right_objects
        )


        # Both sides blocked
        if left_blocked and right_blocked:

            return {
                "status": "STOP",
                "instruction": "WAIT / STOP",
                "reason": (
                    f"{target['class']} ahead "
                    "and both sides are blocked"
                ),
                "priority": "CRITICAL",
                "target": target
            }


        # Left blocked -> move right
        if left_blocked and not right_blocked:

            return {
                "status": "AVOID",
                "instruction": "MOVE RIGHT",
                "reason": (
                    f"{target['class']} ahead; "
                    "right side appears safer"
                ),
                "priority": "CRITICAL",
                "target": target
            }


        # Right blocked -> move left
        if right_blocked and not left_blocked:

            return {
                "status": "AVOID",
                "instruction": "MOVE LEFT",
                "reason": (
                    f"{target['class']} ahead; "
                    "left side appears safer"
                ),
                "priority": "CRITICAL",
                "target": target
            }


        # Both sides appear open
        #
        # Do not blindly tell the user to move.
        # Slow down first because an obstacle is
        # directly ahead.

        return {
            "status": "WARNING",
            "instruction": "SLOW DOWN",
            "reason": (
                f"{target['class']} very close "
                "in forward path"
            ),
            "priority": "CRITICAL",
            "target": target
        }


    # =====================================================
    # CASE 2
    # HIGH PRIORITY OBJECT IN FORWARD PATH
    # =====================================================

    high_forward = [
        d for d in forward_objects
        if d["priority"] == "HIGH"
    ]


    if high_forward:

        target = high_forward[0]


        left_blocked = any(
            d["distance"] in [
                "Very Close",
                "Close"
            ]
            for d in left_objects
        )

        right_blocked = any(
            d["distance"] in [
                "Very Close",
                "Close"
            ]
            for d in right_objects
        )


        if left_blocked and right_blocked:

            return {
                "status": "STOP",
                "instruction": "WAIT / STOP",
                "reason": (
                    f"{target['class']} in forward path; "
                    "both sides have nearby obstacles"
                ),
                "priority": "HIGH",
                "target": target
            }


        if left_blocked:

            return {
                "status": "AVOID",
                "instruction": "MOVE RIGHT",
                "reason": (
                    f"{target['class']} in forward path; "
                    "left side blocked"
                ),
                "priority": "HIGH",
                "target": target
            }


        if right_blocked:

            return {
                "status": "AVOID",
                "instruction": "MOVE LEFT",
                "reason": (
                    f"{target['class']} in forward path; "
                    "right side blocked"
                ),
                "priority": "HIGH",
                "target": target
            }


        return {
            "status": "WARNING",
            "instruction": "SLOW DOWN",
            "reason": (
                f"{target['class']} detected "
                "in forward path"
            ),
            "priority": "HIGH",
            "target": target
        }


    # =====================================================
    # CASE 3
    # SIDE OBSTACLE VERY CLOSE
    # =====================================================

    close_left = [
        d for d in left_objects
        if d["distance"] == "Very Close"
    ]

    close_right = [
        d for d in right_objects
        if d["distance"] == "Very Close"
    ]


    if close_left and not close_right:

        target = close_left[0]

        return {
            "status": "CAUTION",
            "instruction": "KEEP RIGHT",
            "reason": (
                f"{target['class']} very close "
                "on left"
            ),
            "priority": "MEDIUM",
            "target": target
        }


    if close_right and not close_left:

        target = close_right[0]

        return {
            "status": "CAUTION",
            "instruction": "KEEP LEFT",
            "reason": (
                f"{target['class']} very close "
                "on right"
            ),
            "priority": "MEDIUM",
            "target": target
        }


    # =====================================================
    # CASE 4
    # SIDE OBJECTS BUT NOT DANGEROUS
    # =====================================================

    if left_objects or right_objects:

        return {
            "status": "CAUTION",
            "instruction": "PROCEED WITH CAUTION",
            "reason": "Objects detected beside the path",
            "priority": "LOW",
            "target": (
                left_objects[0]
                if left_objects
                else right_objects[0]
            )
        }


    # =====================================================
    # CASE 5
    # OBJECTS DETECTED BUT NOT IN DANGEROUS AREA
    # =====================================================

    return {
        "status": "CLEAR",
        "instruction": "PATH CLEAR",
        "reason": (
            "Detected objects are outside "
            "the immediate forward path"
        ),
        "priority": "LOW",
        "target": None
    }


# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================

def detect(frame):

    frame_height, frame_width = frame.shape[:2]

    detections = []


    # =====================================================
    # 1. POTHOLE DETECTION
    # =====================================================

    pothole_results = pothole_model(
        frame,
        conf=POTHOLE_CONFIDENCE,
        verbose=False
    )


    for result in pothole_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = pothole_model.names[
                class_id
            ]


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            detection = create_detection(
                class_name=class_name,
                object_type="pothole",
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_width=frame_width,
                frame_height=frame_height
            )


            detections.append(
                detection
            )


    # =====================================================
    # 2. GENERAL YOLO OBJECT DETECTION
    # =====================================================
    #
    # IMPORTANT:
    #
    # There is NO class whitelist here.
    #
    # Every object class recognized by yolo11n.pt
    # is returned.
    # =====================================================

    obstacle_results = obstacle_model(
        frame,
        conf=OBSTACLE_CONFIDENCE,
        verbose=False
    )


    for result in obstacle_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = obstacle_model.names[
                class_id
            ]


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            detection = create_detection(
                class_name=class_name,
                object_type="obstacle",
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_width=frame_width,
                frame_height=frame_height
            )


            detections.append(
                detection
            )


    # =====================================================
    # SORT BY IMPORTANCE
    # =====================================================

    priority_value = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }


    detections.sort(
        key=lambda d: (
            priority_value[d["priority"]],
            d["confidence"]
        ),
        reverse=True
    )


    # =====================================================
    # NAVIGATION DECISION
    # =====================================================

    navigation = calculate_navigation(
        detections,
        frame_width,
        frame_height
    )


    # =====================================================
    # RETURN EVERYTHING TO FRONTEND
    # =====================================================

    return {
        "detections": detections,

        "navigation": navigation,

        "frame": {
            "width": frame_width,
            "height": frame_height
        },

        "object_count": len(
            detections
        )
    }