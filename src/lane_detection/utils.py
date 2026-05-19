import carla
import cv2
import numpy as np
import math

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FPS = 10

ALIGN_THRESHOLD_METERS = 0.20
LOOKAHEAD_DISTANCE = 8.0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def carla_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]


def make_canny_and_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    h, w = canny.shape

    roi_points = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.42), int(h * 0.58)),
        (int(w * 0.58), int(h * 0.58)),
        (int(w * 0.95), h)
    ]], dtype=np.int32)

    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi_points, 255)
    canny_roi = cv2.bitwise_and(canny, mask)

    lines = cv2.HoughLinesP(
        canny_roi,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.4:
                continue

            intercept = y1 - slope * x1

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    def average_line(line_list):
        if len(line_list) == 0:
            return None

        slope, intercept = np.mean(line_list, axis=0)

        if abs(slope) < 1e-6:
            return None

        y1 = h
        y2 = int(h * 0.6)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return x1, y1, x2, y2

    left_line = average_line(left_lines)
    right_line = average_line(right_lines)

    return canny_roi, left_line, right_line


def get_labels(world, vehicle):
    transform = vehicle.get_transform()
    location = vehicle.get_location()

    waypoint = world.get_map().get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:
        return None

    lane_center = waypoint.transform.location
    lane_right = waypoint.transform.get_right_vector()

    dx = location.x - lane_center.x
    dy = location.y - lane_center.y

    signed_offset = dx * lane_right.x + dy * lane_right.y

    if abs(signed_offset) <= ALIGN_THRESHOLD_METERS:
        left = 0.0
        right = 0.0
        aligner = True
    elif signed_offset < 0:
        left = abs(signed_offset)
        right = 0.0
        aligner = False
    else:
        left = 0.0
        right = abs(signed_offset)
        aligner = False

    future_waypoints = waypoint.next(LOOKAHEAD_DISTANCE)

    if len(future_waypoints) > 0:
        vehicle_yaw = transform.rotation.yaw
        future_wp = min(
            future_waypoints,
            key=lambda wp: abs(
                normalize_angle(wp.transform.rotation.yaw - vehicle_yaw)
            )
        )
    else:
        future_wp = waypoint

    future_location = future_wp.transform.location

    inverse_matrix = np.array(transform.get_inverse_matrix())
    future_point = np.array([
        future_location.x,
        future_location.y,
        future_location.z,
        1.0
    ])

    local_point = inverse_matrix.dot(future_point)

    local_x = local_point[0]
    local_y = local_point[1]

    target_angle_deg = math.degrees(math.atan2(local_y, local_x))
    steer_label = clamp(target_angle_deg / 35.0, -1.0, 1.0)

    return {
        "left": left,
        "right": right,
        "signed_offset": signed_offset,
        "aligner": aligner,
        "target_angle_deg": target_angle_deg,
        "steer_label": steer_label
    }


def draw_overlay(frame, left_line, right_line, labels):

    overlay = frame.copy()
    h, w = frame.shape[:2]

    if left_line is not None:
        cv2.line(overlay, left_line[:2], left_line[2:], (0, 255, 0), 5)

    if right_line is not None:
        cv2.line(overlay, right_line[:2], right_line[2:], (0, 255, 0), 5)

    car_center_x = w // 2
    cv2.line(
        overlay,
        (car_center_x, h),
        (car_center_x, int(h * 0.55)),
        (255, 0, 0),
        3
    )

    steer = labels["steer_label"]

    if steer > 0.15:
        direction = "RIGHT"
    elif steer < -0.15:
        direction = "LEFT"
    else:
        direction = "STRAIGHT"

    text1 = f"offset={labels['signed_offset']:.3f}m | aligner={labels['aligner']}"
    text2 = f"angle={labels['target_angle_deg']:.2f} deg | steer={steer:.3f} | {direction}"

    cv2.putText(
        overlay,
        text1,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        overlay,
        text2,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return overlay

