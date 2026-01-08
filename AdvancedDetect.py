import cv2
import pygame
from ultralytics import YOLO
from collections import deque
import time
import math
import numpy as np
#interpreter 3.9

# Initialize Pygame mixer and sounds
pygame.mixer.init()
sounds = {
    'person': pygame.mixer.Sound("sounds/human.wav"),
    'airplane': pygame.mixer.Sound("sounds/airplane.wav"),
    'dog': pygame.mixer.Sound("sounds/dog.wav"),
    'car': pygame.mixer.Sound("sounds/car.wav"),
    'alert': pygame.mixer.Sound("sounds/alert.wav"),
    'search': pygame.mixer.Sound("sounds/search.wav"),
}
alert_channels = {
    'person': 1, 'airplane': 2, 'dog': 3, 'car': 4, 'alert': 5, 'search': 6,
}

def play_sound(label):
    ch = pygame.mixer.Channel(alert_channels[label])
    if not ch.get_busy():
        ch.play(sounds[label], loops=-1)

def stop_all_except_search():
    for label in alert_classes + ['alert']:
        pygame.mixer.Channel(alert_channels[label]).stop()

def stop_sounds(active_labels):
    for label in alert_classes:
        if label not in active_labels:
            pygame.mixer.Channel(alert_channels[label]).stop()

model = YOLO('yolov8n.pt')
alert_classes = ['person', 'airplane', 'dog', 'car']
detection_log = deque(maxlen=10)

# Drawing functions...
def draw_fancy_box(img, bbox, color=(0, 255, 255), thickness=2, length=20):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_dashed_rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=1, dash_length=10, gap=5):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length + gap):
        x_end = min(x + dash_length, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness)
    for y in range(y1, y2, dash_length + gap):
        y_end = min(y + dash_length, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness)

def draw_radar(img, center, radius, angle, sweep_width=40):
    for r in range(10, radius + 1, 10):
        alpha = 255 - int((r / radius) * 180)
        color = (0, alpha, 0)
        cv2.circle(img, center, r, color, 1, cv2.LINE_AA)
    for i in range(sweep_width):
        a = angle - i
        a_rad = math.radians(a)
        fade = 255 - int(i * (255 / sweep_width))
        sweep_color = (0, fade, 0)
        end_x = int(center[0] + radius * math.cos(a_rad))
        end_y = int(center[1] + radius * math.sin(a_rad))
        cv2.line(img, center, (end_x, end_y), sweep_color, 1, cv2.LINE_AA)
    cv2.circle(img, center, 2, (0, 255, 0), -1)

def draw_compass(img, center, size, angle):
    cv2.circle(img, center, size, (200, 200, 200), 1, cv2.LINE_AA)
    for i, dir in enumerate(["N", "E", "S", "W"]):
        ang = math.radians(i * 90 - angle)
        x = int(center[0] + size * math.cos(ang))
        y = int(center[1] + size * math.sin(ang))
        cv2.putText(img, dir, (x - 7, y + 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
    needle_len = size - 5
    x = int(center[0] + needle_len * math.cos(math.radians(-angle)))
    y = int(center[1] + needle_len * math.sin(math.radians(-angle)))
    cv2.line(img, center, (x, y), (0, 0, 255), 2)

# Webcam setup
cap = cv2.VideoCapture(1)
cv2.namedWindow("High-Tech Object Detector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("High-Tech Object Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_airplane_time = 0
miss_target_duration = 3
frame_count = 0
airplane_alert_played = False
sweep_angle = 0
searching_sound_playing = False
airplane_heading = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    current_time = time.time()
    active_alerts = set()
    results = model(img, stream=True)
    object_count = 0
    airplane_locked = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            object_count += 1

            if label in alert_classes:
                active_alerts.add(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if label == 'airplane':
                    draw_dashed_rectangle(img, (x1, y1), (x2, y2))
                    airplane_locked = True
                    last_airplane_time = current_time
                    if hasattr(box, 'xywh'):
                        cx, cy, w, h = box.xywh[0]
                        airplane_heading = int((cx / img.shape[1]) * 360) % 360
                    width, height = x2 - x1, y2 - y1
                    cv2.line(img, (x1, y2 + 10), (x2, y2 + 10), (0, 255, 0), 1)
                    cv2.putText(img, f"W: {width}px", ((x1 + x2)//2 - 30, y2 + 25),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    cv2.line(img, (x2 + 10, y1), (x2 + 10, y2), (0, 255, 0), 1)
                    cv2.putText(img, f"H: {height}px", (x2 + 15, (y1 + y2)//2),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                else:
                    draw_fancy_box(img, box.xyxy[0])
                x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10
                cv2.putText(img, label.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                play_sound(label)
                detection_log.append(f"[{time.strftime('%H:%M:%S')}] {label.upper()} DETECTED")

    if active_alerts:
        stop_sounds(active_alerts)
        pygame.mixer.Channel(alert_channels['search']).stop()
        searching_sound_playing = False
    else:
        stop_all_except_search()
        if not searching_sound_playing:
            pygame.mixer.Channel(alert_channels['search']).play(sounds['search'], loops=-1)
            searching_sound_playing = True

    if airplane_locked:
        if not airplane_alert_played:
            pygame.mixer.Channel(alert_channels['alert']).play(sounds['alert'])
            airplane_alert_played = True
    else:
        airplane_alert_played = False
        pygame.mixer.Channel(alert_channels['alert']).stop()

    cv2.putText(img, f"OBJECTS: {object_count}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 1)
    h, w, _ = img.shape
    radar_center = (60, 100)
    radar_radius = 40
    compass_center = (60, 190)

    time_since_airplane = current_time - last_airplane_time

    if airplane_locked:
        cv2.putText(img, ">>> TARGET LOCKED <<<", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
        blip_angle = (sweep_angle + 10) % 360
        blip_x = int(radar_center[0] + (radar_radius - 6) * math.cos(math.radians(blip_angle)))
        blip_y = int(radar_center[1] + (radar_radius - 6) * math.sin(math.radians(blip_angle)))
        cv2.circle(img, (blip_x, blip_y), 3, (0, 255, 255), -1)
    else:
        if 0 < time_since_airplane < miss_target_duration:
            if int(frame_count / 15) % 2 == 0:
                cv2.putText(img, ">>> MISSED TARGET <<<", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
        else:
            if int(frame_count / 5) % 2 == 0:
                cv2.putText(img, "SEARCHING TARGET...", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (0, 255, 0), 1)

    draw_radar(img, radar_center, radar_radius, sweep_angle)
    draw_compass(img, compass_center, 30, airplane_heading)

    # <<< ✅ Add heading label here
    cv2.putText(
        img,
        f"HEADING: {airplane_heading}°",
        (compass_center[0] + 40, compass_center[1] + 5),
        cv2.FONT_HERSHEY_PLAIN,
        0.7,
        (0, 255, 0),
        1
    )

    sweep_angle = (sweep_angle + 5) % 360

    for i, log in enumerate(reversed(detection_log)):
        y = h - 10 - (i * 18)
        cv2.putText(img, log, (10, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

    cv2.imshow("High-Tech Object Detector", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
