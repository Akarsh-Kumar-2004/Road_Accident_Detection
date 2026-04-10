import csv
import os
import smtplib
import time
from email.message import EmailMessage

import cv2
import easyocr

from alert_artifacts import create_incident_map_image
from detection import AccidentDetectionModel
from incident_db import init_database, insert_incident_log
from incident_pipeline import IncidentUnderstandingPipeline, draw_incident_overlays
from location_services import (
    build_google_maps_place_url,
    build_google_maps_route_url,
    get_current_pc_location,
    get_nearest_hospitals,
    get_nearest_police_stations,
)
from plate_reader import detect_number_plate, save_plate_image
from video_viewer import FrameViewer

try:
    import pywhatkit
except ImportError:  # pragma: no cover
    pywhatkit = None


reader = easyocr.Reader(["en"])

CAMERA_ID = os.getenv("ACCIDENT_CAMERA_ID", "PC_WEBCAM_VIDEO")
EMAIL_SENDER = "anamolyalert@gmail.com"
EMAIL_PASSWORD = "nceubqmfdbbdvehi"
EMAIL_RECEIVER = "akarshkumar2004@gmail.com"
WHATSAPP_RECEIVER = "+919540034873"
TEMPORAL_WEIGHTS_PATH = os.getenv("ACCIDENT_TEMPORAL_WEIGHTS")
SAVE_FOLDER = "accident_frames"
SAVE_COOLDOWN = 5
LATEST_FRAME_PATH = os.path.join(SAVE_FOLDER, "latest_frame.jpg")
LEGACY_MODEL_ENABLED = os.getenv("ACCIDENT_USE_LEGACY_MODEL", "1") == "1"
LEGACY_THRESHOLD = float(os.getenv("ACCIDENT_LEGACY_THRESHOLD", "0.00"))

os.makedirs(SAVE_FOLDER, exist_ok=True)


def build_alert_message(
    timestamp,
    state,
    location,
    plate_text,
    hospitals,
    legacy_pred,
    legacy_confidence,
):
    nearest_hospital = hospitals[0] if hospitals else None
    hospital_route = "Unavailable"
    if nearest_hospital:
        hospital_route = build_google_maps_route_url(
            (location["lat"], location["lon"]),
            (nearest_hospital["lat"], nearest_hospital["lon"]),
        )
    place_link = build_google_maps_place_url((location["lat"], location["lon"]))

    return "\n".join(
        [
            "ACCIDENT ALERT",
            "",
            f"Camera ID: {CAMERA_ID}",
            f"Detected PC location: {location['place']}",
            f"Coordinates: ({location['lat']}, {location['lon']})",
            f"Google Maps location: {place_link}",
            f"Video timestamp: {timestamp} seconds",
            f"Legacy classifier result: {legacy_pred}",
            f"Legacy classifier confidence: {legacy_confidence * 100:.2f}%",
            f"Temporal support score: {state.temporal_probability * 100:.2f}%",
            f"Summary: {state.summary}",
            f"Detected plate: {plate_text}",
            f"Nearest hospital: {nearest_hospital['name'] if nearest_hospital else 'Unavailable'}",
            f"Hospital route: {hospital_route}",
            f"Attached map image: incident_map_{timestamp:.2f}s.png",
        ]
    )


def send_email(
    image_path,
    map_path,
    plate_image_path,
    timestamp,
    state,
    location,
    plate_text,
    hospitals,
    police_stations,
    legacy_pred,
    legacy_confidence,
):
    msg = EmailMessage()
    msg["Subject"] = "Accident Alert - Detection + Understanding"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    msg.set_content(
        build_alert_message(
            timestamp,
            state,
            location,
            plate_text,
            hospitals,
            legacy_pred,
            legacy_confidence,
        )
    )

    with open(image_path, "rb") as image_file:
        msg.add_attachment(
            image_file.read(),
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path),
        )
    with open(map_path, "rb") as map_file:
        msg.add_attachment(
            map_file.read(),
            maintype="image",
            subtype="png",
            filename=os.path.basename(map_path),
        )
    if plate_image_path and os.path.exists(plate_image_path):
        with open(plate_image_path, "rb") as plate_file:
            msg.add_attachment(
                plate_file.read(),
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(plate_image_path),
            )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("Email sent successfully.")


def send_whatsapp_alert(
    timestamp,
    state,
    location,
    plate_text,
    hospitals,
    legacy_pred,
    legacy_confidence,
):
    if pywhatkit is None:
        print("WhatsApp alert skipped: pywhatkit is not installed.")
        return

    message = build_alert_message(
        timestamp,
        state,
        location,
        plate_text,
        hospitals,
        legacy_pred,
        legacy_confidence,
    )

    try:
        pywhatkit.sendwhatmsg_instantly(
            WHATSAPP_RECEIVER,
            message,
            wait_time=15,
            tab_close=True,
            close_time=3,
        )
        print("WhatsApp alert sent successfully.")
    except Exception as exc:
        print("WhatsApp alert failed:", exc)


def _get_video_path():
    show_dir = "show"
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv")

    for filename in sorted(os.listdir(show_dir)):
        if filename.lower().endswith(supported_extensions):
            return os.path.join(show_dir, filename)

    raise FileNotFoundError("No video file was found inside the 'show' folder.")


def extract_number_plate(frame, vehicles):
    try:
        candidate = detect_number_plate(frame, [vehicle.bbox for vehicle in vehicles], reader)
        if candidate is None:
            return "Not Detected", None
        return candidate.text, candidate
    except Exception as exc:
        print("OCR error:", exc)
        return "OCR Failed", None


def _open_log_writer():
    csv_file = open("accident_log.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "timestamp_sec",
            "temporal_probability",
            "image_path",
            "camera_id",
            "location",
            "latitude",
            "longitude",
            "vehicles",
            "fire_regions",
            "plate_text",
            "plate_image_path",
            "nearest_hospital",
            "nearest_police_station",
            "legacy_classifier",
        ]
    )
    return csv_file, csv_writer


def _format_places(places):
    if not places:
        return "Unavailable"
    return "; ".join(f"{place['name']} ({place['lat']:.5f}, {place['lon']:.5f})" for place in places)


def _draw_plate_overlay(frame, plate_candidate, plate_text):
    if plate_candidate is None:
        return frame

    output = frame.copy()
    x1, y1, x2, y2 = plate_candidate.bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 215, 255), 2)
    cv2.putText(
        output,
        f"Plate: {plate_text}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 215, 255),
        2,
    )
    return output


def startapplication():
    pipeline = IncidentUnderstandingPipeline(
        sequence_length=16,
        temporal_weights_path=TEMPORAL_WEIGHTS_PATH,
        temporal_threshold=0.6,
    )
    legacy_model = None
    if LEGACY_MODEL_ENABLED:
        legacy_model = AccidentDetectionModel("model.json", "model_weights.h5")

    location = get_current_pc_location()
    hospitals = get_nearest_hospitals(location["lat"], location["lon"])
    police_stations = get_nearest_police_stations(location["lat"], location["lon"])
    last_saved_time = 0.0
    email_sent = False
    viewer = FrameViewer("Accident Detection + Understanding")
    init_database()

    csv_file, csv_writer = _open_log_writer()
    video_path = _get_video_path()
    print("Using video:", video_path)
    print("Location source:", location["source"], "-", location["place"])
    print("Viewer mode:", viewer.mode)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            state = pipeline.analyze(frame)
            legacy_pred = "Disabled"
            legacy_confidence = 0.0
            if legacy_model is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(rgb_frame, (250, 250))
                legacy_pred, legacy_prob = legacy_model.predict_accident(roi[None, :, :])
                legacy_confidence = float(max(legacy_prob[0]))

            classifier_triggered = legacy_pred == "Accident" and legacy_confidence >= LEGACY_THRESHOLD
            triggered = state.triggered or classifier_triggered
            annotated_frame = draw_incident_overlays(frame, state)

            cv2.putText(
                annotated_frame,
                f"Legacy classifier: {legacy_pred} ({legacy_confidence * 100:.1f}%)",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if classifier_triggered else (180, 180, 180),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Time: {timestamp:.2f}s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Location: {location['place'][:80]}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(LATEST_FRAME_PATH, annotated_frame)

            print(
                f"Time {timestamp:.2f}s | temporal={state.temporal_probability * 100:.2f}% | "
                f"legacy={legacy_pred}:{legacy_confidence * 100:.2f}% | "
                f"vehicles={len(state.vehicles)} | fire={len(state.fire_regions)}"
            )

            current_time = time.time()
            if triggered and current_time - last_saved_time > SAVE_COOLDOWN:
                filename = os.path.join(
                    SAVE_FOLDER,
                    f"accident_{timestamp:.2f}s_{state.temporal_probability:.2f}.jpg",
                )
                plate_text, plate_candidate = extract_number_plate(frame, state.vehicles)
                annotated_event_frame = _draw_plate_overlay(annotated_frame, plate_candidate, plate_text)
                cv2.imwrite(filename, annotated_event_frame)
                plate_image_path = None
                if plate_candidate is not None:
                    plate_image_path = os.path.join(
                        SAVE_FOLDER,
                        f"plate_{timestamp:.2f}s.jpg",
                    )
                    save_plate_image(plate_candidate, plate_image_path)

                csv_writer.writerow(
                    [
                        round(timestamp, 2),
                        round(state.temporal_probability, 4),
                        filename,
                        CAMERA_ID,
                        location["place"],
                        location["lat"],
                        location["lon"],
                        ",".join(vehicle.label for vehicle in state.vehicles) or "None",
                        len(state.fire_regions),
                        plate_text,
                        plate_image_path or "",
                        hospitals[0]["name"] if hospitals else "Unavailable",
                        police_stations[0]["name"] if police_stations else "Unavailable",
                        f"{legacy_pred}:{legacy_confidence:.4f}",
                    ]
                )
                csv_file.flush()

                print("Accident event saved:", filename)
                print("Incident summary:", state.summary)
                print("Detected plate:", plate_text)
                if plate_image_path:
                    print("Plate image saved:", plate_image_path)
                print("Nearby hospitals:", _format_places(hospitals))
                print("Nearby police stations:", _format_places(police_stations))

                map_path = os.path.join(
                    SAVE_FOLDER,
                    f"incident_map_{timestamp:.2f}s.png",
                )
                create_incident_map_image(map_path, location, hospitals, police_stations)

                incident_record = {
                    "timestamp_sec": round(timestamp, 2),
                    "temporal_probability": round(state.temporal_probability, 4),
                    "image_path": filename,
                    "map_path": map_path,
                    "camera_id": CAMERA_ID,
                    "location": location["place"],
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "vehicles": ",".join(vehicle.label for vehicle in state.vehicles) or "None",
                    "fire_regions": len(state.fire_regions),
                    "plates": plate_text,
                    "plate_image_path": plate_image_path,
                    "nearest_hospital": hospitals[0]["name"] if hospitals else "Unavailable",
                    "nearest_police_station": police_stations[0]["name"] if police_stations else "Unavailable",
                    "legacy_classifier": f"{legacy_pred}:{legacy_confidence:.4f}",
                    "whatsapp_number": WHATSAPP_RECEIVER,
                }
                insert_incident_log(incident_record)

                last_saved_time = current_time

                if not email_sent:
                    try:
                        send_email(
                            filename,
                            map_path,
                            plate_image_path,
                            round(timestamp, 2),
                            state,
                            location,
                            plate_text,
                            hospitals,
                            police_stations,
                            legacy_pred,
                            legacy_confidence,
                        )
                    except Exception as exc:
                        print("Email failed:", exc)

                    try:
                        send_whatsapp_alert(
                            round(timestamp, 2),
                            state,
                            location,
                            plate_text,
                            hospitals,
                            legacy_pred,
                            legacy_confidence,
                        )
                    except Exception as exc:
                        print("WhatsApp alert failed:", exc)

                    email_sent = True
            elif not triggered:
                email_sent = False

            if not viewer.show(annotated_frame):
                break
    finally:
        video.release()
        csv_file.close()
        viewer.close()


if __name__ == "__main__":
    startapplication()
