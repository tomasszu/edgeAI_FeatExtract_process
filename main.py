from ReceiveDetections import ReceiveDetectionsService
from SendFeatures import SendFeatures
from CheckDetection import CheckDetection

import yaml
import threading
import argparse
from queue import Queue
import signal
import time

# ---------- args ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_conf', type=str, default='inputs_conf.yaml',
                        help='Path to the input configuration file, that contains crop zones and mqtt topics for receiving info from each camera')
    parser.add_argument('--mqtt_broker', type=str, default='edgejet2.edi.lv', help='Address of the MQTT broker')
    parser.add_argument('--mqtt_port', type=int, default=8884, help='Port of the MQTT broker')
    parser.add_argument('--mqtt_send_topic', type=str, default="reid-vehicle-analysis", help='mqtt topic to send the detections to.')
    parser.add_argument('--mqtt_certs_path', type=str, default="/certs", help='Path to the MQTT certificates')
    parser.add_argument('--cafile', type=str, default=None, help='CA certificate filename (in mqtt_certs_path) for MQTT TLS connection')
    parser.add_argument('--certfile', type=str, default=None, help='Client certificate filename (in mqtt_certs_path) for MQTT TLS connection')
    parser.add_argument('--keyfile', type=str, default=None, help='Client key filename (in mqtt_certs_path) for MQTT TLS connection')
    parser.add_argument('--model_name', type=str, default="sp4_ep6_ft_noCEL_070126_2jet.engine", help='Descriptor for metadata to send with the features, e.g. model name or version')
    return parser.parse_args()

# ---------- globals and shutdown ----------
# Could put maxsize=32 ie for the que, so it doesent bloat, BUT currently it would pose problems with the shutdown .put(None) pill
# that would need to be resoved to use maxsize.
#  but this in case needs to be troubleshooted if its the right fit in terms of max size
inference_queue = Queue() 
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    print("Shutdown signal received, cleaning up...")
    shutdown_event.set()
    # wake worker if blocked
    inference_queue.put(None)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- inference worker ----------
def inference_worker(sender):
    """Single thread that owns the CUDA context and runs inference."""
    # import here to ensure CUDA context created on this thread
    from ExtractingFeatures import ExtractingFeatures

    print("[inference_worker] starting and creating ExtractingFeatures (CUDA context created here)")
    extractor = ExtractingFeatures()

    while True:
        item = inference_queue.get()
        if item is None:
            print("[inference_worker] received shutdown pill")
            break
        track_id, cam_id, image, payload_image, bbox = item
        print(f"[inference_worker] got item for track_id={track_id}, cam_id={cam_id}")
        try:
            features = extractor.get_feature(image)
        except Exception as e:
            print(f"[inference_worker] error during get_feature for track_id={track_id}, cam_id={cam_id}: {e}")
            continue
        try:
            sender(track_id, cam_id, payload_image, features, bbox)
            print(f"[inference_worker] sent features for track_id={track_id}, cam_id={cam_id}")
        except Exception as e:
            print(f"[inference_worker] error while sending features for track_id={track_id}, cam_id={cam_id}: {e}")

    # cleanup extractor (pop/detach CUDA context, free memory)
    try:
        if getattr(extractor, "model", None) and hasattr(extractor.model, "destroy"):
            extractor.model.destroy()
            print("[inference_worker] extractor.model.destroy() called")
    except Exception as e:
        print("[inference_worker] error during extractor cleanup:", e)

    print("Inference worker exiting cleanly.")

# ---------- mqtt processing loop ----------
def process_stream_shared(receiver, cam_map):
    print("[INPUT] Starting MQTT processing loop")

    while not shutdown_event.is_set():
        new_images = receiver.get_pending_images()

        if not new_images:
            # no pending images — avoid hot loop
            time.sleep(0.01)
            continue

        for entry in new_images:
            image = entry["image"]
            track_id = entry["track_id"]
            bbox = entry["bbox"]
            payload_image = entry["payload_image"]
            cam_id = entry.get("cam_id")  # MUST come from MQTT

            if cam_id not in cam_map:
                print(f"[WARN] Unknown cam_id={cam_id}, skipping")
                continue

            cam_data = cam_map[cam_id]
            check = cam_data["check"]
            cam_name = cam_data["name"]

            if check.perform_checks(track_id, bbox):
                print(f"[{cam_name}] enqueueing track_id={track_id}")
                inference_queue.put((track_id, cam_id, image, payload_image, bbox))

# ---------- helper functions ----------

#Per cam state and checks builder
def build_camera_map(config):
    cam_map = {}

    for cam_name, cam_params in config["streams"].items():
        cam_id = cam_params["cam_id"]

        cam_map[cam_id] = {
            "name": cam_name,
            "check": CheckDetection(
                cam_params["crop_zone_rows"],
                cam_params["crop_zone_cols"],
                tuple(cam_params["crop_zone_area_bottom_left"]),
                tuple(cam_params["crop_zone_area_top_right"])
            )
        }

    return cam_map

# ---------- main ----------
def main(cons_args):
    with open(cons_args.input_conf, "r") as f:
        config = yaml.safe_load(f)

    cam_map = build_camera_map(config)

    # assume all topics are the same → take first
    any_cam = next(iter(config["streams"].values()))
    receive_topic = any_cam["mqtt_topic"]

    receiver = ReceiveDetectionsService(
        mqtt_broker=cons_args.mqtt_broker,
        mqtt_port=cons_args.mqtt_port,
        mqtt_certs_path=cons_args.mqtt_certs_path,
        cafile=cons_args.cafile,
        certfile=cons_args.certfile,
        keyfile=cons_args.keyfile,
        mqtt_topic=receive_topic
    )

    sender = SendFeatures(
        mqtt_broker=cons_args.mqtt_broker,
        mqtt_port=cons_args.mqtt_port,
        mqtt_topic=cons_args.mqtt_send_topic,
        mqtt_certs_path=cons_args.mqtt_certs_path,
        cafile=cons_args.cafile,
        certfile=cons_args.certfile,
        keyfile=cons_args.keyfile,
        model_name=cons_args.model_name
    )

    # inference worker (keep as is)
    worker = threading.Thread(
        target=inference_worker,
        args=(sender,),
        name="inference_worker"
    )
    worker.start()

    # single input thread now
    input_thread = threading.Thread(
        target=process_stream_shared,
        args=(receiver, cam_map),
        name="mqtt_input"
    )
    input_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
        inference_queue.put(None)

    print("Shutdown requested")

    input_thread.join(timeout=5)

    inference_queue.put(None)
    worker.join(timeout=5)

    print("All threads stopped.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
