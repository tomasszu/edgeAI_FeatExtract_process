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
    parser.add_argument('--mqtt_broker', type=str, default='localhost', help='Address of the MQTT broker')
    parser.add_argument('--mqtt_port', type=int, default=1884, help='Port of the MQTT broker')
    parser.add_argument('--mqtt_topic', type=str, default="tomass/features", help='mqtt topic to send the detections to.')
    parser.add_argument('--model_name', type=str, default="sp4_ep6_ft_noCEL_070126_26ep.engine", help='Descriptor for metadata to send with the features, e.g. model name or version')
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
        track_id, cam_id, image, payload_image = item
        print(f"[inference_worker] got item for track_id={track_id}, cam_id={cam_id}")
        try:
            features = extractor.get_feature(image)
        except Exception as e:
            print(f"[inference_worker] error during get_feature for track_id={track_id}, cam_id={cam_id}: {e}")
            continue
        try:
            sender(track_id, cam_id, payload_image, features)
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

# ---------- per-camera loop ----------
def process_stream(cam_name, cam_params, args):
    print(f"Starting processing thread for {cam_name}")

    receiver = ReceiveDetectionsService(
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        mqtt_topic=cam_params["mqtt_topic"]
    )

    check = CheckDetection(
        cam_params["crop_zone_rows"],
        cam_params["crop_zone_cols"],
        tuple(cam_params["crop_zone_area_bottom_left"]),
        tuple(cam_params["crop_zone_area_top_right"])
    )

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
            cam_id = cam_params.get("cam_id", 0)

            if check.perform_checks(track_id, bbox):
                print(f"[{cam_name}] enqueueing track_id={track_id}")
                inference_queue.put((track_id, cam_id, image, payload_image))

    print(f"{cam_name} thread exiting cleanly.")

# ---------- main ----------
def main(cons_args):
    with open(cons_args.input_conf, "r") as f:
        config = yaml.safe_load(f)

    sender = SendFeatures(mqtt_broker=args.mqtt_broker, mqtt_port=args.mqtt_port,
                              mqtt_topic=args.mqtt_topic, model_name=args.model_name)

    # Start ONE inference worker (non-daemon so it won't be abruptly killed)
    worker = threading.Thread(target=inference_worker, args=(sender,), name="inference_worker") # komats aiz sender, lai but args plural
    worker.start()

    cam_threads = []
    for cam_name, cam_params in config["streams"].items():
        t = threading.Thread(
            target=process_stream,
            args=(cam_name, cam_params, cons_args),
            name=cam_name
        )
        t.start()
        cam_threads.append(t)

    # Main thread waits for shutdown signal
    try:
        while not shutdown_event.is_set():
            # Use a short sleep rather than signal.pause, so this works in environments
            # where signal.pause isn't available or behaves differently (Docker, etc.)
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
        inference_queue.put(None)

    print("Shutdown requested: joining camera threads")
    # Ask camera threads to finish (they check shutdown_event)
    for t in cam_threads:
        t.join(timeout=5)

    # Ensure worker gets shutdown pill (if one not already sent)
    inference_queue.put(None)
    worker.join(timeout=5)

    print("All threads stopped.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
