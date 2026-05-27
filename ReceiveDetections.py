import time
import uuid
import json

import cv2
import cbor2
import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image


class ReceiveDetectionsService:
    def __init__(
        self,
        mqtt_broker,
        mqtt_port,
        mqtt_topic,
        mqtt_certs_path="certs",
        cafile=None,
        certfile=None,
        keyfile=None
    ):

        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.queue = []

        # Unique client ID per instance
        client_id = f"receiver-{uuid.uuid4()}"
        self.client = mqtt.Client(client_id=client_id)

        if cafile:
            self.client.tls_set(
                ca_certs=f"{mqtt_certs_path}/{cafile}",
                certfile=f"{mqtt_certs_path}/{certfile}",
                keyfile=f"{mqtt_certs_path}/{keyfile}"
            )

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False

        self.client.connect(mqtt_broker, mqtt_port, 60)
        self.client.loop_start()  # non-blocking

        self.msg_count = 0

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Receiver for {self.mqtt_topic} connected successfully")
            client.subscribe(self.mqtt_topic)
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):

        try:

            payload = msg.payload

            # --------------------------------------------------
            # OLD JSON MESSAGES
            # --------------------------------------------------

            try:

                json_payload = json.loads(
                    payload.decode("utf-8")
                )

                # old detection metadata messages
                # can safely be ignored
                if isinstance(json_payload, list):

                    print("[MQTT] JSON detection metadata received")
                    return

            except Exception:
                pass

            # --------------------------------------------------
            # NEW CBOR CROP_BATCH MESSAGES
            # --------------------------------------------------

            data = cbor2.loads(payload)

            if not isinstance(data, dict):
                return

            if data.get("type") != "crop_batch":
                return

            cam_id = data.get("cam")
            items = data.get("items", [])

            for item in items:

                # ----------------------------------------------
                # CLASS FILTER
                # ----------------------------------------------

                cls_name = str(
                    item.get("cls", "")
                ).lower()

                # car, motorcycle, bus, truck
                allowed_classes = [
                    "car",
                    "motorcycle",
                    "bus",
                    "truck"
                ]

                if cls_name not in allowed_classes:
                    print(f"[MQTT] Ignoring detection of class: {cls_name}")
                    continue

                # ----------------------------------------------
                # RAW JPEG BYTES
                # ----------------------------------------------

                img_bytes = item.get("img")

                if not isinstance(
                    img_bytes,
                    (bytes, bytearray)
                ):
                    continue

                # ----------------------------------------------
                # PRESERVE OLD FORMAT
                # ----------------------------------------------

                # old implementation expected:
                # payload["image"] -> HEX STRING

                encoded_crop = img_bytes.hex()

                # decode exactly the same way as before
                image_np = self._decode_crop_np(encoded_crop)

                # ----------------------------------------------
                # KEEP OUTPUT FORMAT IDENTICAL
                # ----------------------------------------------

                self.queue.append({
                    "track_id": item.get("track_id"),
                    "cam_id": cam_id,
                    "bbox": item.get("bbox"),
                    "image": image_np,
                    "payload_image": encoded_crop
                })

                print(
                    f"[MQTT] Received and decoded "
                    f"crop for track_id "
                    f"{item.get('track_id')}"
                )

                self.msg_count += 1

                print(
                    f"[MQTT] Total messages received: "
                    f"{self.msg_count}"
                )

        except Exception as e:

            print(
                f"[ERROR] Failed to process "
                f"MQTT message: {e}"
            )

    def on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected (rc={rc}), reconnecting...")

    # ======================================================
    # IMAGE DECODERS
    # ======================================================

    def _decode_crop_np(self, encoded_crop):

        crop_bytes = bytes.fromhex(encoded_crop)

        np_arr = np.frombuffer(
            crop_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Failed to decode image"
            )

        return image

    def _decode_crop_pil(self, encoded_crop):

        crop_bytes = bytes.fromhex(encoded_crop)

        np_arr = np.frombuffer(
            crop_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Failed to decode image"
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(image_rgb)

    # ======================================================
    # QUEUE ACCESS
    # ======================================================

    def get_pending_images(self):
        """
        Retrieve and clear the queue
        of received crops.
        """

        data = self.queue[:]
        self.queue.clear()

        return data