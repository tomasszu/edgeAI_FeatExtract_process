import paho.mqtt.client as mqtt
import json
import time
import uuid



class SendFeatures:
    def __init__(self, mqtt_broker, mqtt_port, mqtt_topic, mqtt_certs_path="certs", cafile=None, certfile=None, keyfile=None, model_name="sp4_ep6_ft_noCEL_070126_26ep.engine"):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.model_name = model_name
        self.connected = False

        # Setup MQTT client
        # Unique client ID per instance
        client_id = f"sender-{uuid.uuid4()}"
        self.client = mqtt.Client(client_id=client_id)

        # TLS (if provided)
        if cafile and certfile and keyfile:
            self.client.tls_set(
                ca_certs=f"{mqtt_certs_path}/{cafile}",
                certfile=f"{mqtt_certs_path}/{certfile}",
                keyfile=f"{mqtt_certs_path}/{keyfile}"
            )
        else:
            print("[WARN] TLS not fully configured, skipping TLS setup")

        #because of self signed certs
        self.client.tls_insecure_set(True)

        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish

        self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
        self.client.loop_start()  # Start in background

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Sender for {self.mqtt_topic} connected successfully")
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_publish(self, client, userdata, mid):
        print("[MQTT] Message published, mid =", mid)
        pass
        
    def __call__(self, track_id, cam_id, payload_image, features, bbox):

        features = features.flatten().tolist() # Convert to list for JSON serialization
        
        data = {
            'track_id': int(track_id),
            'cam_id': cam_id,
            'bbox': bbox,
            'image': payload_image,
            'features': features,
            'model_name': self.model_name,
            'timestamp_ns': time.time_ns()
        }

        self.send_over_mqtt(data)

    def send_over_mqtt(self, data):

        result = self.client.publish(self.mqtt_topic, json.dumps(data))

        # !! DEBUG

        print(f"[MQTT] sent message for track_id {data['track_id']}")

        if result[0] != 0:
            print(f"[MQTT] Failed to send message for track_id {data['track_id']}")