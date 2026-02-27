import numpy as np
import time

class CheckDetection:
    def __init__(self, rows=None, cols=None,
                 area_bottom_left = None, area_top_right=None, use_stationary_surpression=True, max_idle_seconds=300):

        self.zones = []

        # Crop zone parameters
        self.rows = rows
        self.cols = cols

        # Set the cropping area
        self.area_bottom_left = area_bottom_left  # (x_min, y_max)
        self.area_top_right = area_top_right      # (x_max, y_min)

        self.zone_of_detections = {}

        self.min_crop_size = 70  # Minimum size for the bounding box to be considered valid (in pixels)

        # Cached entry spatio-temporal cache for detections, to avoid sending multiple crops of the same vehicle if it is standing still and tracker id resets (e.g. due to occlusion or tracker limitations)

        self.use_stationary_surpression = use_stationary_surpression # Store True
        self.cached_detections = {} #dict
        """ {
            "center": (cx, cy),
            "bbox": [x1, y1, x2, y2],
            "timestamp": time.monotonic()
            }
        """

        self.max_idle_seconds = max_idle_seconds  # Maximum time for detection in cache (default 5 min)
        
        # asserts that the points are of valid structure
        def _assert_point(name, point):
            assert isinstance(point, (tuple, list)), f"{name} must be a tuple or list"
            assert len(point) == 2, f"{name} must have exactly two elements (x, y)"
            assert all(isinstance(v, (int, float)) for v in point), f"{name} values must be int or float"

        if self.rows is not None and self.cols is not None \
        and self.area_bottom_left is not None and self.area_top_right is not None:

            _assert_point("area_bottom_left", self.area_bottom_left)
            _assert_point("area_top_right", self.area_top_right)

            self.zones = self._generate_zones()
            self.use_zones = True
        else:
            self.zones = []
            self.use_zones = False  # zone filtering disabled

    def _generate_zones(self):
        """Generates the zones based on the specified rows and columns within the defined area.
        Returns:
            list: A list of tuples representing the zones, each defined by its top-left and bottom-right coordinates.
        """
        x_min, y_max = self.area_bottom_left
        x_max, y_min = self.area_top_right

        zone_width = (x_max - x_min) / self.cols
        zone_height = (y_max - y_min) / self.rows
        zones = []

        for i in range(self.rows):
            for j in range(self.cols):
                x1 = x_min + j * zone_width
                y1 = y_min + i * zone_height
                x2 = x_min + (j + 1) * zone_width
                y2 = y_min + (i + 1) * zone_height
                zones.append((int(x1), int(y1), int(x2), int(y2)))

        return zones
    
    def check_min_crop_size(self, bbox):
        x1, y1, x2, y2 = bbox

        width  = x2 - x1
        height = y2 - y1

        min_size = self.min_crop_size

        if width < min_size or height < min_size:
            return False

        return True
    
    def check_crop_zones(self, track_id, bbox):
        center = self.get_center(bbox)
        zone = self.zone_of_point(center)

        # Only move forward if zone changed and vehicle moved > pixel threshold
        # shis ir lai vehicle nelēkātu starp zonām, ja tā stāv uz robežas (np linalg norm ļauj mums dabūt euclidean distance starp diviem punktiem)
        threshold = 10  # pixels, adjust based on your frame resolution

        last_zone, last_center = self.zone_of_detections.get(track_id, (None, None))

        if zone == -1:
            if track_id in self.zone_of_detections and \
                self.get_point_distance(center, last_center) > threshold:
                
                # remove from zone tracking
                del self.zone_of_detections[track_id]

                # remove from stationary surpression spatial cache
                self._purge_track_from_static_cache(track_id)

            return False

        if last_zone is None:
            self.zone_of_detections[track_id] = (zone, center)
            return True
        elif last_zone != zone and (last_center is not None and self.get_point_distance(center, last_center) > threshold):
            self.zone_of_detections[track_id] = (zone, center)
            return True


        return False
    
    def check_static_detection(self, track_id, bbox):
        """ 
        Checks if detection is a stationary vehicle whose tracker id reset.
        Returns True if detection should be suppressed

        """
        now_time = time.monotonic()

        # Hardcoded variables for stationary surpression, can be adjusted based on frame size, expectations etc.
        close_center_thresh = 100 # center closeness in pixels for where we trigger the algorithm for stationary surpression
        same_center_thresh = 10 # center closeness in pixels to consider it the same detection
        iou_replace_floor = 0.6 # IOU threshold, below which we confirm different vehicle
        iou_confirm_thresh = 0.925 # IOU threshold above which we confirm same detection, if same center thresh confirmed


        # Remove expired cache entries
        for cached_track_id in list(self.cached_detections.keys()):
            if now_time - self.cached_detections[cached_track_id]["timestamp"] > self.max_idle_seconds:
                del self.cached_detections[cached_track_id]

        # Check against cached detections

        center = self.get_center(bbox)

        candidates = []

        # Collect possible matches
        for cached_track_id, entry in self.cached_detections.items():
            center_distance = self.get_point_distance(center, entry["center"])
            if center_distance < close_center_thresh:
                iou = self.compute_iou(bbox, entry["bbox"])
                candidates.append((cached_track_id, entry, center_distance, iou))
        
        if not candidates:
            # No nearby detections -> insert fresh
            self._insert_cache(track_id, center, bbox, now_time)
            return False
        
        # Pick best match by IoU
        cached_track_id, entry, center_distance, iou = max(candidates, key=lambda x: x[3])

        # SAME TRACK
        # If the center is close to a cached detection, check if it's not the same track_id, for which we dont care, we are looking for imposters
        if track_id == cached_track_id:
            # its the same track id, just update the cache timestamp, box and center
            self.cached_detections[track_id]["timestamp"] = now_time
            self.cached_detections[track_id]["center"] = center
            self.cached_detections[track_id]["bbox"] = bbox
            return False # allow
        
        # DIFFERENT TRACK (Investigation)
        # its a different track id, but close to a cached detection, this is suspicious

        # --- Static detection confirmation ---

        if center_distance < same_center_thresh and iou > iou_confirm_thresh:
            # very close centers and very high IOU, we can be quite confident its the same detection, likely with a reset track id
            return True
        
        # --- Likely replacement (static detection ruled out) ---

        if iou < iou_replace_floor:
            # low IOU, likely a different detection entering that area
            del self.cached_detections[cached_track_id] # remove the cached detection, as it is likely a different vehicle now
            self._insert_cache(track_id, center, bbox, now_time)
            return False # allow the new detection

        # --- Uncertain case → allow the new, dont cache it and do NOT delete the cached (wait it out) ---

        return False # allow the detection, but we are uncertain, so we keep the cached one for now, to see if the situation clarifies in the next frames
        #(e.g. if the suspicious detection moves away or more detections appear etc.)              
                



    def perform_checks(self, track_id, bbox):
        # Perform checks on the bounding box

        #Sanity check

        x1, y1, x2, y2 = bbox

        assert x1 < x2, f"Invalid bbox: x1 ({x1}) should be less than x2 ({x2})"
        assert y1 < y2, f"Invalid bbox: y1 ({y1}) should be less than y2 ({y2})"

        # minimum crop size check
        if not self.check_min_crop_size(bbox):
            return False

        # Zone check (if zones are defined, otherwise skip)
        if self.use_zones:
            # Only move forward if zone changed and vehicle moved > pixel threshold
            if not self.check_crop_zones(track_id, bbox):
                return False
            
        # Stationary surpression check
        if self.use_stationary_surpression:
            if self.check_static_detection(track_id, bbox):
                return False
                    
        return True

    def verify_attention(self, bbox):
        
        center_point = self.get_center(bbox)

        attention = self.is_point_in_attention(center_point)

        return attention

    def zone_of_point(self, point):
        """
        Determine the zone index in which a given point lies.
        
        Args:
        - point (tuple): (x, y) coordinates of the point
        - zones (list): List of zone coordinates, each zone represented as ((x1, y1), (x2, y2))
        
        Returns:
        - zone_index (int): Index of the zone in which the point lies, or -1 if it's not in any zone
        """
        x, y = point
        
        # Iterate through each zone and check if the point lies within it
        for zone_index, zone in enumerate(self.zones):
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_index
        
        # If the point is not in any zone, return -1
        return -1




    def get_center(self,bbox):
        
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2])
        
        return bbox_center
    
    def get_point_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter = inter_w * inter_h

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = areaA + areaB - inter

        return inter / union if union > 0 else 0
    
    def _insert_cache(self, track_id, center, bbox, timestamp):
        self.cached_detections[track_id] = {
            "center": center,
            "bbox": bbox,
            "timestamp": timestamp,
        }

    def _purge_track_from_static_cache(self, track_id):
        # Remove any cache entries related to the given track_id
        if track_id in self.cached_detections:
            del self.cached_detections[track_id]


        
if __name__ == "__main__":
    print("Running CheckDetection sanity tests...")

    print("Testing static detection logic...")

    # --------------------------------------------------
    # Helper to build bbox from center points
    # --------------------------------------------------
    def make_bbox(cx, cy, w=100, h=100):
        return [
            cx - w // 2,
            cy - h // 2,
            cx + w // 2,
            cy + h // 2,
        ]

    # --------------------------------------------------
    # Create detector without zones for static tests
    # --------------------------------------------------
    cd = CheckDetection()
    cd.use_zones = False  # isolate static logic
    cd.max_idle_seconds = 2  # short expiry for testing

    # --------------------------------------------------
    # 1. First detection should be allowed + cached
    # --------------------------------------------------
    bbox1 = make_bbox(500, 500)
    assert cd.check_static_detection(1, bbox1) is False
    assert 1 in cd.cached_detections
    print("Test 1 passed: first detection allowed + cached")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 2. Same track_id update should be allowed
    # --------------------------------------------------
    bbox1_shift = make_bbox(502, 502)
    assert cd.check_static_detection(1, bbox1_shift) is False
    print("Test 2 passed: same track_id updated")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 3. Reset track_id at same location → should suppress
    # --------------------------------------------------
    bbox_same = make_bbox(504, 503)  # close to original
    suppressed = cd.check_static_detection(2, bbox_same)
    assert suppressed is True
    print("Test 3 passed: reset track suppressed")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 4. Distant case (low IoU)
    # --------------------------------------------------
    bbox_far = make_bbox(700, 700)
    surpressed = cd.check_static_detection(3, bbox_far)
    assert surpressed is False
    assert 3 in cd.cached_detections
    print("Test 4 passed: new allowed")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 5. Replacement case (low IoU)
    # --------------------------------------------------
    bbox_replace = make_bbox(570, 570)  # somewhat near original, but low IOU
    surpressed = cd.check_static_detection(4, bbox_replace)
    assert surpressed is False
    assert 4 in cd.cached_detections
    assert 1 not in cd.cached_detections  # old entry should be replaced
    print("Test 5 passed: replacement allowed")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 6. Uncertain case (mid IoU)
    # --------------------------------------------------
    bbox_mid = make_bbox(557, 557)  # near original
    surpressed = cd.check_static_detection(5, bbox_mid)
    assert surpressed is False
    assert 5 not in cd.cached_detections  # uncertain, so not cached
    print("Test 6 passed: uncertain case allowed")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 7. Expiry test
    # --------------------------------------------------
    time.sleep(3)
    bbox_new = make_bbox(900, 900)
    cd.check_static_detection(6, bbox_new)

    # old entries should be expired
    for tid in list(cd.cached_detections.keys()):
        assert tid == 6
    print("Test 7 passed: expiry works")
    print("Current cache entries:", cd.cached_detections.keys())

    # --------------------------------------------------
    # 8. Zone removal test
    # --------------------------------------------------

    print("Testing zone logic...")
    cd_z = CheckDetection(
        rows=2,
        cols=2,
        area_bottom_left=(0, 1000),
        area_top_right=(1000, 0),
    )

    bbox_zone = make_bbox(250, 750)
    cd_z.perform_checks(10, bbox_zone)
    assert 10 in cd_z.zone_of_detections
    print("Current cache entries:", cd_z.cached_detections.keys())
    
    # Move outside zone
    bbox_out = make_bbox(1500, 750)
    cd_z.check_crop_zones(10, bbox_out)
    assert 10 not in cd_z.zone_of_detections
    assert 10 not in cd_z.cached_detections
    print("Test 8 passed: zone exit purges cache")
    print("Current cache entries:", cd_z.cached_detections.keys())


    print("\nAll tests passed successfully.")

    # --------------------------------------------------
    # 9. Integration test
    # --------------------------------------------------
    print("Testing integration of zone and static logic...")

    cd_int = CheckDetection(
        rows=2,
        cols=2,
        area_bottom_left=(0, 1000),
        area_top_right=(1000, 0),
    )

    # Test that zone logic works with static detection logic
    bbox_zone = make_bbox(250, 750)
    cd_int.perform_checks(20, bbox_zone)
    assert 20 in cd_int.zone_of_detections
    assert 20 in cd_int.cached_detections

    # Create static duplicate inside zone → should be suppressed
    bbox_same = make_bbox(252, 751)
    assert cd_int.perform_checks(21, bbox_same) is False
    assert 21 not in cd_int.cached_detections

    # Test replacement IOU logic inside zone
    bbox_replace = make_bbox(320, 680)  # somewhat near original, but low IOU
    assert cd_int.perform_checks(22, bbox_replace) is True  # should be allowed as replacement
    assert 22 in cd_int.cached_detections
    assert 20 not in cd_int.cached_detections  # old entry should be replaced

    # Test switching zones
    bbox_zone2 = make_bbox(750, 250)
    cd_int.perform_checks(22, bbox_zone2)
    assert 22 in cd_int.zone_of_detections
    assert cd_int.zone_of_detections[22][0] == 1  # zone index should update to new zone

    # Test adding uncertain IOU case in same zone
    bbox_mid = make_bbox(740, 260)  # near original
    accepted = cd_int.perform_checks(23, bbox_mid)
    assert accepted is True  # should be allowed, but uncertain
    assert 23 not in cd_int.cached_detections  # uncertain, so not cached
    assert 22 in cd_int.cached_detections  # original should still be cached
    assert cd_int.zone_of_detections[23][0] == 1  # zone should still be the same
    assert cd_int.zone_of_detections[22][0] == 1  # original should still be in same zone

    # test exiting zone purges cache
    bbox_out = make_bbox(1500, 750)
    cd_int.perform_checks(22, bbox_out)
    assert 22 not in cd_int.zone_of_detections
    assert 22 not in cd_int.cached_detections
    print("Test 9 passed: integration of zone and static logic works")
    print("Current cache entries:", cd_int.cached_detections.keys())


        

