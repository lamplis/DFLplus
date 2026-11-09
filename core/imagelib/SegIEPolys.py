import numpy as np
import cv2
from enum import IntEnum


class SegIEPolyType(IntEnum):
    EXCLUDE = 0
    INCLUDE = 1
    


class SegIEPoly():
    def __init__(self, type=None, pts=None, **kwargs):
        self.type = type
        
        if pts is None:
            pts = np.empty( (0,2), dtype=np.float32 )
        else:
            pts = np.float32(pts)
        self.pts = pts
        self.n_max = self.n = len(pts)

    def dump(self):
        return {'type': int(self.type),
                'pts' : self.get_pts(),
               }
    
    def identical(self, b):
        if self.n != b.n:
            return False            
        return (self.pts[0:self.n] == b.pts[0:b.n]).all()        
        
    def get_type(self):
        return self.type

    def add_pt(self, x, y):
        self.pts = np.append(self.pts[0:self.n], [ ( float(x), float(y) ) ], axis=0).astype(np.float32)
        self.n_max = self.n = self.n + 1

    def undo(self):
        self.n = max(0, self.n-1)
        return self.n

    def redo(self):
        self.n = min(len(self.pts), self.n+1)
        return self.n

    def redo_clip(self):
        self.pts = self.pts[0:self.n]
        self.n_max = self.n

    def insert_pt(self, n, pt):
        if n < 0 or n > self.n:
            raise ValueError("insert_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], pt[None,...].astype(np.float32), self.pts[n:]), axis=0)
        self.n_max = self.n = self.n+1
        
    def remove_pt(self, n):
        if n < 0 or n >= self.n:
            raise ValueError("remove_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], self.pts[n+1:]), axis=0)
        self.n_max = self.n = self.n-1
        
    def get_last_point(self):
        return self.pts[self.n-1].copy()

    def get_pts(self):
        return self.pts[0:self.n].copy()
        
    def get_pts_count(self):
        return self.n

    def set_point(self, id, pt):
        self.pts[id] = pt
        
    def set_points(self, pts):
        self.pts = np.array(pts)
        self.n_max = self.n = len(pts)
        
    def mult_points(self, val):
        self.pts *= val       
    
        

class SegIEPolys():
    def __init__(self):
        self.polys = []

    def identical(self, b):
        polys_len = len(self.polys)
        o_polys_len = len(b.polys)
        if polys_len != o_polys_len:
            return False
        
        return all ([ a_poly.identical(b_poly) for a_poly, b_poly in zip(self.polys, b.polys) ])
        
    def add_poly(self, ie_poly_type):       
        poly = SegIEPoly(ie_poly_type)
        self.polys.append (poly)
        return poly

    def remove_poly(self, poly):
        if poly in self.polys:
            self.polys.remove(poly)

    def has_polys(self):
        return len(self.polys) != 0
        
    def get_poly(self, id):
        return self.polys[id]

    def get_polys(self):
        return self.polys
        
    def get_pts_count(self):
        return sum([poly.get_pts_count() for poly in self.polys])
        
    def sort(self):
        poly_by_type = { SegIEPolyType.EXCLUDE : [], SegIEPolyType.INCLUDE : [] }

        for poly in self.polys:
            poly_by_type[poly.type].append(poly)
            
        self.polys = poly_by_type[SegIEPolyType.INCLUDE] + poly_by_type[SegIEPolyType.EXCLUDE]

    def __iter__(self):
        for poly in self.polys:
            yield poly

    def overlay_mask(self, mask):
        h,w,c = mask.shape
        white = (1,)*c
        black = (0,)*c
        for poly in self.polys:
            pts = poly.get_pts().astype(np.int32)
            if len(pts) != 0:
                cv2.fillPoly(mask, [pts], white if poly.type == SegIEPolyType.INCLUDE else black )

    def dump(self):
        return {'polys' : [ poly.dump() for poly in self.polys ] }
    
    def mult_points(self, val):
        for poly in self.polys:
            poly.mult_points(val)

    @staticmethod
    def _hole_is_mouth(contour, landmarks):
        inner = np.asarray(landmarks[60:68], np.float32)
        for point in inner:
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                return True

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return False
        centroid = (float(moments['m10'] / moments['m00']), float(moments['m01'] / moments['m00']))
        hull = cv2.convexHull(np.round(inner).astype(np.int32))
        return cv2.pointPolygonTest(hull, centroid, False) >= 0

    @staticmethod
    def _resample_closed(contour, step_px):
        points = np.asarray(contour, np.float32).reshape(-1, 2)
        if len(points) == 0:
            return points
        step = float(step_px)
        if step <= 0:
            return points
        kept = [points[0]]
        last = points[0]
        for point in points[1:]:
            if float(np.linalg.norm(point - last)) >= step:
                kept.append(point)
                last = point
        return np.stack(kept, axis=0)

    @staticmethod
    def mask_to_ie_polys(mask, min_area_ratio=0.002, epsilon_ratio=0.0015, landmarks=None, step_px=None):
        ie_polys = SegIEPolys()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.size == 0:
            return ie_polys

        if mask.dtype == np.uint8:
            binary = (mask >= 128).astype(np.uint8)
        else:
            binary = (mask >= 0.5).astype(np.uint8)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return ie_polys

        hierarchy = hierarchy[0]
        height, width = binary.shape[:2]
        min_area = float(min_area_ratio) * height * width
        mouth_landmarks = None
        if landmarks is not None:
            landmarks = np.asarray(landmarks, np.float32)
            if landmarks.shape == (68, 2):
                mouth_landmarks = landmarks

        for index, contour in enumerate(contours):
            if abs(cv2.contourArea(contour)) < min_area:
                continue
            parent = int(hierarchy[index][3])
            if parent >= 0 and mouth_landmarks is not None and SegIEPolys._hole_is_mouth(contour, mouth_landmarks):
                continue
            if step_px is None:
                epsilon = float(epsilon_ratio) * cv2.arcLength(contour, True)
                points = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :]
            else:
                points = SegIEPolys._resample_closed(contour, step_px)
            if len(points) < 3:
                continue
            poly_type = SegIEPolyType.INCLUDE if parent < 0 else SegIEPolyType.EXCLUDE
            poly = ie_polys.add_poly(poly_type)
            for point in points:
                poly.add_pt(float(point[0]), float(point[1]))

        ie_polys.sort()
        return ie_polys

    @staticmethod
    def _forehead_from_brows(landmarks):
        # The 68-point set has no hairline. Continue the chin-to-brow direction
        # by about half that distance, which is the usual brow-to-hairline ratio.
        brows = landmarks[26:16:-1]
        brow_mid = np.mean(landmarks[17:27], axis=0)
        direction = brow_mid - landmarks[8]
        length = float(np.linalg.norm(direction))
        if length < 1e-3:
            return brows.copy()
        up = direction / length
        lift = 0.5 * length
        count = len(brows)
        raised = np.empty_like(brows)
        for index, point in enumerate(brows):
            weight = float(np.sin(np.pi * index / (count - 1)))
            raised[index] = point + up * lift * weight
        return raised

    @staticmethod
    def landmarks_to_ie_polys(landmarks):
        landmarks = np.asarray(landmarks, np.float32)
        if landmarks.shape != (68, 2):
            raise ValueError(f'landmarks must have shape (68, 2), got {landmarks.shape}')

        contour = np.vstack((landmarks[0:17], SegIEPolys._forehead_from_brows(landmarks)))
        ie_polys = SegIEPolys()
        poly = ie_polys.add_poly(SegIEPolyType.INCLUDE)
        for x, y in contour:
            poly.add_pt(float(x), float(y))
        ie_polys.sort()
        return ie_polys

    @staticmethod
    def load(data=None):
        ie_polys = SegIEPolys()
        if data is not None:
            if isinstance(data, list):
                # Backward comp
                ie_polys.polys = [ SegIEPoly(type=type, pts=pts) for (type, pts) in data ]
            elif isinstance(data, dict):                
                ie_polys.polys = [ SegIEPoly(**poly_cfg) for poly_cfg in data['polys'] ]
                
        ie_polys.sort()   
                
        return ie_polys