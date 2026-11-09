import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from DFLIMG import *
from facelib import XSegNet, LandmarksProcessor, FaceType
import pickle

def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.')
        
    face_type = None
    
    model_dat = model_path / 'XSeg_data.dat'
    if model_dat.exists():
        dat = pickle.loads( model_dat.read_bytes() )
        dat_options = dat.get('options', None)
        if dat_options is not None:
            face_type = dat_options.get('face_type', None)
        
        
        
    if face_type is None:
        face_type = io.input_str ("XSeg model face type", 'same', ['h','mf','f','wf','head','same'], help_message="Specify face type of trained XSeg model. For example if XSeg model trained as WF, but faceset is HEAD, specify WF to apply xseg only on WF part of HEAD. Default is 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'Applying trained XSeg model to {input_path.name}/ folder.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)
        
    
    
    xseg = XSegNet(name='XSeg', 
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    xseg_res = xseg.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        img_face_type = FaceType.fromString( dflimg.get_face_type() )
        if face_type is not None and img_face_type != face_type:
            lmrks = dflimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( np.asarray(g_p2, dtype=np.float32), np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != xseg_res:
                img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]            
    
        mask = xseg.extract(img)
        
        if face_type is not None and img_face_type != face_type:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()


FACE_PARSER_POLY_STEP_PX = 2.0
EYE_TO_JAW_MIN = 0.20
EYE_ROLL_MAX_DEG = 25.0
EYE_DIST_MIN = 120.0
SOFT_LAPLACIAN_MAX = 16.0
FAR_FACE_DIST = 0.55
FAR_FACE_MIN_AREA = 8000.0
STRAY_INCLUDE_MIN_AREA = 8000.0
STRAY_OUTSIDE_PX = 10.0
NOSE_HOLE_MIN_AREA = 4000.0
NOSE_HOLE_MAX_AREA = 20000.0
NOSE_HOLE_DIST = 0.25
JAW_OUTSIDE_PX = 220.0


def _poly_area(pts):
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def _poly_centroid(pts):
    pts = np.asarray(pts, np.float32)
    moments = cv2.moments(pts)
    if abs(moments['m00']) < 1e-3:
        return pts.mean(0)
    return np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], np.float32)


def _poly_covers_mask(pts, binary):
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return False
    height, width = binary.shape[:2]
    canvas = np.zeros((height, width), np.uint8)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return bool(np.any(canvas.astype(bool) & np.asarray(binary, bool)))


def invalid_aligned_reason(dflimg):
    if dflimg is None or not dflimg.has_data():
        return 'no-dfl'
    landmarks = dflimg.get_dict().get('landmarks')
    if landmarks is None:
        return 'bad-landmarks'
    landmarks = np.asarray(landmarks, np.float32)
    if landmarks.shape != (68, 2):
        return 'bad-landmarks'
    left_eye = landmarks[36:42].mean(0)
    right_eye = landmarks[42:48].mean(0)
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    jaw_width = float(np.linalg.norm(landmarks[16] - landmarks[0]))
    if jaw_width < 1e-6:
        return 'bad-landmarks'
    if eye_dist < EYE_DIST_MIN:
        return f'small-eyes {eye_dist:.0f}'
    ratio = eye_dist / jaw_width
    if ratio < EYE_TO_JAW_MIN:
        return f'collapsed-eyes {ratio:.3f}'
    roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    if abs(roll) >= EYE_ROLL_MAX_DEG:
        return f'tilted-eyes {roll:.1f}'
    return None


def invalid_image_reason(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < SOFT_LAPLACIAN_MAX:
        return f'soft-image {variance:.1f}'
    return None


def invalid_parsing(parsing, landmarks):
    from facelib.FaceParser import FACE_PART_CLASSES, MOUTH_CLASS

    landmarks = np.asarray(landmarks, np.float32)
    if landmarks.shape != (68, 2):
        return None
    jaw_width = float(np.linalg.norm(landmarks[16] - landmarks[0]))
    if jaw_width < 1e-6:
        return None

    parsing = np.asarray(parsing)
    face = np.isin(parsing, list(FACE_PART_CLASSES) + [MOUTH_CLASS])
    height, width = face.shape[:2]
    dots = np.zeros((height, width), np.uint8)
    for point in landmarks:
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        cv2.circle(dots, (x, y), 2, 1, -1)
    distance = cv2.distanceTransform((1 - dots).astype(np.uint8), cv2.DIST_L2, 3)
    far = int((face & (distance > FAR_FACE_DIST * jaw_width)).sum())
    if far >= FAR_FACE_MIN_AREA:
        return f'far-face {far:.0f}'
    return None


def invalid_parser_polys(ie_polys, landmarks):
    landmarks = np.asarray(landmarks, np.float32)
    if landmarks.shape != (68, 2):
        return None
    jaw_width = float(np.linalg.norm(landmarks[16] - landmarks[0]))
    if jaw_width < 1e-6:
        return None

    from core.imagelib.SegIEPolys import SegIEPolyType

    x0 = float(landmarks[:, 0].min())
    x1 = float(landmarks[:, 0].max())
    y1 = float(landmarks[:, 1].max())
    nose = landmarks[27:36]
    includes = []

    for poly in ie_polys.polys:
        pts = poly.get_pts()
        area = _poly_area(pts)
        if int(poly.get_type()) == int(SegIEPolyType.INCLUDE):
            if len(pts) < 3:
                continue
            includes.append(pts)
            contains_landmark = any(cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0 for pt in landmarks)
            if contains_landmark:
                continue
            if area < STRAY_INCLUDE_MIN_AREA:
                continue
            centroid = _poly_centroid(pts)
            outside = (centroid[0] < x0 - STRAY_OUTSIDE_PX or
                       centroid[0] > x1 + STRAY_OUTSIDE_PX or
                       centroid[1] > y1 + STRAY_OUTSIDE_PX)
            if outside:
                return f'second-face {area:.0f}'
        elif int(poly.get_type()) == int(SegIEPolyType.EXCLUDE):
            if area < NOSE_HOLE_MIN_AREA or area >= NOSE_HOLE_MAX_AREA:
                continue
            centroid = _poly_centroid(pts)
            dist = float(np.linalg.norm(nose - centroid, axis=1).min()) / jaw_width
            if dist <= NOSE_HOLE_DIST:
                return f'nose-hole {area:.0f}'

    jaw_out = 0.0
    for index in (0, 16):
        if not includes:
            break
        pt = landmarks[index]
        closest = max(cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), True) for pts in includes)
        if closest < 0:
            jaw_out = max(jaw_out, float(-closest))
    if jaw_out >= JAW_OUTSIDE_PX:
        return f'jaw-outside {jaw_out:.0f}'
    return None


def parsing_to_ie_polys(parsing, inner_mouth, landmarks):
    from core.imagelib.SegIEPolys import SegIEPolys, SegIEPolyType
    from facelib.FaceParser import FACE_PART_CLASSES, HAIR_CLASS, MOUTH_CLASS

    parsing = np.asarray(parsing)
    classes = FACE_PART_CLASSES + ((MOUTH_CLASS,) if inner_mouth else ())
    face = np.isin(parsing, classes)
    hair = parsing == HAIR_CLASS
    union = (face | hair).astype(np.float32)
    ie_polys = SegIEPolys.mask_to_ie_polys(union, step_px=FACE_PARSER_POLY_STEP_PX, landmarks=landmarks)

    kept = []
    for poly in ie_polys.polys:
        if int(poly.get_type()) != int(SegIEPolyType.INCLUDE):
            kept.append(poly)
            continue
        if _poly_covers_mask(poly.get_pts(), face):
            kept.append(poly)
    ie_polys.polys = kept

    hair_polys = SegIEPolys.mask_to_ie_polys(hair.astype(np.float32), step_px=FACE_PARSER_POLY_STEP_PX)
    for poly in hair_polys.polys:
        if int(poly.get_type()) != int(SegIEPolyType.INCLUDE):
            continue
        poly.type = SegIEPolyType.EXCLUDE
        ie_polys.polys.append(poly)
    ie_polys.sort()
    return ie_polys


def apply_face_parser(input_path, model_file, inner_mouth):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if not model_file.exists():
        raise ValueError(f'{model_file} not found. Please ensure it exists.')

    from facelib.FaceParser import FaceParser
    parser = FaceParser(model_file)

    failed_dir = input_path.parent / 'aligned_failed'
    failed_dir.mkdir(parents=True, exist_ok=True)

    def move_failed(filepath, reason):
        dest = failed_dir / filepath.name
        if dest.exists():
            io.log_info(f'{filepath.name} not moved, {dest.name} already exists ({reason})')
        else:
            filepath.rename(dest)
            io.log_info(f'{filepath.name} -> aligned_failed ({reason})')

    io.log_info(f'Applying face parser polygons to {input_path.name}/ folder.')
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        reason = invalid_aligned_reason(dflimg)
        if reason is not None:
            move_failed(filepath, reason)
            continue

        img = cv2_imread(filepath)
        if img is None:
            io.log_info(f'{filepath} is not an image')
            continue

        image_reason = invalid_image_reason(img)
        if image_reason is not None:
            move_failed(filepath, image_reason)
            continue

        parsing = parser.predict_parsing(img)
        landmarks = dflimg.get_dict().get('landmarks')
        ie_polys = parsing_to_ie_polys(parsing, inner_mouth, landmarks)
        parse_reason = invalid_parsing(parsing, landmarks)
        poly_reason = invalid_parser_polys(ie_polys, landmarks)
        dflimg.set_seg_ie_polys(ie_polys)
        dflimg.save()
        if parse_reason is not None:
            move_failed(filepath, parse_reason)
        elif poly_reason is not None:
            move_failed(filepath, poly_reason)


def polys_to_xseg_mask(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    io.log_info(f'Baking polygons to XSeg masks in {input_path.name}/ folder.')
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        ie_polys = dflimg.get_seg_ie_polys()
        if ie_polys.get_pts_count() == 0:
            io.log_info(f'{filepath} has no polygons')
            continue

        img = cv2_imread(filepath)
        if img is None:
            io.log_info(f'{filepath} is not an image')
            continue

        height, width = img.shape[:2]
        mask = np.zeros((height, width, 1), np.float32)
        ie_polys.overlay_mask(mask)
        dflimg.set_xseg_mask(mask)
        dflimg.save()


def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing XSeg polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\nDelete original files?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
            
    
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')