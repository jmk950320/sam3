
import numpy as np
import os
from typing import Optional
import sys, torch

print("Step 1: Importing libraries...")
try:
    import cv2
    print(" - cv2 imported successfully")
except Exception as e:
    print(f" - Error importing cv2: {e}")

# SAM3 import는 GUI 작업 완료 후에 수행 (충돌 방지)
# from sam3.model_builder import build_sam3_video_predictor

# 전역 변수로 박스 정보 저장
boxes = []  # [[x, y, w, h], ...]
box_labels = []  # [1, 1, ...] (1 for positive box)
drawing = False
start_point = None
current_box = None

def mouse_callback(event, x, y, flags, param):
    global boxes, box_labels, drawing, start_point, current_box
    try:
        frame = param['frame']
        display_frame = param['display_frame']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 박스 그리기 시작
            drawing = True
            start_point = (x, y)
            print(f"Box drawing started at: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # 임시 박스 표시
                temp_frame = display_frame.copy()
                cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("First Frame", temp_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 박스 그리기 완료
            if drawing:
                drawing = False
                end_point = (x, y)
                
                # 박스 좌표 계산 (x, y, w, h)
                x1, y1 = start_point
                x2, y2 = end_point
                box_x = min(x1, x2)
                box_y = min(y1, y2)
                box_w = abs(x2 - x1)
                box_h = abs(y2 - y1)
                
                if box_w > 5 and box_h > 5:  # 최소 크기 체크
                    boxes.append([box_x, box_y, box_w, box_h])
                    box_labels.append(1)  # 긍정 박스
                    
                    # 박스를 display_frame에 그리기
                    cv2.rectangle(display_frame, (box_x, box_y), 
                                (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Box {len(boxes)}", 
                              (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                    cv2.imshow("First Frame", display_frame)
                    
                    print(f"Box added: x={box_x}, y={box_y}, w={box_w}, h={box_h}")
                else:
                    print("Box too small, ignored")
                    
    except Exception as e:
        print(f"Error inside mouse_callback: {e}")

def show_first_frame_and_select_boxes(video_path):
    global boxes, box_labels, drawing, start_point
    boxes = []
    box_labels = []
    drawing = False
    start_point = None
    
    print("Step 2: Creating VideoCapture...")
    try:
        cap = cv2.VideoCapture(video_path)
        print(" - VideoCapture created")
    except Exception as e:
        print(f" - Error creating VideoCapture: {e}")
        return None, None

    print("Step 3: Checking if video is opened...")
    try:
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, None
        print(" - Video is open")
    except Exception as e:
        print(f" - Error checking isOpened: {e}")
        return None, None

    print("Step 4: Reading first frame...")
    try:
        ret, frame = cap.read()
        cap.release()
        print(f" - Frame read result: ret={ret}")
    except Exception as e:
        print(f" - Error reading frame: {e}")
        return None, None

    if ret:
        display_frame = frame.copy()
        
        print("Step 5: Creating named window and showing frame...")
        try:
            cv2.namedWindow("First Frame")
            print(" - Window created")
            
            cv2.imshow("First Frame", display_frame)
            print(" - First frame shown")
        except Exception as e:
            print(f"Error: Could not create/show window: {e}")
            return None, None
        
        print("Step 6: Setting mouse callback...")
        try:
            cv2.setMouseCallback("First Frame", mouse_callback, 
                               {'frame': frame, 'display_frame': display_frame})
            print(" - Mouse callback set")
        except Exception as e:
            print(f" - Error setting mouse callback: {e}")
        
        print("Instructions:")
        print(" - Click and drag to draw a bounding box")
        print(" - You can draw multiple boxes")
        print(" - Press 'q' or 'Enter' to finish selection")
        
        print("Step 8: Entering waitKey loop...")
        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13:
                    break
            print(" - Loop finished")
        except Exception as e:
            print(f" - Error in waitKey loop: {e}")
        
        print("Step 9: Destroying windows...")
        try:
            cv2.destroyAllWindows()
            print(" - Windows destroyed")
        except Exception as e:
            print(f" - Error destroying windows: {e}")
        
        return boxes, box_labels
    else:
        print("Error: Could not read the first frame.")
        return None, None

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


    
if __name__ == "__main__":
    video_file = "test_video.mp4"
    print(f"Starting process for {video_file}...")
   
    # GUI 작업 먼저 수행 (OpenCV 윈도우로 박스 선택)
    selected_boxes, selected_box_labels = show_first_frame_and_select_boxes(video_file)
    
    if selected_boxes:
        print("\nSelected Boxes:", selected_boxes)
        print("Selected Box Labels:", selected_box_labels)
        
        # GUI 작업 완료 후 SAM3 import (충돌 방지)
        print("\nStep 10: Importing SAM3 model builder...")
        try:
            from PIL import Image
            from sam3.model_builder import build_sam3_video_predictor
            from sam3.visualization_utils import (
                load_frame,
                prepare_masks_for_visualization,
                visualize_formatted_frame_output,
            )

            print(" - sam3.model_builder imported successfully")
            
        except Exception as e:
            print(f" - Error importing sam3.model_builder: {e}")


        print("Building SAM3 predictor...")
        # use all available GPUs on the machine
        gpus_to_use = range(torch.cuda.device_count())
        predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_file,
            )
        )
        session_id = response["session_id"]

        _ = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        frame_idx = 0

        if isinstance(video_file, str) and video_file.endswith(".mp4"):
            cap = cv2.VideoCapture(video_file)
            if cap is None:
                exit(-1)
            video_frames_for_vis = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        # SET IMG_WIDTH, IMG_HEIGHT
        sample_img = Image.fromarray(load_frame(video_frames_for_vis[0]))
        IMG_WIDTH, IMG_HEIGHT = sample_img.size

        # Convert boxes to relative coordinates (xywh format)
        boxes_xywh_rel = torch.tensor(
            abs_to_rel_coords(selected_boxes, IMG_WIDTH, IMG_HEIGHT, coord_type="box"),
            dtype=torch.float32,
        )
        box_labels_tensor = torch.tensor(selected_box_labels, dtype=torch.int32)
        
        print(f"\nBoxes (relative coords): {boxes_xywh_rel}")
        
        # Box prompts can be used as the first prompt (unlike point prompts)
        print("\nAdding box prompts...")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                boxes_xywh=boxes_xywh_rel,
                box_labels=box_labels_tensor,
            )
        )
        out = response["outputs"]
        print("Box prompts added successfully!")
        
        # Propagate across all frames
        print("\nPropagating segmentation across video frames...")
        outputs_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
            if response["frame_index"] % 50 == 0:
                print(f"  Processed frame {response['frame_index']}")
        
        print(f"\nPropagation completed! Processed {len(outputs_per_frame)} frames.")
        print("Segmentation complete!")

    else:
        print("No boxes selected.")
