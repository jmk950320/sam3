import os
import cv2
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import load_frame, prepare_masks_for_visualization
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Global variables for mouse interaction
points = []
labels = []
current_frame = None
overlay_mask = None
predictor = None
session_id = None
IMG_WIDTH = 0
IMG_HEIGHT = 0
video_path = "test_video.mp4" # Default, can be changed
frame_idx = 0
obj_id = 1

def abs_to_rel_coords(coords, width, height):
    """Convert absolute coordinates to relative coordinates (0-1 range)"""
    return [[x / width, y / height] for x, y in coords]

def mouse_callback(event, x, y, flags, param):
    global points, labels, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click: Positive point (Green)
        points.append([x, y])
        labels.append(1)
        print(f"Added positive point at ({x}, {y})")
        update_display()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: Negative point (Red)
        points.append([x, y])
        labels.append(0)
        print(f"Added negative point at ({x}, {y})")
        update_display()

def update_display():
    global current_frame, overlay_mask, points, labels
    
    display_img = current_frame.copy()
    
    # Draw points
    for pt, label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255) # Green for positive, Red for negative
        cv2.circle(display_img, tuple(pt), 5, color, -1)
    
    # Draw mask if available
    if overlay_mask is not None:
        # Resize mask to image size if needed (though usually it matches)
        # Create a colored mask
        mask_color = np.zeros_like(display_img)
        mask_color[:, :] = (0, 255, 0) # Green mask
        
        # Apply alpha blending
        alpha = 0.5
        mask_indices = overlay_mask > 0
        display_img[mask_indices] = cv2.addWeighted(display_img[mask_indices], 1 - alpha, mask_color[mask_indices], alpha, 0)

    cv2.imshow("SAM3 Interactive", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))

def run_inference():
    global predictor, session_id, points, labels, overlay_mask, IMG_WIDTH, IMG_HEIGHT, frame_idx, obj_id
    
    if not points:
        print("No points to process.")
        return

    print("Running inference...")
    
    # Convert to tensors
    points_tensor = torch.tensor(
        abs_to_rel_coords(points, IMG_WIDTH, IMG_HEIGHT),
        dtype=torch.float32,
    )
    labels_tensor = torch.tensor(labels, dtype=torch.int32)
    
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            points=points_tensor,
            point_labels=labels_tensor,
            obj_id=obj_id,
        )
    )
    
    # Process output for visualization
    # The output structure depends on the model, usually it's a list of masks or similar
    # Based on the notebook: out = response["outputs"]
    out = response["outputs"]
    
    # We need to extract the mask for the current object
    # The notebook uses prepare_masks_for_visualization, let's see what it returns
    # It seems to return a dictionary or list.
    # For simplicity, let's assume 'out' contains the mask data.
    # In SAM3, 'out' usually contains 'pred_masks' which are logits or probabilities.
    
    # Let's try to use the provided visualization util if possible, or manually extract.
    # Looking at notebook: visualize_formatted_frame_output takes outputs_list.
    # But we want to show it in OpenCV window.
    
    # Let's inspect 'out' structure from the notebook context or assume standard SAM output.
    # Usually: out['pred_masks'] -> [C, H, W] or similar.
    
    # For now, let's try to parse 'out'. 
    # If 'out' is a list of objects, we find ours.
    
    # Helper to get mask from response
    # In the notebook: prepare_masks_for_visualization({frame_idx: out})
    # This might be too complex for a simple cv2 overlay.
    
    # Let's look at what 'out' is.
    # It is likely a list of dictionaries, one per object, or a tensor.
    # If we look at the notebook, `out` is passed to `prepare_masks_for_visualization`.
    
    # Let's assume we can get a binary mask.
    # If we can't easily use the complex vis utils, we might need to rely on the fact that
    # we are prompting for a specific obj_id.
    
    # Let's try to find the mask for our obj_id in 'out'.
    # If 'out' is a list, we iterate.
    found_mask = None
    for obj_data in out:
        if obj_data['obj_id'] == obj_id:
            found_mask = obj_data['mask'] # This is likely a RLE or a tensor
            break
            
    if found_mask is None:
        # Maybe the structure is different.
        # Let's try to just use the first item if it exists.
        if len(out) > 0:
             # It might be a tensor directly if it's raw output? No, usually a list of dicts.
             # Let's assume it's a tensor for now if the list logic fails, or check the notebook again.
             # Notebook: out = response["outputs"]
             pass
    
    # To be safe, let's use the visualization utility to get a clean mask image?
    # No, that uses matplotlib.
    
    # Let's try to interpret the mask.
    # If it's a tensor, we convert to numpy.
    # If it's RLE, we decode.
    
    # Let's assume for this script we just want to see *something*.
    # We will try to extract the mask from the response.
    # In SAM3, response["outputs"] is typically a list of dicts:
    # [{'obj_id': 2, 'mask': Tensor(...), ...}]
    
    for item in out:
        if item['obj_id'] == obj_id:
            mask = item['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Mask might be [1, H, W] or [H, W]
            if mask.ndim == 3:
                mask = mask[0]
            
            # Binarize
            overlay_mask = (mask > 0.0).astype(np.uint8)
            break
            
    update_display()

def main():
    global predictor, session_id, current_frame, IMG_WIDTH, IMG_HEIGHT, video_path
    
    # Initialize predictor
    print("Building SAM3 predictor...")
    predictor = build_sam3_video_predictor()
   # print("Done.")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print("video not found")
        exit(-1)

    print(f"Using video: {video_path}")

    # Load first frame
    if os.path.isdir(video_path):
        frames = sorted(os.listdir(video_path))
        frame_path = os.path.join(video_path, frames[0])
        current_frame = load_frame(frame_path)
    else:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to read video.")
            return
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    IMG_HEIGHT, IMG_WIDTH = current_frame.shape[:2]

    # Start session
    print("Starting session...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    print("Session started. ID:", session_id)
    
    # Setup Window
    print("Creating window...")
    cv2.namedWindow("SAM3 Interactive", cv2.WINDOW_NORMAL)
    print("Window created.")
    
    print("Setting mouse callback...")
    cv2.setMouseCallback("SAM3 Interactive", mouse_callback)
    print("Callback set.")
    
    print("\nControls:")
    print("  Left Click  : Add positive point")
    print("  Right Click : Add negative point")
    print("  Space       : Run inference")
    print("  r           : Reset points")
    print("  q           : Quit")
    
    print("Updating display...")
    update_display()
    print("Display updated.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            run_inference()
        elif key == ord('r'):
            points = []
            labels = []
            overlay_mask = None
            print("Reset points.")
            update_display()
            
    # Cleanup
    print("Closing session...")
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
