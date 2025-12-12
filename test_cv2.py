#!/usr/bin/env python3
"""
비디오 파일에서 첫 번째 프레임을 읽어서 이미지로 표시하는 스크립트
Segmentation Fault를 피하기 위한 안전장치 포함
"""

import cv2
import sys
import os
import faulthandler

# Segfault 발생 시 traceback 출력 (디버깅용)
faulthandler.enable()

def show_first_frame(video_path):
    """
    비디오 파일의 첫 번째 프레임을 읽어서 화면에 표시
    
    Args:
        video_path: 비디오 파일 경로
        
    Returns:
        frame: 성공 시 첫 번째 프레임 (numpy array), 실패 시 None
    """
    
    # 1. 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    print(f"📹 Opening video: {video_path}")
    
    # 2. VideoCapture 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return None
    
    # 3. 비디오 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened successfully")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    
    # 4. 첫 번째 프레임 읽기
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read the first frame")
        return None
    
    print(f"✅ First frame read successfully (shape: {frame.shape})")
    
    # 5. GUI 환경 확인
    display = os.environ.get('DISPLAY', '')
    if not display:
        print("⚠️  Warning: No DISPLAY environment variable set")
        print("   Running in headless mode - saving frame to file instead")
        output_path = "first_frame.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Frame saved to: {output_path}")
        return frame
    
    # 6. 윈도우 생성 및 프레임 표시 (GUI 환경에서만)
    try:
        window_name = "First Frame"
        
        # 윈도우 생성 (크기 조절 가능)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 프레임 표시
        cv2.imshow(window_name, frame)
        
        print("\n" + "="*50)
        print("🖼️  First frame is displayed")
        print("="*50)
        print("Instructions:")
        print("  - Press any key to close the window")
        print("  - Press 's' to save the frame to 'first_frame.jpg'")
        print("="*50)
        
        # 키 입력 대기
        key = cv2.waitKey(0) & 0xFF
        
        # 's' 키를 누르면 이미지 저장
        if key == ord('s'):
            output_path = "first_frame.jpg"
            cv2.imwrite(output_path, frame)
            print(f"💾 Frame saved to: {output_path}")
        
        # 윈도우 닫기
        cv2.destroyAllWindows()
        
        # waitKey 후 약간의 대기 (윈도우가 완전히 닫히도록)
        cv2.waitKey(1)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not display frame (GUI error): {e}")
        print("   Saving frame to file instead...")
        output_path = "first_frame.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Frame saved to: {output_path}")
    
    return frame


def main():
    """메인 함수"""
    
    # 비디오 파일 경로 (명령줄 인자 또는 기본값)
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "test_video.mp4"  # 기본 비디오 파일
    
    print("\n" + "="*60)
    print("🎬 Video First Frame Viewer")
    print("="*60 + "\n")
    
    # 첫 번째 프레임 표시
    frame = show_first_frame(video_path)
    
    if frame is not None:
        print("\n✅ Process completed successfully")
    else:
        print("\n❌ Process failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
