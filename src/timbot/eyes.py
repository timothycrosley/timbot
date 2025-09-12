import cv2
import threading
import time
import tempfile
import os
from typing import List, Optional

# Configuration
SCREENSHOT_INTERVAL = 10.0  # Take screenshot every 10 seconds
SCREENSHOT_WIDTH = 320    # Low resolution for faster processing
SCREENSHOT_HEIGHT = 240
IMAGE_FORMAT = '.jpg'
JPEG_QUALITY = 70        # Lower quality for smaller files
IMAGE_FILENAME = 'view_from_your_eyes_looking_at_user.jpg'

# Global state
_capture = None
_screenshot_thread = None
_current_screenshot = None  # Single current screenshot path
_screenshot_lock = threading.Lock()
_watching = False

def _get_webcam_capture():
    """Initialize webcam capture with best available camera."""
    for camera_index in range(4):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)  # Use V4L2 backend for better Linux compatibility
        if cap.isOpened():
            # Don't set resolution properties to avoid system conflicts
            # We'll capture at default resolution and resize afterward
            
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test if we can read a frame
            ret, _ = cap.read()
            if ret:
                print(f"Using camera at index {camera_index}")
                return cap
            cap.release()
    
    raise RuntimeError("No working webcam found")

def _create_screenshot_path() -> str:
    """Create the fixed image file path."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, IMAGE_FILENAME)

def _screenshot_worker():
    """Background worker thread for taking periodic screenshots."""
    global _capture, _watching, _current_screenshot
    
    try:
        _capture = _get_webcam_capture()
        print(f"Taking screenshots every {SCREENSHOT_INTERVAL}s at {SCREENSHOT_WIDTH}x{SCREENSHOT_HEIGHT}")
        
        while _watching:
            try:
                ret, frame = _capture.read()
                if not ret:
                    print("Failed to capture frame, retrying...")
                    time.sleep(1.0)  # Short wait before retry
                    continue
                
                # Validate frame dimensions
                if frame is None or frame.size == 0:
                    print("Invalid frame received, skipping...")
                    time.sleep(1.0)
                    continue
                
                # Resize to low resolution
                frame_resized = cv2.resize(frame, (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
                
                # Create screenshot file path
                screenshot_path = _create_screenshot_path()
                
                # Save as JPEG with compression
                success = cv2.imwrite(
                    screenshot_path, 
                    frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                
                if success:
                    # Update current screenshot (thread-safe)
                    with _screenshot_lock:
                        _current_screenshot = screenshot_path
                        print(f"📸 Screenshot saved: {screenshot_path}")
                else:
                    print("Failed to save screenshot")
                
                # Wait for next screenshot
                time.sleep(SCREENSHOT_INTERVAL)
                
            except cv2.error as e:
                print(f"OpenCV error during capture: {e}")
                time.sleep(2.0)  # Wait before retry
                continue
                
        # Cleanup
        if _capture:
            _capture.release()
            cv2.destroyAllWindows()  # Clean up any OpenCV windows
            _capture = None
        
        print("Screenshot capture stopped")
        
    except Exception as e:
        print(f"Screenshot capture error: {e}")
        _watching = False
        # Ensure cleanup even on error
        if _capture:
            try:
                _capture.release()
                cv2.destroyAllWindows()
            except Exception:
                pass
            _capture = None

def start_watching():
    """Start taking periodic screenshots."""
    global _watching, _screenshot_thread
    
    if _watching:
        print("Already taking screenshots")
        return
    
    print("Starting screenshot capture...")
    _watching = True
    _screenshot_thread = threading.Thread(target=_screenshot_worker, daemon=True)
    _screenshot_thread.start()

def stop_watching():
    """Stop taking screenshots."""
    global _watching, _current_screenshot
    
    if not _watching:
        return
    
    print("Stopping screenshot capture...")
    _watching = False
    
    # Wait for screenshot thread to finish
    if _screenshot_thread:
        _screenshot_thread.join(timeout=5.0)
    
    # Cleanup current screenshot
    with _screenshot_lock:
        if _current_screenshot and os.path.exists(_current_screenshot):
            try:
                os.unlink(_current_screenshot)
                _current_screenshot = None
            except OSError:
                pass

def get_recent_screenshots(_count: Optional[int] = None) -> List[str]:
    """
    Get the current screenshot.
    
    Args:
        _count: Ignored - kept for compatibility
    
    Returns:
        List containing current screenshot file path, or empty list if none available
    """
    with _screenshot_lock:
        if _current_screenshot and os.path.exists(_current_screenshot):
            return [_current_screenshot]
        return []

def get_latest_screenshot() -> Optional[str]:
    """Get the current screenshot."""
    screenshots = get_recent_screenshots()
    return screenshots[0] if screenshots else None

def is_watching() -> bool:
    """Check if screenshot capture is active."""
    return _watching

def get_screenshot_count() -> int:
    """Get current number of stored screenshots."""
    with _screenshot_lock:
        return 1 if _current_screenshot and os.path.exists(_current_screenshot) else 0

# Convenience functions for compatibility
def start_recording():
    """Alias for start_watching() for compatibility."""
    start_watching()

def stop_recording():
    """Alias for stop_watching() for compatibility."""
    stop_watching()
    return None  # No final video file

def yield_current_video():
    """Return current screenshot instead of video."""
    return get_recent_screenshots()

def cleanup_screenshot_files(file_paths: List[str]):
    """Delete screenshot files (use when done processing them)."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"Cleaned up screenshot: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"Failed to cleanup screenshot {file_path}: {e}")

def cleanup_video_file(file_path):
    """Legacy function - now handles screenshot cleanup."""
    if isinstance(file_path, list):
        cleanup_screenshot_files(file_path)
    elif file_path and os.path.exists(file_path):
        cleanup_screenshot_files([file_path])