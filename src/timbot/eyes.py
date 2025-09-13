import cv2
import threading
import time
import tempfile
import os
from typing import List, Optional, Tuple
import mss  # For desktop screenshots

# Configuration
SCREENSHOT_INTERVAL = 2.0  # Take screenshot every 2 seconds
SCREENSHOT_WIDTH = 320  # Low resolution for faster processing
SCREENSHOT_HEIGHT = 240
IMAGE_FORMAT = ".jpg"
JPEG_QUALITY = 70  # Lower quality for smaller files
WEBCAM_FILENAME = "view_from_your_eyes_looking_at_user.jpg"
DESKTOP_FILENAME = "view_of_your_desktop_screen.jpg"

# Global state
_capture = None
_screenshot_thread = None
_current_webcam_screenshot = None  # Current webcam screenshot path
_current_desktop_screenshot = None  # Current desktop screenshot path
_screenshot_lock = threading.Lock()
_watching = False


def _get_webcam_capture():
    """Initialize webcam capture with best available camera."""
    for camera_index in range(4):
        cap = cv2.VideoCapture(
            camera_index, cv2.CAP_V4L2
        )  # Use V4L2 backend for better Linux compatibility
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


def _create_webcam_screenshot_path() -> str:
    """Create the webcam image file path."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, WEBCAM_FILENAME)

def _create_desktop_screenshot_path() -> str:
    """Create the desktop image file path."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, DESKTOP_FILENAME)

def _capture_desktop_screenshot() -> Optional[str]:
    """Capture desktop screenshot and return file path."""
    try:
        with mss.mss() as sct:
            # Capture the primary monitor
            monitor = sct.monitors[1]  # monitors[0] is all monitors combined, [1] is primary
            screenshot = sct.grab(monitor)
            
            # Convert to numpy array and then to OpenCV format
            import numpy as np
            img_array = np.array(screenshot)
            # Convert BGRA to BGR (remove alpha channel)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
            # Resize to low resolution for faster processing
            img_resized = cv2.resize(img_bgr, (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
            
            # Create screenshot file path
            screenshot_path = _create_desktop_screenshot_path()
            
            # Save as JPEG with compression
            success = cv2.imwrite(
                screenshot_path,
                img_resized,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            
            if success:
                return screenshot_path
            else:
                print("Failed to save desktop screenshot")
                return None
                
    except Exception as e:
        print(f"Error capturing desktop screenshot: {e}")
        return None


def _screenshot_worker():
    """Background worker thread for taking periodic screenshots from both webcam and desktop."""
    global _capture, _watching, _current_webcam_screenshot, _current_desktop_screenshot

    try:
        _capture = _get_webcam_capture()
        print(
            f"Taking webcam and desktop screenshots every {SCREENSHOT_INTERVAL}s at {SCREENSHOT_WIDTH}x{SCREENSHOT_HEIGHT}"
        )

        while _watching:
            try:
                webcam_success = False
                desktop_success = False
                
                # Capture webcam screenshot
                ret, frame = _capture.read()
                if ret and frame is not None and frame.size > 0:
                    # Resize to low resolution
                    frame_resized = cv2.resize(frame, (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))

                    # Create webcam screenshot file path
                    webcam_path = _create_webcam_screenshot_path()

                    # Save as JPEG with compression
                    webcam_success = cv2.imwrite(
                        webcam_path,
                        frame_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    
                    if webcam_success:
                        with _screenshot_lock:
                            _current_webcam_screenshot = webcam_path
                        print(f"📸 Webcam screenshot saved: {webcam_path}")
                    else:
                        print("Failed to save webcam screenshot")
                else:
                    print("Failed to capture webcam frame, skipping...")

                # Capture desktop screenshot
                desktop_path = _capture_desktop_screenshot()
                if desktop_path:
                    with _screenshot_lock:
                        _current_desktop_screenshot = desktop_path
                    print(f"🖥️ Desktop screenshot saved: {desktop_path}")
                    desktop_success = True
                else:
                    print("Failed to capture desktop screenshot")

                if not webcam_success and not desktop_success:
                    print("Both screenshot captures failed, retrying...")
                    time.sleep(1.0)
                    continue

                # Wait for next screenshot
                time.sleep(SCREENSHOT_INTERVAL)

            except cv2.error as e:
                print(f"OpenCV error during capture: {e}")
                time.sleep(2.0)  # Wait before retry
                continue
            except Exception as e:
                print(f"Screenshot capture error: {e}")
                time.sleep(2.0)
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
    global _watching, _current_webcam_screenshot, _current_desktop_screenshot

    if not _watching:
        return

    print("Stopping screenshot capture...")
    _watching = False

    # Wait for screenshot thread to finish
    if _screenshot_thread:
        _screenshot_thread.join(timeout=5.0)

    # Cleanup current screenshots
    with _screenshot_lock:
        if _current_webcam_screenshot and os.path.exists(_current_webcam_screenshot):
            try:
                os.unlink(_current_webcam_screenshot)
                _current_webcam_screenshot = None
            except OSError:
                pass
        
        if _current_desktop_screenshot and os.path.exists(_current_desktop_screenshot):
            try:
                os.unlink(_current_desktop_screenshot)
                _current_desktop_screenshot = None
            except OSError:
                pass


def get_recent_screenshots(_count: Optional[int] = None) -> List[str]:
    """
    Get all current screenshots (webcam and desktop combined).

    Args:
        _count: Ignored - kept for compatibility

    Returns:
        List containing current screenshot file paths, or empty list if none available
    """
    screenshots = []
    with _screenshot_lock:
        if _current_webcam_screenshot and os.path.exists(_current_webcam_screenshot):
            screenshots.append(_current_webcam_screenshot)
        if _current_desktop_screenshot and os.path.exists(_current_desktop_screenshot):
            screenshots.append(_current_desktop_screenshot)
    return screenshots

def get_webcam_and_desktop_screenshots() -> Tuple[Optional[str], Optional[str]]:
    """
    Get current webcam and desktop screenshots separately.
    
    Returns:
        Tuple of (webcam_path, desktop_path) where each can be None if not available
    """
    with _screenshot_lock:
        webcam_path = _current_webcam_screenshot if _current_webcam_screenshot and os.path.exists(_current_webcam_screenshot) else None
        desktop_path = _current_desktop_screenshot if _current_desktop_screenshot and os.path.exists(_current_desktop_screenshot) else None
        return (webcam_path, desktop_path)

def get_latest_screenshot() -> Optional[str]:
    """Get the current webcam screenshot (for backward compatibility)."""
    with _screenshot_lock:
        if _current_webcam_screenshot and os.path.exists(_current_webcam_screenshot):
            return _current_webcam_screenshot
    return None

def get_latest_desktop_screenshot() -> Optional[str]:
    """Get the current desktop screenshot."""
    with _screenshot_lock:
        if _current_desktop_screenshot and os.path.exists(_current_desktop_screenshot):
            return _current_desktop_screenshot
    return None

def is_watching() -> bool:
    """Check if screenshot capture is active."""
    return _watching

def get_screenshot_count() -> int:
    """Get current number of stored screenshots."""
    with _screenshot_lock:
        count = 0
        if _current_webcam_screenshot and os.path.exists(_current_webcam_screenshot):
            count += 1
        if _current_desktop_screenshot and os.path.exists(_current_desktop_screenshot):
            count += 1
        return count


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
