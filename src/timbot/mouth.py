from piper import PiperVoice
import sounddevice as sd
import numpy as np
import threading

# Specify the path to your downloaded model and config files
model_path = "models/en_US-lessac-medium.onnx"
config_path = "models/en_US-lessac-medium.onnx.json"

# Load the Piper voice model
voice = PiperVoice.load(model_path, config_path)

# Global state for managing speech queue
_current_speech_thread = None
_speech_lock = threading.Lock()


def say(text: str, force_immediate: bool = False):
    """Non-blocking speech synthesis with queue management.
    
    Args:
        text: The text to speak
        force_immediate: If True, interrupts current speech. If False, waits for current speech to finish.
    """
    global _current_speech_thread
    
    def _play_audio():
        current_thread = threading.current_thread()
        try:
            audio_chunks = voice.synthesize(text)
            sample_rate = voice.config.sample_rate
            
            print("Playing audio...")
            for chunk in audio_chunks:
                # Check if we should stop (for force_immediate interruption)
                if getattr(current_thread, '_stop_requested', False):
                    print("Audio playback interrupted.")
                    return
                    
                audio_array = np.frombuffer(bytes(chunk.audio_int16_bytes), dtype=np.int16)
                sd.play(audio_array, samplerate=sample_rate)
                sd.wait()
            
            print("Audio playback finished.")
        except Exception as e:
            if not getattr(current_thread, '_stop_requested', False):
                print(f"Audio error: {e}")
            # Don't re-raise, just exit gracefully
    
    with _speech_lock:
        # Handle existing speech
        if _current_speech_thread and _current_speech_thread.is_alive():
            if force_immediate:
                # Just mark it to stop, don't call sd.stop()
                _current_speech_thread._stop_requested = True
                _current_speech_thread.join(timeout=0.5)  # Brief wait
            else:
                # Wait for current speech to finish
                _current_speech_thread.join()
        
        # Start new speech
        _current_speech_thread = threading.Thread(target=_play_audio, daemon=True)
        _current_speech_thread.start()
    
    return _current_speech_thread
