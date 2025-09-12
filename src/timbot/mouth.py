from piper import PiperVoice
import sounddevice as sd
import numpy as np
import threading
import queue
import re

# Specify the path to your downloaded model and config files
model_path = "models/en_US-lessac-medium.onnx"
config_path = "models/en_US-lessac-medium.onnx.json"

# Load the Piper voice model
voice = PiperVoice.load(model_path, config_path)

# Global state for managing speech queue
_current_speech_thread = None
_speech_lock = threading.Lock()
_is_speaking = False


def say(text: str, force_immediate: bool = False):
    """Non-blocking speech synthesis with queue management.

    Args:
        text: The text to speak
        force_immediate: If True, interrupts current speech. If False, waits for current speech to finish.
    """
    global _current_speech_thread

    def _play_audio():
        global _is_speaking
        current_thread = threading.current_thread()
        try:
            _is_speaking = True
            audio_chunks = voice.synthesize(text)
            sample_rate = voice.config.sample_rate

            print("Playing audio...")
            for chunk in audio_chunks:
                # Check if we should stop (for force_immediate interruption)
                if getattr(current_thread, "_stop_requested", False):
                    print("Audio playback interrupted.")
                    return

                audio_array = np.frombuffer(
                    bytes(chunk.audio_int16_bytes), dtype=np.int16
                )
                sd.play(audio_array, samplerate=sample_rate)
                sd.wait()

            print("Audio playback finished.")
        except Exception as e:
            if not getattr(current_thread, "_stop_requested", False):
                print(f"Audio error: {e}")
            # Don't re-raise, just exit gracefully
        finally:
            _is_speaking = False

    with _speech_lock:
        # Handle existing speech
        if _current_speech_thread and _current_speech_thread.is_alive():
            if force_immediate:
                # Just mark it to stop, don't call sd.stop()
                _current_speech_thread._stop_requested = True
                _current_speech_thread.join(timeout=0.5)  # Brief wait
                _is_speaking = False  # Reset speaking state when interrupted
            else:
                # Wait for current speech to finish
                _current_speech_thread.join()

        # Start new speech
        _current_speech_thread = threading.Thread(target=_play_audio, daemon=True)
        _current_speech_thread.start()

    return _current_speech_thread


def is_speaking():
    """Check if the robot is currently speaking."""
    global _is_speaking
    return _is_speaking


# Stream-to-speech functionality
_stream_queue = queue.Queue()
_stream_thread = None
_stream_stop_event = threading.Event()


def _process_stream_queue():
    """Process streaming text chunks and speak them."""
    global _is_speaking

    sentence_buffer = ""

    while not _stream_stop_event.is_set():
        try:
            # Get text chunk with timeout
            chunk = _stream_queue.get(timeout=0.1)
            if chunk is None:  # Sentinel to end stream
                # Speak any remaining text
                if sentence_buffer.strip():
                    print(f"🗣️ Final chunk: {sentence_buffer}")
                    _speak_text(sentence_buffer)
                break

            sentence_buffer += chunk

            # Look for sentence endings to speak complete sentences
            sentences = re.split(r"([.!?]+)", sentence_buffer)

            if len(sentences) > 2:  # We have at least one complete sentence
                # Speak complete sentences (text + punctuation pairs)
                for i in range(0, len(sentences) - 2, 2):
                    if sentences[i].strip() and sentences[i + 1]:
                        complete_sentence = sentences[i] + sentences[i + 1]
                        print(f"🗣️ Speaking chunk: {complete_sentence}")
                        _speak_text(complete_sentence)

                # Keep the remaining incomplete sentence
                sentence_buffer = sentences[-1] if sentences[-1] else ""

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Stream processing error: {e}")
            break


def _speak_text(text: str):
    """Synchronously speak text using Piper."""
    try:
        global _is_speaking
        _is_speaking = True

        audio_chunks = voice.synthesize(text.strip())
        sample_rate = voice.config.sample_rate

        for chunk in audio_chunks:
            if _stream_stop_event.is_set():
                break

            audio_array = np.frombuffer(bytes(chunk.audio_int16_bytes), dtype=np.int16)
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()

    except Exception as e:
        print(f"Speech synthesis error: {e}")
    finally:
        _is_speaking = False


def start_stream_speaking():
    """Start the streaming speech processor."""
    global _stream_thread, _stream_stop_event

    _stream_stop_event.clear()
    _stream_thread = threading.Thread(target=_process_stream_queue, daemon=True)
    _stream_thread.start()


def add_to_stream(text_chunk: str):
    """Add a text chunk to the streaming speech queue."""
    _stream_queue.put(text_chunk)


def end_stream():
    """Signal the end of streaming and speak any remaining text."""
    _stream_queue.put(None)  # Sentinel value


def stop_stream_speaking():
    """Stop the streaming speech processor."""
    global _stream_thread, _stream_stop_event

    _stream_stop_event.set()
    if _stream_thread and _stream_thread.is_alive():
        _stream_thread.join(timeout=1.0)
