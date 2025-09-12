import asyncio
import numpy as np
import json
import os
import time
import vosk
import pyaudio
from timbot import mouth
from typing import Optional, AsyncGenerator

# Configuration - Simple and fast
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Smaller buffer for faster processing
SILENCE_DELAY = 1  # seconds of silence before considering speech ended
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Use small model for speed

# Global state
_listening = False
_vosk_model = None
_vosk_recognizer = None
_pyaudio_instance = None
_audio_stream = None


def _get_vosk_model():
    """Load Vosk model (cached) - using small model for speed."""
    global _vosk_model, _vosk_recognizer
    if _vosk_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model not found at {MODEL_PATH}. Please download models first."
            )

        print(f"Loading Vosk model from {MODEL_PATH}...")
        _vosk_model = vosk.Model(MODEL_PATH)

        # Create recognizer with speed optimizations
        _vosk_recognizer = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE)
        _vosk_recognizer.SetWords(False)  # Disable word timestamps for speed
        _vosk_recognizer.SetPartialWords(False)  # Disable partial word info

        print("✅ Vosk model loaded successfully")
    return _vosk_model, _vosk_recognizer


def _init_audio_stream():
    """Initialize PyAudio stream with optimized settings."""
    global _pyaudio_instance, _audio_stream

    try:
        if _pyaudio_instance is None:
            _pyaudio_instance = pyaudio.PyAudio()

        if _audio_stream is None:
            _audio_stream = _pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )

        return _audio_stream
    except Exception as e:
        print(f"Failed to initialize audio stream: {e}")
        _cleanup_audio()
        raise


def _cleanup_audio():
    """Clean up audio resources."""
    global _pyaudio_instance, _audio_stream

    try:
        if _audio_stream:
            _audio_stream.stop_stream()
            _audio_stream.close()
            _audio_stream = None
    except Exception as e:
        print(f"Warning: Error closing audio stream: {e}")

    try:
        if _pyaudio_instance:
            _pyaudio_instance.terminate()
            _pyaudio_instance = None
    except Exception as e:
        print(f"Warning: Error terminating PyAudio: {e}")


def _cleanup_vosk():
    """Clean up Vosk resources."""
    global _vosk_model, _vosk_recognizer
    _vosk_model = None
    _vosk_recognizer = None


async def listen(
    duration_seconds: Optional[float] = None,
    min_sound_duration: float = 0.3,
    silence_timeout: float = SILENCE_DELAY,
    energy_threshold: float = 0.02,
    debug: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Listen for sounds using simple, fast Vosk approach.

    Args:
        duration_seconds: How long to listen (None for infinite)
        min_sound_duration: Minimum duration of sound to record (unused in simple approach)
        silence_timeout: How long to wait for silence before ending recording
        energy_threshold: Energy threshold for voice activity detection (unused in simple approach)
        debug: Print debug information

    Yields:
        Transcribed text from the recorded audio
    """
    global _listening

    _listening = True

    try:
        # Initialize audio and Vosk
        stream = _init_audio_stream()
        model, recognizer = _get_vosk_model()

        print("🎤 Listening for sound...")

        start_time = time.time()
        total_text = ""
        last_spoke = time.time()

        while _listening:
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break

            try:
                # Skip if robot is speaking
                if mouth.is_speaking():
                    await asyncio.sleep(0.1)
                    continue

                # Read audio data
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Process with Vosk
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result["text"]:  # Only process if there's actual text
                        total_text += result["text"] + " "
                        last_spoke = time.time()
                        if debug:
                            print(f"🔄 Accumulating: {total_text.strip()}")

                # Check for silence timeout - yield accumulated text
                elif total_text and (time.time() - last_spoke) > silence_timeout:
                    final_text = total_text.strip()
                    if final_text:
                        if debug:
                            print(f"✅ Complete transcription: {final_text}")
                        yield final_text
                    total_text = ""
                    # Reset recognizer for next utterance
                    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
                    recognizer.SetWords(False)
                    recognizer.SetPartialWords(False)

            except Exception as e:
                if debug:
                    print(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)
                # Try to recover by reinitializing audio
                try:
                    _cleanup_audio()
                    stream = _init_audio_stream()
                except Exception as recovery_error:
                    print(f"Failed to recover audio stream: {recovery_error}")
                    break
                continue

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.01)

    finally:
        _listening = False
        _cleanup_audio()
        _cleanup_vosk()
        if debug:
            print("\n🛑 Stopped listening.")


def stop_listening():
    """Stop the listening process."""
    global _listening
    _listening = False


def test_audio_system():
    """Test if audio system is working before starting main application."""
    try:
        print("Testing audio system...")

        # Test Vosk model loading
        if not os.path.exists(MODEL_PATH):
            return False, f"Vosk model not found at {MODEL_PATH}"

        model, recognizer = _get_vosk_model()
        print("✓ Vosk model loaded successfully")

        # Test audio stream
        _init_audio_stream()
        print("✓ Audio stream initialized")

        return True, f"Audio system test passed - Using {MODEL_PATH} at {SAMPLE_RATE}Hz"

    except Exception as e:
        return False, f"Audio system test failed: {e}"
    finally:
        _cleanup_audio()
        _cleanup_vosk()


# Convenience function for testing
async def listen_once(timeout: float = 10.0) -> Optional[str]:
    """Listen for a single sound and return the transcribed text."""
    async for transcribed_text in listen(duration_seconds=timeout):
        return transcribed_text
    return None


def test_vosk_recognition():
    """Test Vosk speech recognition with the simple approach."""
    try:
        print("🎙️ Testing Vosk recognition...")
        print("Please speak when ready - will capture until 1 second of silence...")

        # Initialize audio system
        stream = _init_audio_stream()
        model, recognizer = _get_vosk_model()

        print(
            f"📊 Audio settings: Sample rate={SAMPLE_RATE}Hz, Chunk size={CHUNK_SIZE}"
        )
        print("🔴 Listening... speak now!")

        total_text = ""
        last_spoke = time.time()
        start_time = time.time()

        try:
            while True:
                # Stop after 30 seconds max
                if time.time() - start_time > 30:
                    break

                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result["text"]:  # Only print if there's actual text
                        total_text += result["text"] + " "
                        last_spoke = time.time()
                        print(f"🔄 Current: {total_text.strip()}")

                elif total_text and (time.time() - last_spoke) > SILENCE_DELAY:
                    final_text = total_text.strip()
                    print(f"🏁 Final result: '{final_text}'")

                    if final_text:
                        words = final_text.split()
                        print(f"📊 Recognized {len(words)} words: {words}")
                        return True, final_text
                    else:
                        print("❌ No speech recognized")
                        return False, "No speech detected"

        except KeyboardInterrupt:
            final_text = total_text.strip()
            if final_text:
                print(f"\n🏁 Interrupted - Final result: '{final_text}'")
                return True, final_text
            else:
                print("\n❌ No speech captured")
                return False, "No speech detected"

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False, str(e)
    finally:
        _cleanup_audio()
        _cleanup_vosk()


def monitor_audio_live(duration: float = 30.0):
    """Monitor audio levels in real-time."""
    try:
        print("🎙️ Live audio monitoring...")
        print("Please speak normally and watch for audio activity...")
        print("Press Ctrl+C to stop early")

        stream = _init_audio_stream()

        start_time = time.time()

        while (time.time() - start_time) < duration:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Show raw audio values for debugging
                audio_array = np.frombuffer(data, dtype=np.int16)
                raw_min = int(np.min(audio_array))
                raw_max = int(np.max(audio_array))
                raw_rms = int(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))

                print(
                    f"Raw audio: min={raw_min:6d} max={raw_max:6d} rms={raw_rms:6d}",
                    end="\r",
                    flush=True,
                )
                time.sleep(0.1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break

        print("\n📊 Monitoring complete")

    except Exception as e:
        print(f"Monitoring failed: {e}")
    finally:
        _cleanup_audio()


def calibrate_microphone(duration: float = 10.0):
    """Simple microphone test - just check if we can hear audio."""
    try:
        print("🎙️ Testing microphone...")
        print("Please speak to test audio input...")

        stream = _init_audio_stream()

        start_time = time.time()
        max_amplitude = 0

        while (time.time() - start_time) < duration:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                chunk_max = np.max(np.abs(audio_array))
                max_amplitude = max(max_amplitude, chunk_max)

                print(
                    f"Max amplitude: {chunk_max:5d} (overall max: {max_amplitude})",
                    end="\r",
                    flush=True,
                )
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during test: {e}")
                break

        print("\n📊 Microphone Test Results:")
        print(f"   Maximum amplitude detected: {max_amplitude}")

        if max_amplitude < 100:
            print("❌ Very low signal - microphone may not be working")
        elif max_amplitude < 1000:
            print("⚠️  Low signal - consider increasing microphone volume")
        else:
            print("✅ Good microphone signal detected")

        return max_amplitude > 100

    except Exception as e:
        print(f"Microphone test failed: {e}")
        return False
    finally:
        _cleanup_audio()


def list_audio_devices():
    """List all available audio input devices."""
    test_pa = pyaudio.PyAudio()

    try:
        device_count = test_pa.get_device_count()
        print(f"Found {device_count} audio devices:")

        for i in range(device_count):
            try:
                device_info = test_pa.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    print(
                        f"  Device {i}: {device_info['name']} (input channels: {device_info['maxInputChannels']})"
                    )
            except Exception:
                continue

    finally:
        test_pa.terminate()
