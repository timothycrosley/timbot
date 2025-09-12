import asyncio
import numpy as np
import sounddevice as sd
from typing import AsyncGenerator, Optional
import queue
import time
import tempfile
import wave
import os
from faster_whisper import WhisperModel

# Configuration
SAMPLE_RATE = 44100  # High quality audio
CHUNK_SIZE = 4096
AUDIO_FORMAT = np.int16

# Global state
_listening = False
_audio_queue = queue.Queue()
_whisper_model = None

def _get_whisper_model():
    """Load Whisper model (cached)."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model

def _audio_callback(indata, _frames, _time_info, status):
    """Audio input callback."""
    if status:
        print(f"Audio input status: {status}")
    
    # Convert to mono if stereo
    if indata.shape[1] > 1:
        audio_data = np.mean(indata, axis=1)
    else:
        audio_data = indata[:, 0]
    
    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(AUDIO_FORMAT)
    
    try:
        _audio_queue.put_nowait(audio_int16)
    except queue.Full:
        # Drop oldest audio if queue is full
        try:
            _audio_queue.get_nowait()
            _audio_queue.put_nowait(audio_int16)
        except queue.Empty:
            pass

def _detect_sound(audio_chunk: np.ndarray, threshold: float = 1000) -> bool:
    """Simple sound detection based on energy level."""
    energy = np.mean(np.abs(audio_chunk.astype(np.float32)))
    return energy > threshold

def _calculate_background_noise(audio_chunks: list, percentile: int = 50) -> float:
    """Calculate background noise level from recent audio chunks."""
    if not audio_chunks:
        return 1000.0  # Default threshold
    
    energies = [np.mean(np.abs(chunk.astype(np.float32))) for chunk in audio_chunks]
    background_level = np.percentile(energies, percentile)
    # Adaptive threshold: background + 50% margin
    return background_level * 1.5

async def _transcribe_audio_async(audio_data: np.ndarray) -> Optional[str]:
    """Transcribe audio data using Whisper in a thread pool to avoid blocking."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_audio_sync, audio_data)

def _transcribe_audio_sync(audio_data: np.ndarray) -> Optional[str]:
    """Synchronous transcription function for thread pool execution."""
    try:
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='timbot_audio_')
        os.close(temp_fd)  # Close the file descriptor, we'll use wave module
        
        # Write audio data to WAV file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        # Transcribe with Whisper
        model = _get_whisper_model()
        segments, info = model.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return text if text else None
        
    except Exception as e:
        print(f"Transcription error: {e}")
        # Clean up temporary file if it exists
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except OSError:
            pass
        return None

async def listen(
    duration_seconds: Optional[float] = None,
    min_sound_duration: float = 0.5,
    sound_threshold: float = 1000,
    silence_timeout: float = 1.2,
    max_recording_duration: float = 8.0,
    debug: bool = False
) -> AsyncGenerator[str, None]:
    """
    Listen for sounds and yield transcribed text when sound segments are detected.
    
    Args:
        duration_seconds: How long to listen (None for infinite)
        min_sound_duration: Minimum duration of sound to record
        sound_threshold: Energy threshold for sound detection
        silence_timeout: How long to wait for silence before ending recording
        max_recording_duration: Maximum duration to record continuously
        debug: Print debug information
    
    Yields:
        Transcribed text from the recorded audio
    """
    global _listening
    
    _listening = True
    
    # Clear audio queue
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
    
    print("Starting to listen for sounds...")
    
    # Initialize variables
    audio_buffer = np.array([], dtype=AUDIO_FORMAT)
    sound_detected = False
    sound_start_time = None
    silence_duration = 0
    start_time = time.time()
    
    # For adaptive threshold calculation
    background_audio_chunks = []
    adaptive_threshold = sound_threshold
    chunks_processed = 0
    
    # Start audio stream
    stream = sd.InputStream(
        callback=_audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype=np.float32
    )
    
    try:
        with stream:
            
            while _listening:
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                try:
                    # Get audio chunk with timeout
                    chunk = _audio_queue.get(timeout=0.1)
                    chunks_processed += 1
                    
                    # Update adaptive threshold every 20 chunks (about 2 seconds)
                    if chunks_processed % 20 == 0:
                        background_audio_chunks.append(chunk)
                        if len(background_audio_chunks) > 10:  # Keep last 10 chunks
                            background_audio_chunks.pop(0)
                        adaptive_threshold = _calculate_background_noise(background_audio_chunks)
                    
                    # Check for sound activity using adaptive threshold
                    has_sound = _detect_sound(chunk, adaptive_threshold)
                    
                    if debug:
                        energy = np.mean(np.abs(chunk.astype(np.float32)))
                        print(f"Energy: {energy:.1f}, Threshold: {adaptive_threshold:.1f}, Sound: {has_sound}")
                    
                    if has_sound:
                        if not sound_detected:
                            sound_detected = True
                            sound_start_time = time.time()
                            print("Sound detected, recording...")
                        
                        audio_buffer = np.concatenate([audio_buffer, chunk])
                        silence_duration = 0
                    else:
                        if sound_detected:
                            silence_duration += len(chunk) / SAMPLE_RATE
                            
                            # End recording if silence is long enough OR max duration reached
                            sound_duration = time.time() - sound_start_time
                            
                            if silence_duration >= silence_timeout or sound_duration >= max_recording_duration:
                                if sound_duration >= min_sound_duration and len(audio_buffer) > 0:
                                    reason = "silence" if silence_duration >= silence_timeout else "max duration"
                                    print(f"Recording complete ({sound_duration:.2f}s, {reason})")
                                    
                                    # Transcribe audio and yield the text
                                    transcribed_text = await _transcribe_audio_async(audio_buffer)
                                    if transcribed_text:
                                        print(f"Transcribed: {transcribed_text}")
                                        yield transcribed_text
                                
                                # Reset for next sound segment
                                audio_buffer = np.array([], dtype=AUDIO_FORMAT)
                                sound_detected = False
                                sound_start_time = None
                                silence_duration = 0
                
                except queue.Empty:
                    # No new audio, continue
                    continue
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    continue
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
    
    finally:
        # Process any remaining audio in buffer before stopping
        if sound_detected and len(audio_buffer) > 0:
            sound_duration = time.time() - sound_start_time if sound_start_time else 0
            if sound_duration >= min_sound_duration:
                print(f"Processing remaining audio ({sound_duration:.2f}s)")
                transcribed_text = await _transcribe_audio_async(audio_buffer)
                if transcribed_text:
                    print(f"Final transcription: {transcribed_text}")
                    yield transcribed_text
        
        _listening = False
        print("Stopped listening.")

def stop_listening():
    """Stop the listening process."""
    global _listening
    _listening = False

# Convenience function for testing
async def listen_once(timeout: float = 10.0) -> Optional[str]:
    """Listen for a single sound and return the transcribed text."""
    async for transcribed_text in listen(duration_seconds=timeout):
        return transcribed_text
    return None