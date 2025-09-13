import typer
import asyncio
import subprocess
import os
from async_downloader import AsyncFileDownloader

app = typer.Typer()


def _check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _install_ollama():
    """Install Ollama if not present."""
    if _check_ollama_installed():
        print("✓ Ollama is already installed")
        return True

    print("Installing Ollama...")
    try:
        # Install Ollama using their install script
        result = subprocess.run(
            ["curl", "-fsSL", "https://ollama.ai/install.sh"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Execute the install script
            process = subprocess.run(result.stdout, shell=True, timeout=300)
            if process.returncode == 0:
                print("✓ Ollama installed successfully")
                return True
            else:
                print("✗ Ollama installation failed")
                return False
        else:
            print("✗ Failed to download Ollama install script")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Ollama installation timed out")
        return False
    except Exception as e:
        print(f"✗ Ollama installation error: {e}")
        return False


def _setup_gemma_model():
    """Setup Gemma model in Ollama."""
    if not _check_ollama_installed():
        print("Ollama not installed, skipping model setup")
        return False

    try:
        print("Setting up Gemma3n model in Ollama...")

        # Start Ollama service if not running
        subprocess.run(["ollama", "serve"], timeout=2, capture_output=True)

        # Pull the Gemma2 model
        print("Pulling Gemma3n model (this may take several minutes)...")
        result = subprocess.run(
            ["ollama", "pull", "gemma3n"],
            timeout=1800,  # 30 minute timeout
            text=True,
        )

        if result.returncode == 0:
            print("✓ Gemma3n model ready")
            return True
        else:
            print("✗ Failed to pull Gemma3n model")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Model download timed out")
        return False
    except Exception as e:
        print(f"✗ Model setup error: {e}")
        return False


async def _download_models_async():
    downloads = [
        {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "filepath": "models/en_US-lessac-medium.onnx",
        },
        {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            "filepath": "models/en_US-lessac-medium.onnx.json",
        },
        {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "filepath": "models/vosk-model-small-en-us-0.15.zip",
        },
    ]

    downloader = AsyncFileDownloader(max_concurrent=3)
    results = await downloader.download_multiple(downloads)

    for url, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {url}")

    # Extract Vosk model if downloaded successfully
    vosk_zip_path = "models/vosk-model-small-en-us-0.15.zip"
    vosk_model_dir = "models/vosk-model-small-en-us-0.15"

    if os.path.exists(vosk_zip_path) and not os.path.exists(vosk_model_dir):
        print("Extracting Vosk model...")
        try:
            import zipfile

            with zipfile.ZipFile(vosk_zip_path, "r") as zip_ref:
                zip_ref.extractall("models/")
            print("✓ Vosk model extracted")
            # Clean up zip file
            os.remove(vosk_zip_path)
        except Exception as e:
            print(f"✗ Failed to extract Vosk model: {e}")

    # Setup Ollama and Gemma after downloading other models
    print("\nSetting up LLM components...")
    ollama_success = _install_ollama()

    if ollama_success:
        gemma_success = _setup_gemma_model()
        if gemma_success:
            print("✓ Brain (LLM) setup complete")
        else:
            print("✗ Brain (LLM) setup failed - Timbot will have limited functionality")
    else:
        print("✗ Ollama installation failed - Timbot will have limited functionality")

    return results


async def _start_interactive():
    """Start Timbot in interactive multimodal mode."""
    import signal
    from timbot import ears, eyes, brain

    print("🤖 Starting Timbot Interactive Mode")
    print("=" * 50)

    # Initialize brain
    print("Initializing brain...")
    brain_ready = await brain.initialize_brain()
    if not brain_ready:
        print("⚠️  Brain initialization failed. Text-only mode available.")

    # Start screenshot capture
    print("Starting eyes (screenshot capture)...")
    eyes.start_recording()

    # Global state
    running = True

    def signal_handler(_signum, _frame):
        nonlocal running
        print("\n🛑 Shutting down Timbot...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    print("\n✅ Timbot is ready!")
    print("Commands:")
    print("  - Speak naturally (audio will be recorded)")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Press Ctrl+C to interrupt")
    print("-" * 50)

    # Start concurrent tasks
    try:
        await asyncio.gather(
            _handle_audio_input(),
            _handle_text_input(),
            _process_multimodal_communication(),
            return_exceptions=True,
        )
    except KeyboardInterrupt:
        running = False
    finally:
        # Cleanup
        print("\nCleaning up...")
        ears.stop_listening()
        eyes.stop_recording()
        print("👋 Timbot stopped.")


async def _handle_audio_input():
    """Handle continuous audio input from microphone."""
    from timbot import ears

    print("🎤 Listening for speech...")

    try:
        async for transcribed_text in ears.listen(
            min_sound_duration=0.3,  # Shorter minimum duration
            silence_timeout=1.5,  # Silence timeout before ending
            energy_threshold=0.002,  # VAD energy threshold
            debug=True,  # Enable debug output
        ):
            print(f"🗣️ Speech transcribed: {transcribed_text}")
            # Add to processing queue as text (handled by _process_multimodal_communication)
            await _add_to_processing_queue("text", transcribed_text)
    except Exception as e:
        print(f"Audio input error: {e}")


async def _handle_text_input():
    """Handle text input from command line."""

    while True:
        try:
            # Non-blocking input simulation
            print("\n💬 Type a message (or 'quit' to exit, 'clear' to clear memory):")

            # Use asyncio to make input non-blocking
            text = await asyncio.get_event_loop().run_in_executor(None, input, "> ")

            if text.lower().strip() in ["quit", "exit", "stop"]:
                print("Goodbye!")
                break
            elif text.lower().strip() in ["clear", "clear memory"]:
                from timbot import brain

                brain.clear_conversation_history()
                continue

            if text.strip():
                print(f"📝 Text input: {text}")
                print("🔄 Adding text to processing queue...")
                await _add_to_processing_queue("text", text)

        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"Text input error: {e}")


# Processing queue for multimodal inputs
_processing_queue = asyncio.Queue()


async def _add_to_processing_queue(input_type: str, data):
    """Add input to processing queue."""
    print(f"📬 Adding {input_type} to queue: {data}")
    await _processing_queue.put((input_type, data, asyncio.get_event_loop().time()))
    print(f"✅ Added to queue. Queue size: {_processing_queue.qsize()}")


async def _process_multimodal_communication():
    """Process queued inputs and generate responses."""

    # Buffer for collecting inputs within time window
    input_buffer = []
    last_processing_time = 0
    processing_window = 3.0  # seconds

    while True:
        try:
            # Wait for input or timeout
            try:
                print("⏳ Waiting for queue input...")
                input_type, data, timestamp = await asyncio.wait_for(
                    _processing_queue.get(), timeout=1.0
                )
                print(f"📨 Got {input_type} from queue: {data}")
                input_buffer.append((input_type, data, timestamp))

                # Process if we have inputs and enough time has passed
                current_time = asyncio.get_event_loop().time()

                if (current_time - last_processing_time) >= processing_window or len(
                    input_buffer
                ) >= 3:
                    print(f"🚀 Processing buffer with {len(input_buffer)} items...")
                    await _process_input_buffer(input_buffer)
                    input_buffer.clear()
                    last_processing_time = current_time

            except asyncio.TimeoutError:
                print("⏰ Queue timeout")
                # Process any pending inputs after timeout
                if input_buffer:
                    print(
                        f"🚀 Processing buffer after timeout with {len(input_buffer)} items..."
                    )
                    await _process_input_buffer(input_buffer)
                    input_buffer.clear()
                    last_processing_time = asyncio.get_event_loop().time()

        except Exception as e:
            print(f"Processing error: {e}")
            await asyncio.sleep(1)


async def _process_input_buffer(input_buffer):
    """Process collected inputs and generate streaming response."""
    from timbot import brain, mouth, eyes

    if not input_buffer:
        return

    print(f"\n🧠 Processing {len(input_buffer)} input(s)...")

    # Collect all text inputs (both from typing and transcribed speech)
    text_inputs = []

    for input_type, data, _timestamp in input_buffer:
        if input_type == "text":
            text_inputs.append(data)

    # Combine all text inputs
    combined_text = " ".join(text_inputs) if text_inputs else None

    # Get current webcam and desktop screenshots separately
    webcam_path, desktop_path = eyes.get_webcam_and_desktop_screenshots()

    # Generate streaming response using brain
    try:
        print("🧠 Calling brain.communicate_stream with:")
        print(f"   - Webcam: {'Yes' if webcam_path else 'No'}")
        print(f"   - Desktop: {'Yes' if desktop_path else 'No'}")
        print(f"   - Text: {combined_text}")

        # Start streaming speech processor
        mouth.start_stream_speaking()

        # Start filler words task while waiting for first chunk
        filler_task = asyncio.create_task(_say_filler_words())
        first_chunk_received = False
        response_text = ""

        try:
            async for chunk in brain.communicate_stream(
                audio_path=None,  # No longer using audio files
                webcam_image_path=webcam_path,
                desktop_image_path=desktop_path,
                text_input=combined_text,
                context={
                    "input_count": len(input_buffer),
                    "timestamp": asyncio.get_event_loop().time(),
                },
            ):
                # Cancel filler words when first chunk arrives
                if not first_chunk_received:
                    filler_task.cancel()
                    try:
                        await filler_task
                    except asyncio.CancelledError:
                        pass
                    first_chunk_received = True

                # Add chunk to streaming speech
                mouth.add_to_stream(chunk)
                response_text += chunk
                print(f"🤖 Chunk: {chunk}", end="", flush=True)

            # End the stream
            mouth.end_stream()

        finally:
            # Ensure filler task is cancelled
            if not filler_task.done():
                filler_task.cancel()
                try:
                    await filler_task
                except asyncio.CancelledError:
                    pass

        print(f"\n🤖 Complete response: {response_text}")

        # Note: Screenshots are managed by eyes.py automatically

    except Exception as e:
        print(f"Communication error: {e}")
        error_response = "Sorry, I'm having trouble processing that right now."
        print(f"🤖 Timbot: {error_response}")
        mouth.say(error_response)


async def _say_filler_words():
    """Say filler words while waiting for LLM response."""
    from timbot import mouth
    import random

    filler_words = [
        "Hmm",
        "Let me think about that",
        "Mmm",
        "One moment",
        "Uh huh",
        "Let me see",
        "Well",
        "OK",
        "Right",
    ]

    # Wait a bit before starting filler words
    await asyncio.sleep(2.0)

    while True:
        try:
            # Say a random filler word
            filler = random.choice(filler_words)
            print(f"🤔 Filler: {filler}")
            mouth.say(filler, force_immediate=False)  # Don't interrupt ongoing speech

            # Wait before next filler word (3-5 seconds)
            wait_time = random.uniform(3.0, 5.0)
            await asyncio.sleep(wait_time)

        except asyncio.CancelledError:
            print("Filler words cancelled")
            break
        except Exception as e:
            print(f"Filler words error: {e}")
            break


@app.command()
def download_models():
    """Installs models needed to run timbot."""
    asyncio.run(_download_models_async())


@app.command()
def start():
    """Starts timbot in interactive mode."""
    asyncio.run(_start_interactive())


@app.command()
def clear_memory():
    """Clears timbot's conversation memory."""
    from timbot import brain

    brain.clear_conversation_history()


if __name__ == "__main__":
    app()
