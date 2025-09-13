import asyncio
import os
import base64
from typing import Optional, Dict, Any
import subprocess
import wave

# Try to import ollama for local LLM inference
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Model state
_model_loaded = False

# Conversation memory
_conversation_history = []
_max_history_length = 20  # Keep last 20 exchanges

# Model configuration
MODEL_NAME = "gemma3n"
MODEL_PATH = "models/"
SYSTEM_PROMPT = """You are Timbot, a friendly and helpful robot assistant. You communicate live with users through voice, video, and text. 

Key traits:
- You are warm, personable, and enthusiastic about helping
- You respond conversationally and naturally, as if speaking aloud
- You can see and hear the user through video and audio inputs
- Keep responses concise but engaging since you're communicating live
- You have a playful personality but are always helpful and informative
- When you receive audio, video, or multimodal input, acknowledge what you perceive
- You should give short answers, 1 sentance when possible, and avoid use of emoticons, as your text will be converted to speech.
- You remember previous conversations in this session, so you can refer back to earlier topics
- You don't give meta information about yourself or what you are doing unless specifically asked.

Remember: You are speaking to the user live, so respond as if you're having a real-time conversation."""


def _check_ollama_installation():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _ensure_model_available():
    """Ensure the Gemma model is available in Ollama."""
    if not _check_ollama_installation():
        raise RuntimeError(
            "Ollama is not installed or running. Please install Ollama and start the service."
        )

    try:
        # Check if model is already available
        models = ollama.list()
        model_names = [model.model for model in models["models"]]

        if MODEL_NAME not in model_names:
            print(f"Pulling {MODEL_NAME} model... This may take a while.")
            ollama.pull(MODEL_NAME)
            print(f"Model {MODEL_NAME} is now available.")

        return True
    except Exception as e:
        print(f"Error ensuring model availability: {e}")
        return False


def _audio_to_text_description(audio_path: str) -> str:
    """Convert audio file to a text description for the LLM."""
    try:
        # Read audio file properties
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            duration = frames / sample_rate

        return f"[Audio input: {duration:.1f} seconds of audio received]"
    except Exception as e:
        return f"[Audio input: Unable to analyze audio file - {str(e)}]"


def _encode_image_for_vision(image_path: str) -> str:
    """Encode image as base64 for vision models (if supported)."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception:
        return ""


def _add_to_conversation_history(role: str, content: str):
    """Add a message to conversation history."""
    global _conversation_history, _max_history_length

    _conversation_history.append({"role": role, "content": content})

    # Keep only the most recent exchanges
    if len(_conversation_history) > _max_history_length:
        _conversation_history = _conversation_history[-_max_history_length:]


def _get_conversation_messages(user_message: dict) -> list:
    """Get full conversation including history and current message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    messages.extend(_conversation_history)

    # Add current user message
    messages.append(user_message)

    return messages


def clear_conversation_history():
    """Clear the conversation history."""
    global _conversation_history
    _conversation_history = []
    print("🧠 Conversation memory cleared")


async def communicate(
    audio_path: Optional[str] = None,
    image_paths: Optional[list] = None,
    webcam_image_path: Optional[str] = None,
    desktop_image_path: Optional[str] = None,
    text_input: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Communicate with Timbot using multimodal inputs.

    Args:
        audio_path: Path to audio file (WAV format)
        image_paths: List of paths to screenshot images (JPG format) - for backward compatibility
        webcam_image_path: Path to webcam image (JPG format)
        desktop_image_path: Path to desktop screenshot (JPG format)
        text_input: Text message from user
        context: Additional context for the conversation

    Returns:
        Text response from Timbot that should be converted to speech
    """
    global _model_loaded

    if not _model_loaded:
        if not OLLAMA_AVAILABLE:
            return "Sorry, I need Ollama to be installed to communicate. Please install Ollama first."

        # Ensure model is available
        if not _ensure_model_available():
            return "Sorry, I'm having trouble loading my language model. Please check that Ollama is running."
        _model_loaded = True

    # Build the input message
    message_parts = []
    images_base64 = []

    # Add audio description if provided
    if audio_path and os.path.exists(audio_path):
        audio_desc = _audio_to_text_description(audio_path)
        message_parts.append(audio_desc)

    # Handle backward compatibility with image_paths
    if image_paths and len(image_paths) > 0:
        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = _encode_image_for_vision(image_path)
                if encoded_image:
                    images_base64.append(encoded_image)

    # Handle webcam image separately
    if webcam_image_path and os.path.exists(webcam_image_path):
        encoded_image = _encode_image_for_vision(webcam_image_path)
        if encoded_image:
            images_base64.append(encoded_image)
            message_parts.append("[Visual input from webcam: Current view of user/environment through your camera]")

    # Handle desktop image separately  
    if desktop_image_path and os.path.exists(desktop_image_path):
        encoded_image = _encode_image_for_vision(desktop_image_path)
        if encoded_image:
            images_base64.append(encoded_image)
            message_parts.append("[Visual input from desktop: Current view of user's computer screen/desktop]")

    # Legacy handling for image_paths (when new parameters not provided)
    if image_paths and len(image_paths) > 0 and not webcam_image_path and not desktop_image_path:
        if images_base64:
            message_parts.append(
                f"[Visual input: {len(images_base64)} screenshot(s)]"
            )
        else:
            message_parts.append("[Visual input: Unable to process images]")

    # Add text input if provided
    if text_input:
        message_parts.append(f"Text input: {text_input}")

    # Add context if provided
    if context:
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        message_parts.append(f"Context: {context_str}")

    # If no input provided, return a default response
    if not message_parts:
        return "Hello! I'm Timbot. I didn't receive any input from you. How can I help you today?"

    # Combine all inputs into a single message
    full_message = "\n".join(message_parts)

    try:
        # Build the message with images
        user_message = {"role": "user", "content": full_message}
        if images_base64:
            user_message["images"] = images_base64

        # Get conversation messages including history
        messages = _get_conversation_messages(user_message)

        # Make request to local LLM
        response = await asyncio.to_thread(
            ollama.chat,
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200,  # Keep responses concise for live conversation
            },
        )

        response_content = response["message"]["content"].strip()

        # Add both user message and assistant response to conversation history
        _add_to_conversation_history("user", full_message)
        _add_to_conversation_history("assistant", response_content)

        return response_content

    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return "Sorry, I'm having trouble thinking right now. Could you try again in a moment?"


async def initialize_brain():
    """Initialize the brain module and ensure everything is ready."""
    if not OLLAMA_AVAILABLE:
        print("Warning: Ollama not available. Brain functionality will be limited.")
        return False

    if not _check_ollama_installation():
        print("Warning: Ollama is not running. Please start Ollama service.")
        return False

    print("Initializing Timbot's brain...")
    success = _ensure_model_available()

    if success:
        print("Brain initialized successfully!")
        # Test the model with a simple message
        test_response = await communicate(text_input="Hello, are you working?")
        print(f"Brain test: {test_response}")

    return success


async def communicate_stream(
    audio_path: Optional[str] = None,
    image_paths: Optional[list] = None,
    webcam_image_path: Optional[str] = None,
    desktop_image_path: Optional[str] = None,
    text_input: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """Stream communication with Timbot using multimodal inputs.

    Args:
        audio_path: Path to audio file (WAV format)
        image_paths: List of paths to screenshot images (JPG format) - for backward compatibility
        webcam_image_path: Path to webcam image (JPG format)
        desktop_image_path: Path to desktop screenshot (JPG format)
        text_input: Text message from user
        context: Additional context for the conversation

    Yields:
        Text chunks from Timbot response that should be converted to speech
    """
    global _model_loaded

    if not _model_loaded:
        if not OLLAMA_AVAILABLE:
            yield "Sorry, I need Ollama to be installed to communicate. Please install Ollama first."
            return

        # Ensure model is available
        if not _ensure_model_available():
            yield "Sorry, I'm having trouble loading my language model. Please check that Ollama is running."
            return
        _model_loaded = True

    # Build the input message
    message_parts = []
    images_base64 = []

    # Add audio description if provided
    if audio_path and os.path.exists(audio_path):
        audio_desc = _audio_to_text_description(audio_path)
        message_parts.append(audio_desc)

    # Handle backward compatibility with image_paths
    if image_paths and len(image_paths) > 0:
        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = _encode_image_for_vision(image_path)
                if encoded_image:
                    images_base64.append(encoded_image)

    # Handle webcam image separately
    if webcam_image_path and os.path.exists(webcam_image_path):
        encoded_image = _encode_image_for_vision(webcam_image_path)
        if encoded_image:
            images_base64.append(encoded_image)
            message_parts.append("[Visual input from webcam: Current view of user/environment through your camera]")

    # Handle desktop image separately  
    if desktop_image_path and os.path.exists(desktop_image_path):
        encoded_image = _encode_image_for_vision(desktop_image_path)
        if encoded_image:
            images_base64.append(encoded_image)
            message_parts.append("[Visual input from desktop: Current view of user's computer screen/desktop]")

    # Legacy handling for image_paths (when new parameters not provided)
    if image_paths and len(image_paths) > 0 and not webcam_image_path and not desktop_image_path:
        if images_base64:
            message_parts.append(
                f"[Visual input: {len(images_base64)} screenshot(s)]"
            )
        else:
            message_parts.append("[Visual input: Unable to process images]")

    # Add text input if provided
    if text_input:
        message_parts.append(f"Text input: {text_input}")

    # Add context if provided
    if context:
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        message_parts.append(f"Context: {context_str}")

    # If no input provided, return a default response
    if not message_parts:
        yield "Hello! I'm Timbot. I didn't receive any input from you. How can I help you today?"
        return

    # Combine all inputs into a single message
    full_message = "\n".join(message_parts)

    try:
        # Build the message with images
        user_message = {"role": "user", "content": full_message}
        if images_base64:
            user_message["images"] = images_base64

        # Get conversation messages including history
        messages = _get_conversation_messages(user_message)

        # Stream response from local LLM
        response_stream = await asyncio.to_thread(
            ollama.chat,
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200,  # Keep responses concise for live conversation
            },
            stream=True,  # Enable streaming
        )

        # Collect full response for history
        full_response = ""

        # Stream the response chunks
        for chunk in response_stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content.strip():  # Only yield non-empty content
                    full_response += content
                    yield content

        # Add both user message and assistant response to conversation history
        _add_to_conversation_history("user", full_message)
        _add_to_conversation_history("assistant", full_response.strip())

    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        yield "Sorry, I'm having trouble thinking right now. Could you try again in a moment?"


# Convenience function for simple text-only communication
async def chat(message: str) -> str:
    """Simple text-based chat with Timbot."""
    return await communicate(text_input=message)


# Convenience function for streaming text-only communication
async def chat_stream(message: str):
    """Simple streaming text-based chat with Timbot."""
    async for chunk in communicate_stream(text_input=message):
        yield chunk
