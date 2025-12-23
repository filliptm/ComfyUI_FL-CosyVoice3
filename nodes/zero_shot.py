"""
FL CosyVoice3 Zero-Shot Voice Cloning Node
Clone any voice from a reference audio sample
"""

import torch
import random
from typing import Tuple, Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.audio_utils import tensor_to_comfyui_audio, save_raw_audio_to_tempfile, cleanup_temp_file
except (ImportError, ValueError):
    from utils.audio_utils import tensor_to_comfyui_audio, save_raw_audio_to_tempfile, cleanup_temp_file

# Import language detection utility
import re
_chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

def contains_chinese(text):
    """Check if text contains Chinese characters"""
    return bool(_chinese_char_pattern.search(text))

# Whisper model cache for auto-transcription
_whisper_model = None

def get_whisper_model():
    """Get cached Whisper model for transcription"""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print("[FL CosyVoice3 ZeroShot] Loading Whisper model for auto-transcription...")
            _whisper_model = whisper.load_model("base")
            print("[FL CosyVoice3 ZeroShot] Whisper model loaded successfully")
        except Exception as e:
            print(f"[FL CosyVoice3 ZeroShot] Failed to load Whisper: {e}")
            return None
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    """
    Auto-transcribe audio using Whisper when no reference text is provided.

    Args:
        audio_path: Path to the audio file to transcribe

    Returns:
        Transcribed text, or empty string if transcription fails
    """
    try:
        model = get_whisper_model()
        if model is None:
            return ""

        result = model.transcribe(audio_path, language=None)  # Auto-detect language
        transcript = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        print(f"[FL CosyVoice3 ZeroShot] Whisper detected language: {detected_lang}")
        return transcript
    except Exception as e:
        print(f"[FL CosyVoice3 ZeroShot] Whisper transcription failed: {e}")
        return ""

def is_cosyvoice3_model(model_info: Dict[str, Any]) -> bool:
    """Check if the loaded model is CosyVoice3"""
    version = model_info.get("model_version", "").lower()
    return "cosyvoice3" in version or "fun-cosyvoice3" in version


class FL_CosyVoice3_ZeroShot:
    """
    Zero-shot voice cloning from reference audio
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone_voice"
    CATEGORY = "🔊FL CosyVoice3/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL", {
                    "description": "CosyVoice model from ModelLoader"
                }),
                "text": ("STRING", {
                    "default": "Hello, this is my cloned voice speaking.",
                    "multiline": True,
                    "description": "Text to synthesize in cloned voice"
                }),
                "reference_audio": ("AUDIO", {
                    "description": "Reference voice to clone (max 30 seconds, recommended 3-10s)"
                }),
                "reference_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "description": "IMPORTANT: Exact transcript of what is spoken in the reference audio. Required for good results!"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Speech speed multiplier"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "description": "Random seed (-1 for random)"
                }),
            }
        }

    def clone_voice(
        self,
        model: Dict[str, Any],
        text: str,
        reference_audio: Dict[str, Any],
        reference_text: str,
        speed: float = 1.0,
        seed: int = -1
    ) -> Tuple[Dict[str, Any]]:
        """
        Clone voice from reference audio

        Args:
            model: CosyVoice model info dict
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            reference_text: Transcript of reference audio
            speed: Speech speed
            seed: Random seed

        Returns:
            Tuple containing audio dict
        """
        print(f"\n{'='*60}")
        print(f"[FL CosyVoice3 ZeroShot] Cloning voice...")
        print(f"[FL CosyVoice3 ZeroShot] Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"[FL CosyVoice3 ZeroShot] Reference text: {reference_text[:50] if reference_text else 'Not provided'}{'...' if len(reference_text) > 50 else ''}")
        print(f"[FL CosyVoice3 ZeroShot] Speed: {speed}x")
        print(f"{'='*60}\n")

        temp_file = None

        try:
            # Set seed if specified
            if seed >= 0:
                torch.manual_seed(seed)
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Get model instance
            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate  # Use actual model sample rate (24000 for v2/v3)

            # Prepare reference audio
            print(f"[FL CosyVoice3 ZeroShot] Preparing reference audio...")
            print(f"[FL CosyVoice3 ZeroShot] Model sample rate: {sample_rate} Hz")

            # Check audio duration - CosyVoice only supports up to 30 seconds for voice cloning
            ref_waveform = reference_audio['waveform']
            ref_sample_rate = reference_audio['sample_rate']
            ref_duration = ref_waveform.shape[-1] / ref_sample_rate

            if ref_duration > 30:
                error_msg = (
                    f"Reference audio is too long ({ref_duration:.1f} seconds). "
                    f"CosyVoice only supports reference audio up to 30 seconds for voice cloning. "
                    f"Please use the FL Audio Crop node to trim your audio to 30 seconds or less. "
                    f"Recommended: 3-10 seconds for best quality."
                )
                print(f"\n{'='*60}")
                print(f"[FL CosyVoice3 ZeroShot] ERROR: {error_msg}")
                print(f"{'='*60}\n")
                raise ValueError(error_msg)

            print(f"[FL CosyVoice3 ZeroShot] Reference audio duration: {ref_duration:.1f}s (max 30s)")

            # DEBUG: Save detailed info about the incoming audio
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Incoming audio waveform shape: {ref_waveform.shape}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Incoming audio dtype: {ref_waveform.dtype}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Incoming audio min/max: {ref_waveform.min():.4f} / {ref_waveform.max():.4f}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Incoming audio sample_rate: {ref_sample_rate}")

            # Save audio directly WITHOUT preprocessing - CosyVoice's load_wav() handles mono/resampling
            temp_file = save_raw_audio_to_tempfile(reference_audio)
            print(f"[FL CosyVoice3 ZeroShot] Saved raw audio to: {temp_file}")

            # DEBUG: Re-load and check the saved temp file using soundfile (more compatible)
            import soundfile as sf
            debug_data, debug_sr = sf.read(temp_file)
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Re-loaded temp file shape: {debug_data.shape}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Re-loaded temp file sr: {debug_sr}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Re-loaded temp file min/max: {debug_data.min():.4f} / {debug_data.max():.4f}")

            # DEBUG: Save a copy of the temp file for analysis
            import shutil
            debug_copy_path = os.path.join(parent_dir, "debug_comfyui_input_audio.wav")
            shutil.copy(temp_file, debug_copy_path)
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Saved input audio copy to: {debug_copy_path}")

            # Detect model version for proper formatting (use cached flag or detect from version string)
            is_v3 = model.get("is_cosyvoice3", False) or is_cosyvoice3_model(model)

            # Determine transcript - use provided text or auto-transcribe
            use_cross_lingual_fallback = False
            if reference_text and reference_text.strip():
                transcript = reference_text.strip()
                print(f"[FL CosyVoice3 ZeroShot] Using provided reference text")
            else:
                # Auto-transcribe using Whisper
                print(f"[FL CosyVoice3 ZeroShot] No reference text provided - auto-transcribing with Whisper...")
                transcript = transcribe_audio(temp_file)
                if transcript:
                    print(f"[FL CosyVoice3 ZeroShot] Auto-transcribed: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
                else:
                    print(f"[FL CosyVoice3 ZeroShot] WARNING: Transcription failed or empty audio.")
                    print(f"[FL CosyVoice3 ZeroShot] Falling back to cross-lingual mode (voice cloning without transcript)...")
                    use_cross_lingual_fallback = True

            # Format prompt text based on model version
            if is_v3:
                # CosyVoice3 format: system_prompt<|endofprompt|>transcript
                if transcript and not use_cross_lingual_fallback:
                    formatted_prompt_text = f"You are a helpful assistant.<|endofprompt|>{transcript}"
                else:
                    formatted_prompt_text = None  # Signal to use cross-lingual
            else:
                # CosyVoice v1/v2 format: just the reference text, no system prompt
                formatted_prompt_text = transcript if transcript else None

            # Generate speech
            if use_cross_lingual_fallback or formatted_prompt_text is None:
                # Use cross-lingual as fallback (extracts voice without needing transcript)
                print(f"[FL CosyVoice3 ZeroShot] Using cross-lingual mode (no transcript required)...")

                # For CosyVoice3 cross-lingual, add system prompt to tts_text
                if is_v3:
                    formatted_tts_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                else:
                    formatted_tts_text = text

                print(f"[FL CosyVoice3 ZeroShot] DEBUG - tts_text: {formatted_tts_text[:100]}...")
                print(f"[FL CosyVoice3 ZeroShot] DEBUG - prompt_wav: {temp_file}")

                output = cosyvoice_model.inference_cross_lingual(
                    tts_text=formatted_tts_text,
                    prompt_wav=temp_file,
                    stream=False,
                    speed=speed
                )
            else:
                # Use standard zero-shot with transcript
                print(f"[FL CosyVoice3 ZeroShot] Running zero-shot inference...")
                print(f"[FL CosyVoice3 ZeroShot] DEBUG - tts_text: {text[:100]}...")
                print(f"[FL CosyVoice3 ZeroShot] DEBUG - prompt_text: {formatted_prompt_text[:100]}...")
                print(f"[FL CosyVoice3 ZeroShot] DEBUG - prompt_wav: {temp_file}")

                output = cosyvoice_model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=formatted_prompt_text,
                    prompt_wav=temp_file,
                    stream=False,
                    speed=speed
                )

            # Convert generator to list if needed
            if hasattr(output, '__iter__') and not isinstance(output, dict):
                output = list(output)[0]

            waveform = output['tts_speech']

            # Ensure waveform is on CPU
            if waveform.device != torch.device('cpu'):
                waveform = waveform.cpu()

            # DEBUG: Save output audio before ComfyUI format conversion
            debug_output_path = os.path.join(parent_dir, "debug_comfyui_output_audio.wav")
            # Use soundfile for compatibility
            output_np = waveform.numpy()
            if output_np.ndim == 2:
                output_np = output_np.T  # (channels, samples) -> (samples, channels)
            sf.write(debug_output_path, output_np, sample_rate)
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Saved output audio to: {debug_output_path}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Output waveform shape: {waveform.shape}")
            print(f"[FL CosyVoice3 ZeroShot] DEBUG - Output waveform min/max: {waveform.min():.4f} / {waveform.max():.4f}")

            # Convert to ComfyUI AUDIO format
            audio = tensor_to_comfyui_audio(waveform, sample_rate)

            duration = waveform.shape[-1] / sample_rate

            print(f"\n{'='*60}")
            print(f"[FL CosyVoice3 ZeroShot] Voice cloned successfully!")
            print(f"[FL CosyVoice3 ZeroShot] Duration: {duration:.2f} seconds")
            print(f"[FL CosyVoice3 ZeroShot] Sample rate: {sample_rate} Hz")
            print(f"{'='*60}\n")

            return (audio,)

        except Exception as e:
            error_msg = f"Error cloning voice: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[FL CosyVoice3 ZeroShot] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            # Return empty audio on error
            empty_audio = {
                "waveform": torch.zeros(1, 1, 22050),
                "sample_rate": 22050
            }
            return (empty_audio,)

        finally:
            # Clean up temp file
            cleanup_temp_file(temp_file)
