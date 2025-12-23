"""
FL CosyVoice3 Audio Crop Node
Crop/trim audio to specific start and end times
"""

import torch
from typing import Tuple, Dict, Any


class FL_CosyVoice3_AudioCrop:
    """
    Crop (trim) audio to a specific start and end time.
    Useful for trimming reference audio to the recommended 3-10 second range.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop_audio"
    CATEGORY = "🔊FL CosyVoice3/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "description": "Input audio to crop"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "description": "Start time in seconds"
                }),
                "end_time": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "description": "End time in seconds"
                }),
            }
        }

    def crop_audio(
        self,
        audio: Dict[str, Any],
        start_time: float = 0.0,
        end_time: float = 10.0
    ) -> Tuple[Dict[str, Any]]:
        """
        Crop audio to specific start and end times

        Args:
            audio: Input audio dict with 'waveform' and 'sample_rate'
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Tuple containing cropped audio dict
        """
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        # Calculate frame indices
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)

        # Get total frames
        total_frames = waveform.shape[-1]

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        if start_frame >= end_frame:
            print(f"[FL CosyVoice3 AudioCrop] Warning: Invalid time range, returning original audio")
            return (audio,)

        # Crop waveform
        cropped_waveform = waveform[..., start_frame:end_frame]

        cropped_audio = {
            'waveform': cropped_waveform,
            'sample_rate': sample_rate
        }

        duration = (end_frame - start_frame) / sample_rate
        print(f"[FL CosyVoice3 AudioCrop] Cropped: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")

        return (cropped_audio,)
