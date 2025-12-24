"""
FL CosyVoice3 - Advanced Text-to-Speech for ComfyUI
Zero-shot voice cloning, cross-lingual synthesis, and instruction-based control
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import nodes
from .nodes.model_loader import FL_CosyVoice3_ModelLoader
from .nodes.zero_shot import FL_CosyVoice3_ZeroShot
from .nodes.cross_lingual import FL_CosyVoice3_CrossLingual
from .nodes.voice_conversion import FL_CosyVoice3_VoiceConversion
from .nodes.audio_crop import FL_CosyVoice3_AudioCrop
from .nodes.dialog import FL_CosyVoice3_Dialog

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FL_CosyVoice3_ModelLoader": FL_CosyVoice3_ModelLoader,
    "FL_CosyVoice3_ZeroShot": FL_CosyVoice3_ZeroShot,
    "FL_CosyVoice3_CrossLingual": FL_CosyVoice3_CrossLingual,
    "FL_CosyVoice3_VoiceConversion": FL_CosyVoice3_VoiceConversion,
    "FL_CosyVoice3_AudioCrop": FL_CosyVoice3_AudioCrop,
    "FL_CosyVoice3_Dialog": FL_CosyVoice3_Dialog,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_CosyVoice3_ModelLoader": "FL CosyVoice3 Model Loader",
    "FL_CosyVoice3_ZeroShot": "FL CosyVoice3 Zero-Shot Clone",
    "FL_CosyVoice3_CrossLingual": "FL CosyVoice3 Cross-Lingual",
    "FL_CosyVoice3_VoiceConversion": "FL CosyVoice3 Voice Conversion",
    "FL_CosyVoice3_AudioCrop": "FL CosyVoice3 Audio Crop",
    "FL_CosyVoice3_Dialog": "FL CosyVoice3 Dialog",
}

# ASCII art banner
ascii_art = """

 ███████╗██╗         ██████╗ ██████╗ ███████╗██╗   ██╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██████╗
 ██╔════╝██║        ██╔════╝██╔═══██╗██╔════╝╚██╗ ██╔╝██║   ██║██╔═══██╗██║██╔════╝██╔════╝╚════██╗
 █████╗  ██║        ██║     ██║   ██║███████╗ ╚████╔╝ ██║   ██║██║   ██║██║██║     █████╗   █████╔╝
 ██╔══╝  ██║        ██║     ██║   ██║╚════██║  ╚██╔╝  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝   ╚═══██╗
 ██║     ███████╗   ╚██████╗╚██████╔╝███████║   ██║    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██████╔╝
 ╚═╝     ╚══════╝    ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═════╝

    Advanced Text-to-Speech | Zero-Shot Voice Cloning | Multi-lingual Synthesis

"""

print(ascii_art)
print("=" * 90)
print("FL CosyVoice3 Custom Nodes Loaded")
print("Version: 1.0.0")
print("=" * 90)
print("")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
