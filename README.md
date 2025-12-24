# FL CosyVoice3

Advanced text-to-speech nodes for ComfyUI powered by the CosyVoice3 model family. Features zero-shot voice cloning, cross-lingual synthesis, and voice conversion.

[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

![Workflow Preview](assets/workflow_preview.png)

## Features

- **Zero-Shot Voice Cloning** - Clone any voice from 3-10 seconds of reference audio
- **Cross-Lingual Synthesis** - Speak different languages while preserving voice characteristics
- **Voice Conversion** - Transform one voice to sound like another
- **9 Languages** - Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- **Auto Transcription** - Built-in Whisper integration for reference audio
- **Speed Control** - Adjustable speech rate (0.5x - 2.0x)

## Nodes

| Node | Description |
|------|-------------|
| **Model Loader** | Downloads and caches CosyVoice models |
| **Zero-Shot Clone** | Clone voices from reference audio |
| **Cross-Lingual** | Generate speech in different languages |
| **Voice Conversion** | Convert source audio to target voice |
| **Audio Crop** | Trim audio to specific time ranges |

## Installation

### ComfyUI Manager
Search for "FL CosyVoice3" and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI_FL-CosyVoice3.git
cd ComfyUI_FL-CosyVoice3
pip install -r requirements.txt
```

## Quick Start

1. Add **FL CosyVoice3 Model Loader** and select `Fun-CosyVoice3-0.5B`
2. Connect to **Zero-Shot Clone** or **Cross-Lingual** node
3. Provide reference audio (3-10 seconds recommended)
4. Enter your text and generate

## Models

| Model | Size | Notes |
|-------|------|-------|
| Fun-CosyVoice3-0.5B | ~2GB | Recommended |
| CosyVoice2-0.5B | ~2GB | Alternative |
| CosyVoice-300M | ~1.2GB | Lightweight |

Models download automatically on first use to `ComfyUI/models/cosyvoice/`.

## Requirements

- Python 3.10+
- 8GB RAM minimum (16GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM recommended (CPU supported)

## Credits

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by FunAudioLLM
- ComfyUI integration by [Machine Delusions](https://github.com/filliptm)

## License

Apache 2.0
