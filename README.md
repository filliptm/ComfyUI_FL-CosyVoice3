# FL CosyVoice3 - Advanced Text-to-Speech for ComfyUI

Advanced text-to-speech custom nodes for ComfyUI featuring zero-shot voice cloning, cross-lingual synthesis, instruction-based control, and voice conversion powered by the CosyVoice3 model family.

## Features

✨ **7 Powerful Nodes:**
- 🔧 **Model Loader** - Auto-download and cache models
- 📁 **Audio Loader** - Load audio files into workflows
- 🎤 **TTS (SFT)** - Standard TTS with pre-trained speakers
- 👥 **Zero-Shot Clone** - Clone any voice from a reference
- 🌍 **Cross-Lingual** - Speak different languages with same voice
- 🎭 **Instruct** - Control emotions, style, and characteristics
- 🔄 **Voice Conversion** - Convert one voice to another

✅ **Key Capabilities:**
- 9 languages supported (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)
- 18+ Chinese dialects/accents
- Automatic model downloading from ModelScope or HuggingFace
- Model caching to avoid reloading
- Seed control for reproducible outputs
- Speed control (0.5x - 2.0x)
- Streaming support (optional)
- GPU/CPU auto-detection

---

## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "FL CosyVoice3"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI_FL-CosyVoice3.git
cd ComfyUI_FL-CosyVoice3
pip install -r requirements.txt
```

### Method 3: Install CosyVoice Library (Optional but Recommended)
For best compatibility, clone the official CosyVoice repository into the same parent directory as this node pack:

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
```

---

## Quick Start

### Basic TTS Workflow
```
[FL CosyVoice3 Model Loader]
  model_version: "Fun-CosyVoice3-0.5B"
  ↓ (COSYVOICE_MODEL)
[FL CosyVoice3 TTS (SFT)]
  text: "Hello, this is a test."
  speaker_id: "英文女"
  speed: 1.0
  ↓ (AUDIO)
[SaveAudio] → output.wav
```

### Voice Cloning Workflow
```
[FL Audio Loader]
  audio_path: "/path/to/reference_voice.wav"
  ↓ (AUDIO - Reference Voice)

[FL CosyVoice3 Model Loader] → (MODEL)
  ↓
[FL CosyVoice3 Zero-Shot Clone]
  text: "This is my cloned voice speaking."
  reference_audio: (from Audio Loader)
  reference_text: "Original transcript..."
  ↓ (AUDIO)
[SaveAudio] → cloned_voice.wav
```

---

## Node Documentation

### 1. FL CosyVoice3 Model Loader

Downloads and loads CosyVoice models with automatic caching.

**Inputs:**
- `model_version`: Model to load
  - `"Fun-CosyVoice3-0.5B"` ⭐ Recommended
  - `"CosyVoice2-0.5B"`
  - `"CosyVoice-300M"`
- `download_source`: `"ModelScope"` or `"HuggingFace"`
- `device`: `"auto"`, `"cuda"`, `"cpu"`, `"mps"`
- `force_redownload`: Force re-download (optional)
- `force_reload`: Force reload from disk (optional)

**Outputs:**
- `COSYVOICE_MODEL`: Model info dictionary

**Notes:**
- First run will download model (~2-4GB)
- Models stored in `ComfyUI/models/cosyvoice/`
- Models are cached in memory after first load

---

### 2. FL Audio Loader

Load audio files from disk into ComfyUI AUDIO format.

**Inputs:**
- `audio_path`: Path to audio file (WAV, MP3, FLAC, OGG)
- `target_sample_rate`: Target sample rate (0 = keep original)

**Outputs:**
- `AUDIO`: Audio data

**Example:**
```
audio_path: "/Users/you/voice_samples/speaker1.wav"
target_sample_rate: 22050
```

---

### 3. FL CosyVoice3 TTS (SFT)

Standard fine-tuned TTS with pre-trained speaker voices.

**Inputs:**
- `model`: COSYVOICE_MODEL
- `text`: Text to synthesize (multiline)
- `speaker_id`: Pre-trained speaker (e.g., "中文女", "英文男", "日语男", "粤语女")
- `speed`: Speech speed (0.5 - 2.0)
- `stream`: Enable streaming (optional)
- `seed`: Random seed (-1 for random)

**Outputs:**
- `AUDIO`: Synthesized speech

**Common Speaker IDs:**
- Chinese: `"中文女"`, `"中文男"`
- English: `"英文女"`, `"英文男"`
- Japanese: `"日语男"`, `"日语女"`
- Korean: `"韩语女"`
- Cantonese: `"粤语女"`

---

### 4. FL CosyVoice3 Zero-Shot Clone

Clone any voice from a reference audio sample.

**Inputs:**
- `model`: COSYVOICE_MODEL
- `text`: Text to synthesize in cloned voice
- `reference_audio`: AUDIO (voice to clone)
- `reference_text`: Transcript of reference (optional, improves quality)
- `speed`: Speech speed (0.5 - 2.0)
- `stream`: Enable streaming
- `seed`: Random seed

**Outputs:**
- `AUDIO`: Synthesized audio in cloned voice

**Tips:**
- Use 3-10 seconds of clean reference audio
- Providing `reference_text` significantly improves quality
- Reference audio can be in any language

---

### 5. FL CosyVoice3 Cross-Lingual

Speak text in a different language using the same voice characteristics.

**Inputs:**
- `model`: COSYVOICE_MODEL
- `text`: Text in target language
- `reference_audio`: AUDIO (voice in any language)
- `target_language`: `"auto"`, `"zh"`, `"en"`, `"ja"`, `"ko"`, etc.
- `speed`: Speech speed (0.5 - 2.0)
- `stream`: Enable streaming
- `seed`: Random seed

**Outputs:**
- `AUDIO`: Cross-lingual synthesized audio

**Example Use Case:**
- Reference: Chinese speaker audio
- Text: "Hello, how are you today?" (English)
- Result: English speech with Chinese speaker's voice characteristics

---

### 6. FL CosyVoice3 Instruct

Advanced TTS with natural language instruction-based control.

**Inputs:**
- `model`: COSYVOICE_MODEL
- `text`: Text to synthesize
- `instruction`: Natural language instruction (e.g., "Speak with enthusiasm and joy", "女性，温柔")
- `reference_audio`: Optional reference for voice (CosyVoice2/3 only)
- `speed`: Speech speed (0.5 - 2.0)
- `stream`: Enable streaming
- `seed`: Random seed

**Outputs:**
- `AUDIO`: Instruction-controlled speech

**Instruction Examples:**
- English: `"A female speaker with a calm and professional tone"`
- English: `"Speak with excitement and happiness, young male voice"`
- Chinese: `"女性，温柔而甜美的声音"`
- Chinese: `"男性，严肃认真的语气"`

---

### 7. FL CosyVoice3 Voice Conversion

Convert one voice to sound like another (voice-to-voice).

**Inputs:**
- `model`: COSYVOICE_MODEL
- `source_audio`: AUDIO to convert
- `target_audio`: AUDIO reference for target voice
- `speed`: Speech speed (0.5 - 2.0)
- `stream`: Enable streaming
- `seed`: Random seed

**Outputs:**
- `AUDIO`: Voice-converted audio

**Use Cases:**
- Convert speech to different speaker
- Change voice gender
- Apply voice style transfer

---

## Example Workflows

### 1. Multi-Language Voice Cloning
```
[FL Audio Loader] → Chinese voice sample → (AUDIO)
  ↓
[FL CosyVoice3 Model Loader] → (MODEL)
  ↓
[FL CosyVoice3 Cross-Lingual]
  text: "Hello world, this is amazing!" (English)
  reference_audio: (Chinese voice)
  target_language: "en"
  ↓ (AUDIO - English speech in Chinese voice)
[SaveAudio]
```

### 2. Emotion-Controlled TTS
```
[FL CosyVoice3 Model Loader] → (MODEL)
  ↓
[FL CosyVoice3 Instruct]
  text: "I'm so excited to announce this!"
  instruction: "Speak with enthusiasm and joy, female voice"
  speed: 1.1
  ↓ (AUDIO)
[SaveAudio]
```

### 3. Voice Conversion Chain
```
[FL Audio Loader] → source.wav → (AUDIO 1)
[FL Audio Loader] → target.wav → (AUDIO 2)

[FL CosyVoice3 Model Loader] → (MODEL)
  ↓
[FL CosyVoice3 Voice Conversion]
  source_audio: (AUDIO 1)
  target_audio: (AUDIO 2)
  ↓ (AUDIO)
[SaveAudio] → converted.wav
```

---

## Troubleshooting

### Model Download Issues
**Problem:** Model download fails or times out

**Solutions:**
1. Try switching `download_source` to `"HuggingFace"` (or vice versa)
2. Check your internet connection
3. Manually download from [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) and place in `ComfyUI/models/cosyvoice/Fun-CosyVoice3-0.5B/`
4. Enable `force_redownload` to retry

### CUDA Out of Memory
**Problem:** GPU runs out of memory

**Solutions:**
1. Use smaller model: `"CosyVoice-300M"` instead of `"Fun-CosyVoice3-0.5B"`
2. Set `device` to `"cpu"` (slower but uses system RAM)
3. Close other GPU-intensive applications
4. Process shorter text segments

### CosyVoice Import Error
**Problem:** `ModuleNotFoundError: No module named 'cosyvoice'`

**Solutions:**
1. Clone CosyVoice repository to `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   pip install -r requirements.txt
   ```
2. Ensure CosyVoice is in the same parent directory as this node pack

### Audio Quality Issues
**Problem:** Generated audio sounds poor or robotic

**Solutions:**
1. For Zero-Shot: Provide `reference_text` transcript
2. Use higher quality reference audio (clean, 3-10 seconds)
3. Try different speaker IDs for SFT mode
4. Adjust `speed` parameter (try values between 0.9-1.1)

---

## System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 4GB disk space (for models)
- CPU: Modern multi-core processor

### Recommended
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8 or later
- 10GB disk space

---

## Model Information

| Model | Size | Parameters | Quality | Speed | Recommended |
|-------|------|------------|---------|-------|-------------|
| Fun-CosyVoice3-0.5B | ~2GB | 500M | Excellent | Medium | ⭐ Yes |
| CosyVoice2-0.5B | ~2GB | 500M | Very Good | Medium | |
| CosyVoice-300M | ~1.2GB | 300M | Good | Fast | |

**Model Storage:**
- Location: `ComfyUI/models/cosyvoice/`
- Each model automatically downloads on first use
- Models are cached after download

---

## Credits

- **CosyVoice Team** - Original model and research ([GitHub](https://github.com/FunAudioLLM/CosyVoice))
- **FunAudioLLM** - Model development and training
- **FL Nodes** - ComfyUI integration by Machine Delusions

---

## License

Apache 2.0 License

---

## Links

- **GitHub Repository:** https://github.com/filliptm/ComfyUI_FL-CosyVoice3
- **CosyVoice GitHub:** https://github.com/FunAudioLLM/CosyVoice
- **CosyVoice3 Demo:** https://funaudiollm.github.io/cosyvoice3/
- **Research Papers:**
  - [CosyVoice 3](https://arxiv.org/abs/2505.17589)
  - [CosyVoice 2](https://arxiv.org/abs/2412.10117)

---

## Support

For issues, questions, or feature requests:
- Open an issue on [GitHub](https://github.com/filliptm/ComfyUI_FL-CosyVoice3/issues)
- Check existing issues for solutions

---

**Enjoy creating amazing voices with FL CosyVoice3! 🎤🔊**
