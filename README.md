# ASR Benchmarking for Bangalore Locality Names

This project benchmarks Automatic Speech Recognition (ASR) systems on Indian conversational speech (Hindi/Hinglish), specifically targeting the accuracy of transcribing Bangalore locality names under various noise, speaker, and recording conditions.

The benchmark compares:
1. **Deepgram Nova-2** (General-purpose, high-speed API)
2. **Sarvam AI Saarika:v1** (India-centric, regional dialect-tuned API)
3. **OpenAI Whisper Local** (General-purpose, open-source model)
4. **AI4Bharat IndicWhisper** (Fine-tuned open-source model for Indian languages)

---

## Folder Structure

```text
ASR/
├── audio/                      # Audio recording files (20 total)
├── ground_truth.csv            # Expected transcripts and target localities
├── requirements.txt            # Python dependencies
├── generate_mock_audio.py      # Quick testing script to generate silent WAV audios
├── asr_benchmark.py            # Primary evaluation script
└── README.md                   # This instruction file
```

---

## Local Setup

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```
*(On Windows, you may need to install PyTorch separately first if you run into installation issues with standard packages: `pip install torch --index-url https://download.pytorch.org/whl/cpu`)*

### 2. Set Up API Keys
Get API keys from [Deepgram Console](https://console.deepgram.com/) and [Sarvam AI Developer Portal](https://www.sarvam.ai/).
Set them as environment variables:

**In PowerShell (Windows):**
```powershell
$env:DEEPGRAM_API_KEY="0c386e1cf781eaa9acac2e83a9f5878c4d66f818"
$env:SARVAM_API_KEY="sk_1lnojldo_khBQP8miRGWEM7GFr2dptSyx"
```

**In Command Prompt (Windows):**
```cmd
set DEEPGRAM_API_KEY=0c386e1cf781eaa9acac2e83a9f5878c4d66f818
set SARVAM_API_KEY=sk_1lnojldo_khBQP8miRGWEM7GFr2dptSyx
```

**In Terminal (Linux/macOS):**
```bash
export DEEPGRAM_API_KEY="0c386e1cf781eaa9acac2e83a9f5878c4d66f818"
export SARVAM_API_KEY="sk_1lnojldo_khBQP8miRGWEM7GFr2dptSyx"
```

---

## How to Run

### Step 1: Generate Mock Files (for verification)
Before recording your own voice, verify the pipeline is working by running:
```bash
python generate_mock_audio.py
```
This generates 20 valid silent `.wav` files inside the `./audio` folder corresponding to the names in `ground_truth.csv`.

### Step 2: Run Benchmarking Pipeline
Run the benchmark script:
```bash
python asr_benchmark.py
```
- If you have **not** set API keys and do not have Whisper/Transformers installed, the pipeline will run in **Demo Mode** using simulated outputs. This lets you check that all metrics (WER, CER, Fuzzy matching) compute correctly.
- Once you set API keys and install local libraries, the script will run live transcriptions using the actual services.

The script outputs a summary table in the terminal and writes detailed evaluations to `results.csv`.

---

## Google Colab Execution (Free GPU Tier)

Since local Whisper models (especially AI4Bharat's IndicWhisper) run much faster on GPUs, it is highly recommended to run this on Google Colab's free T4 GPU tier:

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Change the runtime type to **T4 GPU** (`Runtime` -> `Change runtime type` -> select `T4 GPU`).
3. Upload the files:
   - Create a folder named `audio` in Colab files and upload your 20 recording files.
   - Upload `ground_truth.csv` and `asr_benchmark.py` directly to the `/content` workspace.
4. Add a cell at the top of the notebook to install dependencies:
   ```python
   !pip install -q jiwer rapidfuzz openai-whisper transformers torch soundfile requests
   ```
5. Set the API keys and run the pipeline:
   ```python
   import os
   os.environ["DEEPGRAM_API_KEY"] = "0c386e1cf781eaa9acac2e83a9f5878c4d66f818"
   os.environ["SARVAM_API_KEY"] = "sk_1lnojldo_khBQP8miRGWEM7GFr2dptSyx"

   # Run the script
   %run asr_benchmark.py
   ```
6. Download the resulting `results.csv` from Colab files for your report.
