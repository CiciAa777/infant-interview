# Audio Transcription and Video Object Detection

This project performs two main tasks :

1. **Audio Transcription**: Uses OpenAI Whisper to transcribe audio clips and save the transcript with timestamps as CSV.  
2. **Object Detection**: Uses YOLOv8 to detect objects in a video, outputting an annotated video and a CSV with detection details.

---

## Setup and run

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# dependency
pip install -r requirements.txt

# Run code
python src/transcribe.py
python src/detect_objects.py

```
Results are saved in the output folder.
