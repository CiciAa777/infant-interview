import pandas as pd
import numpy as np
import whisper
import csv 
import os 


def transcribe_audio(audio_path, output_dir):
    # run whisper model 
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result['segments']

    # save transcript with timestamp as csv
    with open(output_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "text"])
        for seg in segments:
            writer.writerow([seg['start'], seg['end'], seg['text']])

if __name__ == "__main__":
    # file directory
    audio_file = "input/1minaudio.m4a"
    output_dir= "output/audio_transcript.csv"

    # run function
    transcribe_audio(audio_file, output_dir)
    print(f"Transcript saved")
    