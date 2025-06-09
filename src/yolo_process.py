import pandas as pd
import numpy as np
from ultralytics import YOLO
import csv
import os


def run_yolo_detection(video_path, output_dir):
    # Run model
    model = YOLO("yolov8n.pt") 
    # annotated video saved in output
    results = model(video_path, save=True, project="output", name="annotated_video")
    
    # Write detection results to CSV
    with open(output_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "object_class", "confidence_score", "x1", "y1", "x2", "y2"])

        for frame_idx, r in enumerate(results):
            boxes = r.boxes
            for box, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.conf.cpu().numpy(),
                                      boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box
                writer.writerow([frame_idx, int(cls), float(conf), x1, y1, x2, y2])

if __name__ == "__main__":
    # directory 
    video_file = "input/1minvideo.mp4"
    output_dir = "output/yolo_results.csv"
    run_yolo_detection(video_file, output_dir)
    print(f"results saved")

