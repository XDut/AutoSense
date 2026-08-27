# AutoSense - Smart Vehicle Detector

AutoSense combines real-time vehicle detection using YOLOv12 with intelligent analysis powered by gemini-3.5-flash-lite (VLM). The system detects vehicles in a video stream and identifies vehicle type, color, manufacturer, model, estimated year built, country.

## Demo

![demo](Demo.gif)

## Features

- Real-time detection of cars, trucks, buses, motorcycles using YOLO.
- Persistent object tracking with unique track IDs.
- Automatic vehicle cropping from clean frame.
- Country detection for vehicle analysis based on geography.

## Sample JSON Output

{  
    "timestamp": "2026-08-27T21:36:35.059+05:30",  
    "track_id": 3,  
    "vehicle_type": "car",  
    "vehicle_color": "black and yellow",  
    "vehicle_company": "Hyundai",  
    "vehicle_model": "Santro",  
    "estimated_year_built": "2003-2005",  
    "country": "India",  
    "country_confidence": 1.0,  
    "detector_class": "car",  
    "analysis_model": "gemini-3.5-flash-lite",  
    "analysis_status": "complete",  
    "crop_path": "cropped_vehicles/vehicle_3_2026-08-27T21-36-35.059+05-30.jpg"  
  }

## Installation

```bash
git clone https://github.com/XDut/AutoSense.git
cd AutoSense
python smart_detector.py
```
