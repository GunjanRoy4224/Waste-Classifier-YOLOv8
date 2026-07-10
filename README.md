# Team Zero Waste - Waste Classification using YOLOv8

An intelligent waste classification and detection system powered by **YOLOv8** and a **Streamlit** web application. This project is built to identify and categorize waste objects using deep learning, with support for various inputs including images and live webcam streams.

## Features

- **Waste Detection**: Accurately classifies and creates bounding boxes around different types of waste in real-time.
- **Multiple Input Sources**: Supports waste detection on uploaded images and live webcam streams.
- **YOLOv8 Integration**: Uses the state-of-the-art YOLOv8 object detection model by Ultralytics.
- **Interactive UI**: An intuitive, easy-to-use frontend built with Streamlit.
- **Data Preprocessing**: Scripts included to fetch the TACO dataset, parse COCO annotations into YOLO formats, and automatically create train/val/test splits.

## Project Structure

```text
Waste-Classifcation-YOLOv8-TZW/
│
├── streamlit - waste detection/ # Streamlit web application frontend and backend
│   ├── app.py                   # Main Streamlit application
│   ├── helper.py                # Helper functions for YOLO inference and rendering
│   ├── settings.py              # Configuration and path settings
│   ├── images/                  # Default and placeholder images
│   └── weights/                 # YOLOv8 model weights (.pt files)
│
├── preprocessing/               # Dataset preparation scripts
│   ├── coco_to_yolo.py          # Converts COCO JSON annotations to YOLO txt format
│   ├── download_teco_data.py    # Downloads the TACO image dataset via Flickr URLs
│   └── split_dataset.py         # Splits dataset into training, validation, and testing sets
│
├── notebook/                    # Jupyter Notebooks for model training and experimentation
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Waste-Classifcation-YOLOv8-TZW
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App

To launch the Streamlit frontend, navigate to the application directory and run the server:

```bash
cd "streamlit - waste detection"
streamlit run app.py
```

This will open the application in your default web browser at `http://localhost:8501`.

## Data Preprocessing (For Training)

If you intend to train your own YOLOv8 model on the TACO dataset, you can use the provided preprocessing scripts:

1. **Download Data**: Run `download_teco_data.py` to fetch images.
2. **Convert Annotations**: Run `coco_to_yolo.py` to translate COCO annotations to the YOLO format.
3. **Split Dataset**: Use `split_dataset.py` to divide your images into train, val, and test subsets.
