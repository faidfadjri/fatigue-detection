# Facial Landmark Recognition Model for Fatigue Detection

This project demonstrates the use of facial landmark recognition to detect fatigue through eye aspect ratios and other indicators. Follow the steps below to set up the environment and run the model.

## Prerequisites

1. **Download the Model**  
   Download the pre-trained facial landmark model `shape_predictor_68_face_landmarks.dat` from this repository:  
   [Download shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/tree/master).

2. **Extract and Move the Model**  
   After downloading, extract the ZIP file and copy `shape_predictor_68_face_landmarks.dat` into the project folder.

## Setting Up the Environment

### A. Create a Virtual Environment

**For Windows / macOS:**

````bash
python -m venv env


Windows / Mac

```bash
    python -m venv env
````

Linux

```bash
    python3 -m venv venv
```

### B. Run the virtual environment

Windows

```bash
    env/Scripts/activate
```

Linux / Mac

```bash
    source env/bin/activate
```

### C. Install Dependencies

```bash
    pip install -r requirements.txt
```

### D. Run the program 🚀

```bash
    python fatigue-detector.py
```
