# Civet Behavior Detector

Labels a civet's behavior (moving, eating, resting) in CCTV video and builds an observation sheet.

## Setup

```bash
pip install -r requirements.txt     # Python 3.12
```

## Run the app

```bash
python app.py
```

Open http://127.0.0.1:5001 and upload a video.

## Train on your own videos

1. Put your videos in `videos/`.
2. Run `notebooks/1_sam_labelling.ipynb` to label them with SAM.
3. Run `notebooks/2_train_yolo.ipynb` to train the model.
4. Restart the app.
