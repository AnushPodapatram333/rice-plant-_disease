# OryzaScan — Rice Leaf Pathology Scanner

A local web app for the rice leaf disease CNN from your notebook. Upload or
photograph a leaf and it classifies it as **Bacterial Blight**, **Brown
Spot**, or **Leaf Smut**, with a full breakdown UI: confidence dial,
probability ledger, a pathology reference section, and real training-curve
charts pulled straight from your notebook's own logs.

Flask backend + a hand-built HTML/CSS/JS frontend. No React build step, no
external services — everything runs on `localhost`.

---

## ⚠️ One thing to do first: export the actual model

I read through the notebook closely, and found something worth flagging:
**it only ever saves the *first*, simpler model** —

```python
model.save('riceplantdetectionmodel.h5', include_optimizer=True)   # cell 10
```

That's the version *before* data augmentation, dropout, and early stopping
were added — it peaks at **95.2%** validation accuracy. The model you
mentioned (**97.4%**, with dropout + early stopping) is `model_augmented`,
trained a few cells later — but it's only ever kept in the notebook's
memory. There's no `model_augmented.save(...)` line anywhere, so that model
disappears the moment the notebook session ends.

**Fix:** open the notebook, go to the cell right after `model_augmented`
finishes training (after `history_augmented = model_augmented.fit(...)`),
and add:

```python
model_augmented.save('rice_disease_model.h5')
```

Run that cell, then download `rice_disease_model.h5` from the Colab/Kaggle
file browser and drop it into this project's `model/` folder. That's it —
`app.py` picks up any `.h5` or `.keras` file in `model/` automatically.

**Want to test the app immediately instead?** Your notebook already saves
`riceplantdetectionmodel.h5` in cell 10. Copy that into `model/` and the app
will run right now — just with the earlier, ~95%-accuracy model until you
swap in the real one. Both save to the same input/output shape, so the app
doesn't need to know which one it's running.

> A heads-up on file size: because of the large Dense(128) layer after
> flattening, `model_augmented` is roughly **43.7M parameters (~166 MB)**
> saved — vs. ~1.7M params (~6.5 MB) for the basic model. Totally fine to
> run locally; just don't be surprised by the file size.

---

## Setup

```bash
# from inside this folder
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# now place your exported model file (see above) into model/

python app.py
```

Open **http://localhost:5000** in your browser.

The header shows a status pill — green means a model loaded successfully;
if it's red, check that a `.h5`/`.keras` file is actually sitting in
`model/` and re-check the terminal for the exact error (a corrupted file,
wrong TensorFlow version, etc. will print there).

---

## Project structure

```
rice_disease_app/
├── app.py                  Flask backend + prediction API
├── requirements.txt
├── model/                  ← put your exported .h5 / .keras file here
├── templates/
│   └── index.html
└── static/
    ├── css/style.css       all visual design
    └── js/app.js           upload, camera, predict call, charts
```

## API endpoints, if you want to extend it

- `GET  /api/status` — whether a model is loaded, its filename, class list
- `GET  /api/model-info` — pathology reference text + real training-history
  arrays used to draw the calibration charts
- `POST /api/predict` — send `multipart/form-data` with an `image` file,
  get back `{predicted_class, display_name, confidence, probabilities,
  case_id}`

## Notes on the preprocessing

The model's own first layer is `Rescaling(1./255)`, matching how
`image_dataset_from_directory` feeds it during training (raw 0–255 pixel
values, resized to 300×300, **not** pre-normalized). `app.py` mirrors this
exactly — if you retrain with different preprocessing, update
`preprocess_image()` in `app.py` to match, and double-check `CLASS_NAMES`
still matches your training run's alphabetical folder order.

## Disclaimer

This is an educational/demo tool built around a student CNN project. The
pathology notes are general background, not field-specific treatment
guidance — for real diagnosis and treatment decisions, consult a local
agricultural extension service or plant pathologist.
