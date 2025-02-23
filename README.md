# Pest Classification API and GUI

This project provides a FastAPI-based backend (`app.py`) and a Streamlit-based GUI (`app-gui.py`) for classifying rice pests using a trained machine learning model (`model.keras`).

## Prerequisites

Ensure you have Python installed (recommended: Python 3.9+). You also need to install the required dependencies.

### Install Dependencies

```sh
pip install -r requirements.txt
```

## Running the API Server

To start the FastAPI server, navigate to the `model-api/main` folder and run:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

- The API will be accessible at `http://127.0.0.1:8000`

## Running the GUI

To start the Streamlit-based GUI, run:

```sh
streamlit run app-gui.py
```

This will launch the web interface where you can upload images for classification.


## Folder Structure

```
model-api/
│── main/
│   ├── app.py          # FastAPI backend
│   ├── app-gui.py      # Streamlit GUI
│   ├── model.keras     # Trained model
│   ├── requirements.txt # Dependencies
│   ├── train.py        # Model training script
```


