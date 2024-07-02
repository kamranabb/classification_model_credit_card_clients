# ---------------------------------------
# Libraries
# ---------------------------------------
import gradio as gr
import pickle
from utils.save_load_model import load_model
from utils.load_data import *


# ---------------------------------------
# Constants
# ---------------------------------------
MODEL_PATH = "./models/best_model.pkl"
XLS_PATH = "data/default of credit card clients.xls"
title = "Credit Card Default Prediction"
description = "Enter the details to correctly identify default of credit card clients"
# article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."


# ---------------------------------------
# Load the model
# ---------------------------------------
model = load_model(MODEL_PATH)

# ---------------------------------------
# Load the data
# ---------------------------------------
df = load_xls(XLS_PATH, verbose=False)

# ---------------------------------------
# Preprocess the data
# ---------------------------------------
features = ["LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Drop other columns except features
df = df[features]
# Fix type of PAY_0 - PAY_6
df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].astype(str)

# Get the min and max of Limit bal
min_limit_bal = int(df["LIMIT_BAL"].min())
max_limit_bal = int(df["LIMIT_BAL"].max())
# Get value counts of Pay 0 - pay 6
pay_0_counts = list(df["PAY_0"].value_counts().index)
pay_2_counts = list(df["PAY_2"].value_counts().index)
pay_3_counts = list(df["PAY_3"].value_counts().index)
pay_4_counts = list(df["PAY_4"].value_counts().index)
pay_5_counts = list(df["PAY_5"].value_counts().index)
pay_6_counts = list(df["PAY_6"].value_counts().index)

# ---------------------------------------
# Functions
# ---------------------------------------
def predict(limit_bal, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6):
    features = [limit_bal, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
    y_pred = str(model.predict([features])[0])
    label = f"Predicted Default: {y_pred}"
    return label


# ---------------------------------------
# Gradio Interface
# ---------------------------------------
inputs = [
    gr.Slider(min_limit_bal, max_limit_bal, step=1, label="LIMIT_BAL"),
    gr.Radio(pay_0_counts, label="PAY_0"),
    gr.Radio(pay_2_counts, label="PAY_2"),
    gr.Radio(pay_3_counts, label="PAY_3"),
    gr.Radio(pay_4_counts, label="PAY_4"),
    gr.Radio(pay_5_counts, label="PAY_5"),
    gr.Radio(pay_6_counts, label="PAY_6")
]
outputs = [gr.Label(num_top_classes=2)]

examples = df.head(5).values.tolist()


gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    # article=article,
    theme=gr.themes.Soft(),
).launch()








