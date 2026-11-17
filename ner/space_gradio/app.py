import gradio as gr
from transformers import pipeline

ner = pipeline("token-classification", model="quorumbio/biomed-ner-chem-disease", aggregation_strategy="simple")

def run(text):
    ents = ner(text) if text.strip() else []
    return {"text": text, "entities": [{"entity": e["entity_group"], "start": e["start"], "end": e["end"]} for e in ents]}

demo = gr.Interface(
    run, gr.Textbox(lines=6, label="Biomedical text (English)"), gr.HighlightedText(label="Chemical and Disease mentions"),
    examples=[["Naloxone reverses the antihypertensive effect of clonidine."],
              ["Lidocaine-induced cardiac asystole developed in a patient with chronic renal failure."]],
    title="Quorum Bio: biomedical NER (Chemical, Disease)",
    description="Research demo. English only. Not for clinical decisions.")
demo.launch()
