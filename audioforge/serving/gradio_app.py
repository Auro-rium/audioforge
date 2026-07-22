from __future__ import annotations

import os

from audioforge.inference.predict_event import EventPredictor


def create_demo():
    """Build the optional Gradio event-classification UI."""
    import gradio as gr

    checkpoint = os.environ.get("AUDIOFORGE_EVENT_CHECKPOINT")
    label_map = os.environ.get("AUDIOFORGE_LABEL_MAP")
    if not checkpoint or not label_map:
        raise RuntimeError("Set AUDIOFORGE_EVENT_CHECKPOINT and AUDIOFORGE_LABEL_MAP")
    predictor = EventPredictor(checkpoint, label_map)

    def classify(audio):
        predictions = predictor.predict(audio, top_k=10)
        return {item.label: item.score for item in predictions}

    return gr.Interface(
        fn=classify,
        inputs=gr.Audio(type="filepath"),
        outputs=gr.Label(num_top_classes=10),
        title="AudioForge environmental sound classifier",
    )


if __name__ == "__main__":
    create_demo().launch()
