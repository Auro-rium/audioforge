from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from audioforge.inference.predict_event import EventPredictor


def create_app(
    *,
    checkpoint_path: str | Path | None = None,
    label_map_path: str | Path | None = None,
    device: str = "auto",
) -> FastAPI:
    """Create the AudioForge HTTP API.

    Model loading is explicit at app creation time so startup fails clearly if
    deployment artifacts are missing, instead of returning opaque request errors.
    """
    checkpoint = checkpoint_path or os.environ.get("AUDIOFORGE_EVENT_CHECKPOINT")
    label_map = label_map_path or os.environ.get("AUDIOFORGE_LABEL_MAP")
    predictor = None
    if checkpoint and label_map:
        predictor = EventPredictor(checkpoint, label_map, device=device)

    app = FastAPI(title="AudioForge API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "model_loaded": predictor is not None,
        }

    @app.post("/predict/event")
    async def predict_event(file: UploadFile = File(...), top_k: int = 5) -> dict:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Event model is not configured")
        if top_k <= 0 or top_k > 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(await file.read())
                temporary.flush()
                predictions = predictor.predict(temporary.name, top_k=top_k)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "filename": file.filename,
            "predictions": [item.__dict__ for item in predictions],
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("audioforge.serving.api:app", host="0.0.0.0", port=8000)
