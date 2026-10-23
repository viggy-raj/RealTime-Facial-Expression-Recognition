"""
Live scores for a frontend. Does not replace run.bat.

Start from the AU folder:
    python -m facial_cues.api.server
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

AU_ROOT = Path(__file__).resolve().parents[2]
os.chdir(AU_ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv(AU_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facial_cues.config.schema import load_config
from facial_cues.pipeline.orchestrator import PipelineOrchestrator

_LINE = re.compile(r"^([A-Za-z]+)\s*(?:\(([0-9.]+)\))?\s*$")

pipeline: PipelineOrchestrator | None = None
_worker: threading.Thread | None = None


def _setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _url(path: str) -> str:
    base = _setting("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _split_line(text: str) -> dict:
    text = (text or "").strip()
    match = _LINE.match(text)
    if not match:
        return {"line": text, "label": "", "score": None}
    score = match.group(2)
    return {
        "line": text,
        "label": match.group(1).upper(),
        "score": float(score) if score is not None else None,
    }


def snapshot() -> dict:
    if pipeline is None:
        return {"ready": False, "deception": None, "emotion": None}
    deception_text = getattr(pipeline, "_last_deception_text", "") or ""
    reasons = list(getattr(pipeline, "_last_deception_reasons", []) or [])
    emotion_text = getattr(pipeline, "_last_emotion_text", "") or ""
    stored = getattr(pipeline, "_last_deception", {}) or {}
    latest = None
    if stored:
        latest = next(reversed(stored.values()))
    deception = _split_line(deception_text)
    deception["reasons"] = reasons
    deception["comment"] = reasons[0] if reasons else ""
    deception["p_lie"] = None if latest is None else latest.get("p_lie")
    deception["model_label"] = None if latest is None else latest.get("label")
    return {
        "ready": bool(deception_text or emotion_text),
        "deception": deception,
        "emotion": _split_line(emotion_text),
    }


def _run_camera() -> None:
    assert pipeline is not None
    show = _setting("API_VISUALIZE", "false").lower() in {"1", "true", "yes"}
    try:
        pipeline.run(visualize=show)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Camera loop stopped")


app = FastAPI(title="Facial cues live scores", version="1.0")

_origins = [item.strip() for item in _setting("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global pipeline, _worker
    config = load_config(None)
    config.deception.enabled = True
    if _setting("API_VISUALIZE", "false").lower() not in {"1", "true", "yes"}:
        config.visualization.enabled = False
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()
    _worker = threading.Thread(target=_run_camera, name="facial-cues-camera", daemon=True)
    _worker.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    if pipeline is not None:
        pipeline.shutdown()


def _index():
    return {
        "deception": _url(_setting("ENDPOINT_DECEPTION", "/deception")),
        "emotion": _url(_setting("ENDPOINT_EMOTION", "/emotion")),
        "live": _url(_setting("ENDPOINT_LIVE", "/live")),
        "health": _url(_setting("ENDPOINT_HEALTH", "/health")),
    }


def _health():
    alive = _worker is not None and _worker.is_alive()
    return {"status": "ok" if alive else "starting", "camera_running": alive}


def _live():
    return snapshot()


def _deception():
    body = snapshot()
    return {"ready": body["ready"], **(body["deception"] or {})}


def _emotion():
    body = snapshot()
    return {"ready": body["ready"], **(body["emotion"] or {})}


app.add_api_route(_setting("ENDPOINT_INDEX", "/"), _index, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_HEALTH", "/health"), _health, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_LIVE", "/live"), _live, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_DECEPTION", "/deception"), _deception, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_EMOTION", "/emotion"), _emotion, methods=["GET"])


def main() -> None:
    import uvicorn

    uvicorn.run(
        "facial_cues.api.server:app",
        host=_setting("API_HOST", "0.0.0.0"),
        port=int(_setting("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
