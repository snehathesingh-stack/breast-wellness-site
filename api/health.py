import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        model = load_model()
        payload = {
            "status": "ok",
            "service": "breast-wellness-ml",
            "platform": "vercel",
            "model": {
                "available": True,
                "type": model.get("model_type"),
                "version": model.get("version"),
                "metrics": model.get("metrics"),
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def load_model():
    path = Path(__file__).resolve().parents[1] / "aws" / "model.json"
    return json.loads(path.read_text(encoding="utf-8"))
