import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        wellness_model = load_model("model.json")
        diagnostic_model = load_model("diagnostic_model.json")
        payload = {
            "status": "ok",
            "service": "breast-wellness-ml",
            "platform": "vercel",
            "model": {
                "available": True,
                "type": wellness_model.get("model_type"),
                "version": wellness_model.get("version"),
                "metrics": wellness_model.get("metrics"),
            },
            "diagnostic_model": {
                "available": True,
                "type": diagnostic_model.get("model_type"),
                "version": diagnostic_model.get("version"),
                "metrics": diagnostic_model.get("metrics"),
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def load_model(filename):
    path = Path(__file__).resolve().parents[1] / "aws" / filename
    return json.loads(path.read_text(encoding="utf-8"))
