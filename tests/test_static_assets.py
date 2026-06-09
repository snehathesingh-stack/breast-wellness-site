import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_uses_vercel_api_route():
    for filename in ["index.html", "breast-cancer-prediction.html"]:
        html = (ROOT / filename).read_text(encoding="utf-8")
        assert 'const API_ENDPOINT = "/api/predict";' in html
        assert "AI cloud check" in html
        assert "AWS-ready" not in html


def test_model_report_loads_report_json():
    html = (ROOT / "model-report.html").read_text(encoding="utf-8")
    assert 'fetch("ml/model_report.json")' in html
    assert "Confusion Matrix" in html
    assert "Top Feature Weights" in html


def test_html_script_blocks_are_present():
    for filename in ["index.html", "breast-cancer-prediction.html", "model-report.html"]:
        html = (ROOT / filename).read_text(encoding="utf-8")
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
        assert scripts, f"{filename} should include a script block"
