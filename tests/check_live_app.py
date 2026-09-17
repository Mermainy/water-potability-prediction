"""Optional live smoke check: submit Streamlit's form to the running Docker API."""
import os
from pathlib import Path
import sys

import requests
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "code"))
api_url = os.getenv("SMOKE_API_URL", "http://localhost:8000")
os.environ["API_URL"] = api_url
health_response = requests.get(f"{api_url}/health", timeout=15)
health_response.raise_for_status()
health = health_response.json()
ui = AppTest.from_file(str(root / "code/deployment/app/main.py"))
ui.run(timeout=30)
assert not ui.exception, ui.exception
assert len(ui.number_input) == 9
ui.button[0].click().run(timeout=30)
assert not ui.exception, ui.exception
assert not ui.error, "The app could not obtain a prediction from the API"
assert ui.subheader[0].value in {"Potable", "Not potable"}
assert any(caption.value == f"MLflow run: {health['run_id']}" for caption in ui.caption)
print(f"Streamlit form -> Docker API -> displayed prediction: {ui.subheader[0].value}")
print(f"Serving MLflow run: {health['run_id']}")
