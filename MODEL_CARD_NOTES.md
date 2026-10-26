# Model / engine notes

**Intended use:** education and care navigation. Not a diagnostic device; never a substitute for a clinician.

**Screening engine:** a hand-written rule base (9 conditions, 15 symptoms, 5 lab values). Confidence = points matched / points possible. It is **not a probability**, and has not been clinically validated. The browser and `/api/screen` produce identical results.

**Red flags:** simple rules (severe chest pain; chest discomfort with breathlessness; urgent phrases in free text). They can miss emergencies. Anyone worried should seek care regardless of the result.

**Lab-report parsing:** regex pattern matching with unit conversion. Always verify extracted values against the original report.

**Image screening:** no real model ships with this project. Without models in `models\`, scores are deterministic placeholders from the image pixels and carry **no clinical meaning** (responses say `DEMO MODE`). To use real models, place `<skin|eye|oncology>.onnx` and `<same>.labels.json` in `models\` and `pip install onnxruntime numpy`. That integration is untested; validate any model clinically before exposing results.

**Data:** stored unencrypted in `data\diagnova.sqlite3`. Images are processed in memory and not saved.
