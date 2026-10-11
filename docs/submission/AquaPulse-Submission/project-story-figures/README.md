# AquaPulse project diagrams

1. **Observation workflow:** From a resident's photo to analysis, a saved observation, community verification, and follow-up.
2. **Training and evaluation:** Dataset cleaning, grouped splits, YOLO training, held-out evaluation, and ONNX export for CPU inference.
3. **Application and deployment:** HTTPS on AWS EC2 connects the web app, local ONNX models, PostgreSQL/PostGIS, and external reference services.
4. **Evidence and human verification:** Model findings, scene descriptions, and field answers combine into indicators and a human review route.
