# AquaPulse — judge review guide

AquaPulse connects residents' river and canal observations with photo analysis, maps, human verification, and community follow-up. I built it in Vietnam, with an initial focus on urban waterways in Ho Chi Minh City.

## Start here

- Extract this ZIP, then open **index.html** for the guide, four diagrams, and ten interface screenshots. Images open at full resolution when clicked. The attachment can be browsed offline.
- Live website: https://aquapulse.3.148.215.37.sslip.io/?lang=en
- Source code and setup: https://github.com/khaluc/AquaPulse
- Verifier access and questions: nkluc2212@gmail.com
- Full project story: project-story.en.html (readable version) and project-story.en.md (Markdown source).
- Research report: research-report/AquaPulse-Research-Report.pdf, with its editable English manuscript alongside it.

## Suggested live walkthrough

1. Home: explore the river map and select a canal stretch.
2. Canal passport: inspect the evidence, history, rainfall context, and confidence.
3. Send observation: select a reporting mode, upload a photo, answer the questions, and review the analysis. Save it if you want to contribute a real observation.
4. Community Tasks: inspect public missions. Verification and expert actions require a coordinator-issued sign-in code; request access by email.
5. Dashboard: inspect reporting and verification metrics. The live site may contain fewer records than the sample-data screenshots.
6. Switch VI / EN: the report form keeps the photo, answers, and analysis while the language changes.

## What runs locally and what uses an API

- I fine-tuned YOLO11n for five container classes and YOLO11s-seg for water inside containers. The backend runs ONNX exports on the CPU.
- Kimi kimi-k3 is accessed through Alibaba Cloud Model Studio's OpenAI-compatible endpoint for scene descriptions. I did not train Kimi.
- A rule engine combines the available findings and reporter answers, keeps risk separate from confidence, and routes cases to human review.
- FastAPI, PostgreSQL/PostGIS, Leaflet, Docker Compose, and Caddy support the application. The public instance runs on Amazon Linux AWS EC2 with HTTPS.

## Saved detector evidence

Validation threshold: 0.35. Held-out test: 167 images / 316 objects.
Precision: 75.43%; box recall: 82.59%; F1: 78.85%.
The raw test report and training logs are in model-evidence/. See its README for the evaluation limits.

## Reading the demo screenshots

The yellow sample-data banner identifies simulated reports, reporters, and activities from the local demo. Screenshot 02 reuses an existing successful Kimi response. Screenshot 04 shows one YOLO tyre detection on a training-split photo beside the water; the segmenter returns no water masks for that image. It illustrates the interface and is separate from held-out evaluation.

The current live verifier form uses a sign-in code and displays nkluc2212@gmail.com as the contact address. The older mission screenshot has the same sign-in flow with an earlier helper sentence. There is no email signup or automatic emailed login-code feature in the current application.

The project displays environmental observations and preliminary indicators. Water safety and contamination require appropriate sampling and laboratory assessment; demo activity is not evidence of real environmental improvement.

## Reproduce the local sample-data walkthrough

Clone the repository and follow README.en.md to install the backend and start the local PostGIS database. Then, from the repository root in Windows PowerShell:

```powershell
docker compose up -d db
backend/.venv/Scripts/python.exe backend/scripts/seed_local_demo.py
backend/.venv/Scripts/python.exe backend/scripts/run_local_demo.py
```

Open http://localhost:8008/?demo=1&lang=en. Local demo sign-in codes: `local-demo-volunteer` (volunteer), `local-demo-expert` (expert). These codes belong to the local demo. Live verifier access is issued separately by the coordinator.

See https://github.com/khaluc/AquaPulse/blob/main/docs/local-demo.md for local setup details. Image uploads use Kimi when the server has a configured API key.

## Attachment contents

- index.html: offline guide and galleries.
- project-story.en.html / .md: complete project story, including data preparation, model training, evaluation, API integration, and deployment.
- project-story-figures/: four PNG diagrams and captions.
- interface-screenshots/: ten English screenshots, captions, and provenance manifest.
- model-evidence/: three detector test reports and training configurations/CSV logs for all five research-report experiments.
- research-report/: illustrated research PDF and its English manuscript.
- manifest.json: file sizes and SHA-256 checksums.
