"""Build the judge attachment from an explicit set of public project artifacts."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'docs' / 'submission'
BUNDLE = OUTPUT / 'AquaPulse-Submission'
ARCHIVE = OUTPUT / 'AquaPulse-Submission-Package.zip'
LIVE = 'https://aquapulse.3.148.215.37.sslip.io'
REPO = 'https://github.com/khaluc/AquaPulse'
CONTACT = 'nkluc2212@gmail.com'
FIGURES = [
    ('01-observation-pipeline.png', 'Observation workflow',
     "From a resident's photo to analysis, a saved observation, community verification, and follow-up."),
    ('02-training-and-evaluation.png', 'Training and evaluation',
     'Dataset cleaning, grouped splits, YOLO training, held-out evaluation, and ONNX export for CPU inference.'),
    ('03-system-architecture.png', 'Application and deployment',
     'HTTPS on AWS EC2 connects the web app, local ONNX models, PostgreSQL/PostGIS, and external reference services.'),
    ('04-evidence-and-verification.png', 'Evidence and human verification',
     'Model findings, scene descriptions, and field answers combine into indicators and a human review route.'),
]

CSS = '''
body{max-width:1100px;margin:0 auto;padding:32px 24px 60px;font:16px/1.65 Arial,sans-serif;color:#202020;background:#fff}
h1,h2,h3{line-height:1.25}h1{font-size:34px}h2{margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:10px}
a{color:#135c91}nav{display:flex;flex-wrap:wrap;gap:20px;border-bottom:1px solid #ddd;padding:12px 0}
.intro{max-width:850px}.note{padding:16px;background:#fff9dd;border-left:4px solid #b69e37}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}
figure{margin:0;border:1px solid #ddd;padding:14px;background:white}figure img{width:100%;height:260px;object-fit:contain}
figcaption{font-size:14px;margin-top:10px}figcaption strong{display:block;font-size:16px;margin-bottom:6px}
.story img{display:block;max-width:100%;height:auto;margin:24px auto;max-height:740px;object-fit:contain}
table{border-collapse:collapse;max-width:100%;font-size:14px}th,td{border:1px solid #ccc;padding:9px;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:16px}code{font-family:Consolas,monospace}
.links{display:flex;gap:16px;flex-wrap:wrap}footer{margin-top:48px;border-top:1px solid #ddd;padding-top:16px;font-size:14px}
@media print{nav{display:none}body{padding:0;font-size:11px}.grid{display:block}figure{break-inside:avoid;margin-bottom:20px}}
'''


def inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',
                     r'<img src="\2" alt="\1" loading="lazy">', escaped)
    escaped = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)


def markdown(text: str) -> str:
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            blocks.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
            i += 1
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading:
            level = len(heading[1])
            blocks.append(f'<h{level}>{inline(heading[2])}</h{level}>')
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            table = '<table>'
            for n, row in enumerate(rows):
                tag = 'th' if n == 0 else 'td'
                table += '<tr>' + ''.join(f'<{tag}>{inline(cell)}</{tag}>' for cell in row) + '</tr>'
            blocks.append(table + '</table>')
            continue
        ordered = bool(re.match(r'^\d+\. ', line))
        if line.startswith('- ') or ordered:
            tag = 'ol' if ordered else 'ul'
            items = []
            while i < len(lines):
                item = lines[i].strip()
                match = re.match(r'^\d+\. (.+)', item) if ordered else re.match(r'^- (.+)', item)
                if not match:
                    break
                items.append('<li>' + inline(match[1]) + '</li>')
                i += 1
            blocks.append(f'<{tag}>' + ''.join(items) + f'</{tag}>')
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            if re.match(r'^(?:#|\||- |\d+\. |```)', lines[i].strip()):
                break
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    return '\n'.join(blocks)


def page(title: str, content: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
            '<nav><a href="index.html">Review guide and gallery</a>'
            '<a href="project-story.en.html">Full project story</a>'
            f'<a href="{LIVE}/?lang=en">Live website</a><a href="{REPO}">GitHub</a></nav>'
            + content + f'<footer>AquaPulse · Contact: <a href="mailto:{CONTACT}">{CONTACT}</a>'
            '</footer></body></html>')


def copy(source: Path, relative: str) -> None:
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def gallery(items: list[dict], directory: str) -> str:
    cards = []
    for number, item in enumerate(items, 1):
        path = f'{directory}/{item["file"]}'
        title, caption = html.escape(item['title']), html.escape(item['caption'])
        cards.append(f'<figure><a href="{path}" target="_blank"><img src="{path}" '
                     f'alt="{title}" loading="lazy"></a><figcaption><strong>{number:02d}. {title}</strong>'
                     f'{caption}</figcaption></figure>')
    return '<div class="grid">' + ''.join(cards) + '</div>'


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    screenshots_dir = ROOT / 'docs/interface-screenshots'
    screenshot_manifest = json.loads((screenshots_dir / 'manifest.json').read_text(encoding='utf-8'))
    screenshots = screenshot_manifest['screenshots']
    assert len(screenshots) == 10
    for item in screenshots:
        source = screenshots_dir / item['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
        copy(source, 'interface-screenshots/' + item['file'])
    for name in ('README.md', 'manifest.json'):
        copy(screenshots_dir / name, 'interface-screenshots/' + name)

    diagrams = []
    for name, title, caption in FIGURES:
        copy(ROOT / 'docs/project-story-figures' / name, 'project-story-figures/' + name)
        diagrams.append({'file': name, 'title': title, 'caption': caption})
    diagram_notes = '# AquaPulse project diagrams\n\n' + '\n'.join(
        f'{i}. **{d["title"]}:** {d["caption"]}' for i, d in enumerate(diagrams, 1)) + '\n'
    (BUNDLE / 'project-story-figures/README.md').write_text(diagram_notes, encoding='utf-8')

    story = (ROOT / 'docs/project-story.en.md').read_text(encoding='utf-8')
    (BUNDLE / 'project-story.en.md').write_text(story, encoding='utf-8')
    (BUNDLE / 'project-story.en.html').write_text(
        page('AquaPulse — full project story', '<article class="story"><h1>AquaPulse</h1>'
             + markdown(story) + '</article>'), encoding='utf-8')
    research_pdf = OUTPUT / 'AquaPulse-Research-Report.pdf'
    if research_pdf.exists():
        copy(research_pdf, 'research-report/AquaPulse-Research-Report.pdf')
        copy(OUTPUT / 'AquaPulse-Research-Report.en.md', 'research-report/AquaPulse-Research-Report.en.md')

    evidence_files = {
        'ml/runs/n_clean/eval/report.json': 'model-evidence/detector-test-report.json',
        'ml/runs/n_clean/results.csv': 'model-evidence/detector-training-results.csv',
        'ml/runs/n_clean/args.yaml': 'model-evidence/detector-training-args.yaml',
        'ml/runs/water_seg_s/results.csv': 'model-evidence/water-training-results.csv',
        'ml/runs/water_seg_s/args.yaml': 'model-evidence/water-training-args.yaml',
        'ml/runs/s_clean/eval/report.json': 'model-evidence/detector-small-test-report.json',
        'ml/runs/s_clean/results.csv': 'model-evidence/detector-small-training-results.csv',
        'ml/runs/s_clean/args.yaml': 'model-evidence/detector-small-training-args.yaml',
        'ml/runs/s_stock/eval/report.json': 'model-evidence/detector-stock-test-report.json',
        'ml/runs/s_stock/results.csv': 'model-evidence/detector-stock-training-results.csv',
        'ml/runs/s_stock/args.yaml': 'model-evidence/detector-stock-training-args.yaml',
        'ml/runs/water_seg/results.csv': 'model-evidence/water-nano-training-results.csv',
        'ml/runs/water_seg/args.yaml': 'model-evidence/water-nano-training-args.yaml',
    }
    for source, destination in evidence_files.items():
        copy(ROOT / source, destination)
    report = json.loads((BUNDLE / 'model-evidence/detector-test-report.json').read_text())
    metrics = report['test_box_overall']
    with (BUNDLE / 'model-evidence/detector-training-results.csv').open(newline='') as stream:
        detector_epochs = len(list(csv.DictReader(stream)))
    with (BUNDLE / 'model-evidence/water-training-results.csv').open(newline='') as stream:
        water_epochs = len(list(csv.DictReader(stream)))
    evidence = f'''# Saved model evidence

These files are existing training and evaluation outputs, copied without rerunning training.

- Clean YOLO11n detector: {detector_epochs} completed training epochs; validation-selected threshold {report['threshold']}.
- Held-out test: {report['test_images']} images and {metrics['gt']} annotated objects; precision {metrics['precision']:.2%}, recall {metrics['recall']:.2%}, F1 {metrics['f1']:.2%}.
- YOLO11s-seg water model: {water_epochs} completed training epochs. Its CSV records training/validation metrics, not an independent field evaluation.
- The detector test has no background-only images; image-level false-alarm rate is unknown. The tyre class has only three test objects.
- `args.yaml` paths refer to the original Linux training container. Model checkpoints and data preparation scripts are documented in the GitHub repository.
- The gallery's tyre photo comes from the training split and demonstrates the interface. The test report uses the held-out test split.
- The clean small detector and stock-inclusive detector reports, configurations, and logs are included for the research report's comparison. The stock-inclusive experiment uses a different test split.
- Nano water training logs and configuration are also included. Both water CSVs contain validation metrics; neither is an independent test report.
'''
    (BUNDLE / 'model-evidence/README.md').write_text(evidence, encoding='utf-8')

    guide = f'''# AquaPulse — judge review guide

AquaPulse connects residents' river and canal observations with photo analysis, maps, human verification, and community follow-up. I built it in Vietnam, with an initial focus on urban waterways in Ho Chi Minh City.

## Start here

- Extract this ZIP, then open **index.html** for the guide, four diagrams, and ten interface screenshots. Images open at full resolution when clicked. The attachment can be browsed offline.
- Live website: {LIVE}/?lang=en
- Source code and setup: {REPO}
- Verifier access and questions: {CONTACT}
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

Validation threshold: {report['threshold']}. Held-out test: {report['test_images']} images / {metrics['gt']} objects.
Precision: {metrics['precision']:.2%}; box recall: {metrics['recall']:.2%}; F1: {metrics['f1']:.2%}.
The raw test report and training logs are in model-evidence/. See its README for the evaluation limits.

## Reading the demo screenshots

The yellow sample-data banner identifies simulated reports, reporters, and activities from the local demo. Screenshot 02 reuses an existing successful Kimi response. Screenshot 04 shows one YOLO tyre detection on a training-split photo beside the water; the segmenter returns no water masks for that image. It illustrates the interface and is separate from held-out evaluation.

The current live verifier form uses a sign-in code and displays {CONTACT} as the contact address. The older mission screenshot has the same sign-in flow with an earlier helper sentence. There is no email signup or automatic emailed login-code feature in the current application.

The project displays environmental observations and preliminary indicators. Water safety and contamination require appropriate sampling and laboratory assessment; demo activity is not evidence of real environmental improvement.

## Reproduce the local sample-data walkthrough

Clone the repository and follow README.en.md to install the backend and start the local PostGIS database. Then, from the repository root in Windows PowerShell:

```powershell
docker compose up -d db
backend/.venv/Scripts/python.exe backend/scripts/seed_local_demo.py
backend/.venv/Scripts/python.exe backend/scripts/run_local_demo.py
```

Open http://localhost:8008/?demo=1&lang=en. Local demo sign-in codes: `local-demo-volunteer` (volunteer), `local-demo-expert` (expert). These codes belong to the local demo. Live verifier access is issued separately by the coordinator.

See {REPO}/blob/main/docs/local-demo.md for local setup details. Image uploads use Kimi when the server has a configured API key.

## Attachment contents

- index.html: offline guide and galleries.
- project-story.en.html / .md: complete project story, including data preparation, model training, evaluation, API integration, and deployment.
- project-story-figures/: four PNG diagrams and captions.
- interface-screenshots/: ten English screenshots, captions, and provenance manifest.
- model-evidence/: three detector test reports and training configurations/CSV logs for all five research-report experiments.
- research-report/: illustrated research PDF and its English manuscript.
- manifest.json: file sizes and SHA-256 checksums.
'''
    (BUNDLE / 'README.md').write_text(guide, encoding='utf-8')

    content = ('<h1>AquaPulse — review guide</h1><div class="intro">'
               '<p>A photo, a location, and a clear route to human follow-up for urban rivers and canals.</p>'
               f'<p class="links"><a href="{LIVE}/?lang=en">Open live website</a>'
               f'<a href="{REPO}">Source code</a><a href="mailto:{CONTACT}">Request verifier access</a></p>'
               '<p><a href="research-report/AquaPulse-Research-Report.pdf">Read the illustrated research report (PDF)</a></p>'
               '<p class="note">The screenshots below use labeled sample data. The live website may have fewer '
               'observations and missions. Protected verification actions require a coordinator-issued code.</p>'
               '<h2>How to review</h2><ol><li>Explore the live home map and a canal passport.</li>'
               '<li>Try photo analysis in Send Observation and switch VI / EN.</li>'
               '<li>Review community tasks and dashboard metrics.</li>'
               '<li>Read the full project story for training, evaluation, API integration, and deployment details.</li></ol>'
               f'<p>Saved YOLO11n test: {report["test_images"]} images, {metrics["gt"]} objects; '
               f'precision {metrics["precision"]:.2%}, recall {metrics["recall"]:.2%}, F1 {metrics["f1"]:.2%}. '
               '<a href="model-evidence/README.md">Evaluation evidence and limits</a>.</p></div>'
               '<h2>Four project diagrams</h2>' + gallery(diagrams, 'project-story-figures')
               + '<h2>Ten interface screenshots</h2>' + gallery(screenshots, 'interface-screenshots')
               + '<h2>Photo-analysis provenance</h2><p>Screenshot 02 uses a cached successful Kimi description. '
               'Screenshot 04 runs YOLO on a training-split photo of discarded tyres at the water\'s edge: '
               'one tyre is detected and no water masks are returned. It demonstrates the interface, '
               'rather than held-out model performance.</p><h2>Local reproduction and access</h2>'
               '<p><a href="README.md">Read the complete guide</a> for local setup, local demo sign-in codes, '
               'package contents, and the distinction between demo and live data.</p>'
               f'<p>Live sign-in codes: email <a href="mailto:{CONTACT}">{CONTACT}</a>.</p>')
    (BUNDLE / 'index.html').write_text(page('AquaPulse — submission attachment', content), encoding='utf-8')

    files = sorted(path for path in BUNDLE.rglob('*') if path.is_file() and path != BUNDLE / 'manifest.json')
    manifest = {'project': 'AquaPulse', 'screenshots': 10, 'diagrams': 4, 'files': [
        {'file': path.relative_to(BUNDLE).as_posix(), 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]}
    (BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file():
                assert path.suffix.lower() not in {'.pem', '.env', '.pt', '.onnx', '.zip'}
                archive.write(path, path.relative_to(OUTPUT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith('.png')]) == 14
    assert ARCHIVE.stat().st_size < 35_000_000
    print(json.dumps({'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
                      'size_mb': round(ARCHIVE.stat().st_size / 1_000_000, 2),
                      'files': len(manifest['files']) + 1, 'screenshots': 10, 'diagrams': 4}))


if __name__ == '__main__':
    main()
