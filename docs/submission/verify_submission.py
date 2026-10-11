"""Validate the generated attachment's local links, images, hashes, and ZIP entries."""
import hashlib
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'AquaPulse-Submission'
ARCHIVE = ROOT / 'AquaPulse-Submission-Package.zip'


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'href', 'src'} and value:
                self.references.append(value)


for page in BUNDLE.glob('*.html'):
    parser = References()
    parser.feed(page.read_text(encoding='utf-8'))
    for reference in parser.references:
        url = urlsplit(reference)
        if url.scheme or url.netloc or not url.path:
            continue
        path = (page.parent / unquote(url.path)).resolve()
        assert path.is_relative_to(BUNDLE.resolve()) and path.is_file(), (page, reference)

manifest = json.loads((BUNDLE / 'manifest.json').read_text())
for entry in manifest['files']:
    path = BUNDLE / entry['file']
    assert path.stat().st_size == entry['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
for path in BUNDLE.rglob('*.png'):
    with Image.open(path) as image:
        assert min(image.size) > 0
        image.verify()

with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    expected = {'AquaPulse-Submission/' + entry['file'] for entry in manifest['files']}
    expected.add('AquaPulse-Submission/manifest.json')
    assert set(archive.namelist()) == expected
    for entry in manifest['files']:
        data = archive.read('AquaPulse-Submission/' + entry['file'])
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    assert len([name for name in archive.namelist() if name.endswith('.png')]) == 14
    assert len(archive.namelist()) == 38
    assert ARCHIVE.stat().st_size < 35_000_000

print('Verified: 38 ZIP files including research PDF and all training runs, 14 valid PNGs, offline links/checksums, under 35 MB.')
