"""Typeset the English research report as a selectable-text, illustrated A4 PDF."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'mathtext.fontset': 'stix', 'pdf.compression': 9})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'AquaPulse-Research-Report.en.md'
DEST = ROOT / 'AquaPulse-Research-Report.pdf'
PAGE_W, PAGE_H = 595.28, 841.89
MARGIN, TOP, BOTTOM = 54.0, 57.0, 782.0
BODY_W = PAGE_W - 2 * MARGIN
FONT_DIR = Path('C:/Windows/Fonts')
FONTS = {'normal': FONT_DIR / 'times.ttf', 'bold': FONT_DIR / 'timesbd.ttf',
         'italic': FONT_DIR / 'timesi.ttf'}
MEASURE = RendererAgg(1, 1, 72)


def font(size: float, style: str = 'normal') -> FontProperties:
    return FontProperties(fname=str(FONTS[style]), size=size)


@lru_cache(maxsize=24000)
def width(text: str, size: float, style: str = 'normal') -> float:
    return MEASURE.get_text_width_height_descent(text, font(size, style), ismath=False)[0]


def clean(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return re.sub(r'`([^`]+)`', r'\1', text)


def wrap(text: str, limit: float, size: float, style: str = 'normal') -> list[str]:
    words = clean(text).split()
    lines, line = [], ''
    for word in words:
        if width(word, size, style) > limit:
            pieces, piece = [], ''
            for character in word:
                if piece and width(piece + character, size, style) > limit:
                    pieces.append(piece)
                    piece = character
                else:
                    piece += character
            pieces.append(piece)
        else:
            pieces = [word]
        for piece in pieces:
            candidate = line + ' ' + piece if line else piece
            if line and width(candidate, size, style) > limit:
                lines.append(line)
                line = piece
            else:
                line = candidate
    if line:
        lines.append(line)
    return lines or ['']


class Paper:
    def __init__(self, output: Path):
        self.pdf = PdfPages(output, metadata={
            'Title': 'AquaPulse: Citizen Observations and Human Verification for Urban River Monitoring',
            'Author': 'AquaPulse project', 'Subject': 'Technical research report and prototype evaluation',
            'Keywords': 'urban rivers, citizen observations, YOLO11, Kimi, human verification',
        })
        self.fig = None
        self.number = 0
        self.y = TOP
        self.figure_count = 0
        self.bounds = []
        self.new_page()

    def text(self, x, y, value, size=11.2, style='normal', align='left', url=None, color='#141414'):
        self.fig.text(x / PAGE_W, 1 - y / PAGE_H, value, fontproperties=font(size, style),
                      ha=align, va='top', color=color, url=url)

    def new_page(self):
        if self.fig is not None:
            self.finish_page()
        self.number += 1
        self.fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=72, facecolor='white')
        self.y = TOP
        if self.number > 1:
            self.text(MARGIN, 29, 'AquaPulse | Technical research report', 8.5, 'italic', color='#555555')
            self.fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W],
                                          [1 - 44 / PAGE_H] * 2,
                                          transform=self.fig.transFigure, lw=0.4, color='#aaaaaa'))

    def finish_page(self):
        self.text(PAGE_W / 2, PAGE_H - 29, str(self.number), 9, align='center', color='#555555')
        self.pdf.savefig(self.fig)
        preview = ROOT / 'research-pdf-preview'
        preview.mkdir(exist_ok=True)
        self.fig.savefig(preview / f'page-{self.number:02d}.png', dpi=110, facecolor='white')
        plt.close(self.fig)

    def ensure(self, height):
        if self.y + height > BOTTOM:
            self.new_page()

    def paragraph(self, value, size=11.2, style='normal', gap=8, indent=0, centered=False):
        leading = size * 1.38
        lines = wrap(value, BODY_W - indent, size, style)
        if self.y + min(3, len(lines)) * leading > BOTTOM:
            self.new_page()
        url = None
        if value.startswith('['):
            found = re.search(r'https?://\S+', value)
            if found:
                url = found[0]
        for i, line in enumerate(lines):
            if self.y + leading > BOTTOM:
                self.new_page()
            if len(lines) - i == 2 and self.y + 2 * leading > BOTTOM:
                self.new_page()
            x = PAGE_W / 2 if centered else MARGIN + indent
            self.text(x, self.y, line, size, style, 'center' if centered else 'left', url)
            self.bounds.append((self.number, self.y, self.y + leading))
            self.y += leading
        self.y += gap

    def heading(self, title, level):
        if level == 1:
            self.paragraph(title, 18.5, 'bold', gap=13, centered=True)
        else:
            size = 12.8 if level == 2 else 11.7
            self.ensure(120 if level == 2 else 84)
            self.y += 8 if level == 2 else 4
            self.paragraph(title, size, 'bold', gap=8)

    def equation(self, expression):
        self.ensure(52)
        self.fig.text(0.5, 1 - (self.y + 8) / PAGE_H, '$' + expression + '$',
                      ha='center', va='top', fontsize=13.5, color='#141414')
        self.bounds.append((self.number, self.y, self.y + 46))
        self.y += 48

    def figure(self, source: Path, caption: str):
        with Image.open(source) as image:
            image.load()
            ratio = image.height / image.width
            drawn_width = BODY_W
            drawn_height = drawn_width * ratio
            if drawn_height > 425:
                drawn_height = 425
                drawn_width = drawn_height / ratio
            caption_lines = wrap(caption, BODY_W, 9.3, 'italic')
            height = drawn_height + len(caption_lines) * 12.8 + 19
            self.ensure(height)
            left = (PAGE_W - drawn_width) / 2
            axes = self.fig.add_axes([left / PAGE_W,
                                     1 - (self.y + drawn_height) / PAGE_H,
                                     drawn_width / PAGE_W, drawn_height / PAGE_H])
            axes.imshow(image)
            axes.axis('off')
            self.bounds.append((self.number, self.y, self.y + drawn_height))
            self.y += drawn_height + 7
            self.paragraph(caption, 9.3, 'italic', gap=12)
            self.figure_count += 1

    def table(self, rows, caption):
        count = len(rows[0])
        proportions = ({4: [0.39, 0.23, 0.23, 0.15],
                        5: [0.42, 0.145, 0.145, 0.145, 0.145],
                        6: [0.25, 0.17, 0.145, 0.145, 0.145, 0.145],
                        7: [0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]})[count]
        if rows[0][0] == 'Class':
            proportions = [0.31, 0.08, 0.08, 0.08, 0.08, 0.19, 0.18]
        columns = [BODY_W * value for value in proportions]
        size, leading, pad = 9.1, 12.2, 5
        wrapped = [[wrap(cell, limit - 2 * pad, size)
                    for cell, limit in zip(row, columns)] for n, row in enumerate(rows)]
        heights = [max(len(cell) for cell in row) * leading + 2 * pad for row in wrapped]
        caption_height = len(wrap(caption, BODY_W, 9.3, 'italic')) * 12.8 + 6
        self.ensure(caption_height + sum(heights) + 16)
        self.paragraph(caption, 9.3, 'italic', gap=6)
        self.y += 2

        def rule(position, weight):
            self.fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W],
                                          [1 - position / PAGE_H] * 2,
                                          transform=self.fig.transFigure, linewidth=weight,
                                          color='#202020', solid_capstyle='butt'))

        rule(self.y, 0.8)
        for n, (row, height) in enumerate(zip(wrapped, heights)):
            self.ensure(height)
            start = MARGIN
            for index, (lines, column) in enumerate(zip(row, columns)):
                numeric = bool(re.fullmatch(r'[\d\s.,/%–-]+', ' '.join(lines)))
                if index == 0:
                    x, align = start + pad, 'left'
                elif n == 0:
                    x, align = start + column / 2, 'center'
                elif numeric:
                    x, align = start + column - pad, 'right'
                else:
                    x, align = start + pad, 'left'
                offset = (height - len(lines) * leading) / 2
                for k, line in enumerate(lines):
                    self.text(x, self.y + offset + k * leading, line, size, align=align)
                start += column
            self.bounds.append((self.number, self.y, self.y + height))
            self.y += height
            if n == 0:
                rule(self.y, 0.5)
        rule(self.y, 0.8)
        self.y += 13

    def close(self):
        self.finish_page()
        self.pdf.close()
        assert all(TOP - 1 <= start < end <= BOTTOM + 1 for _, start, end in self.bounds)


def main():
    source = SOURCE.read_text(encoding='utf-8')
    lines = source.splitlines()
    paper = Paper(DEST)
    i, caption = 0, ''
    in_references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        heading = re.match(r'^(#{1,3})\s+(.+)', line)
        if heading:
            paper.heading(heading[2], len(heading[1]))
            in_references = heading[2] == 'References' or in_references
            i += 1
            continue
        image = re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)', line)
        if image:
            paper.figure((SOURCE.parent / image[2]).resolve(), image[1])
            i += 1
            continue
        if line.startswith('$$') and line.endswith('$$'):
            paper.equation(line[2:-2])
            i += 1
            continue
        if re.match(r'^Table \d+\. ', line):
            caption = line
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            paper.table(rows, caption)
            caption = ''
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            paragraph.append(lines[i].strip())
            i += 1
        text = ' '.join(paragraph)
        if paper.number == 1 and paper.y < 210 and not text.startswith('Urban river'):
            paper.paragraph(text, 9.7, 'italic', gap=6, centered=True)
        elif text.startswith('Keywords:') or text.startswith('*Different test set'):
            paper.paragraph(text.lstrip('*'), 9.5, 'italic', gap=8)
        else:
            paper.paragraph(text, 9.8 if in_references else 11.2)
    paper.close()
    reader = PdfReader(DEST)
    extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
    assert len(reader.pages) == paper.number
    assert paper.figure_count == 4
    for text in ['75.43', '82.59', '78.85', 'References', 'nkluc2212@gmail.com']:
        assert text in extracted, text
    assert len(extracted) > 18000
    (ROOT / 'research-pdf-check.json').write_text(json.dumps({
        'pages': len(reader.pages), 'figures': paper.figure_count,
        'tables': 5, 'words': len(re.findall(r'\b\S+\b', source)),
        'bytes': DEST.stat().st_size, 'selectable_text_characters': len(extracted),
        'layout_bounds_checked': True,
    }, indent=2) + '\n')
    print((ROOT / 'research-pdf-check.json').read_text())
    previews = sorted((ROOT / 'research-pdf-preview').glob('page-*.png'))
    rows = (len(previews) + 3) // 4
    contact = Image.new('RGB', (1240, rows * 460), '#e5e5e5')
    draw = ImageDraw.Draw(contact)
    for index, path in enumerate(previews):
        with Image.open(path) as preview:
            thumbnail = preview.convert('RGB')
            thumbnail.thumbnail((290, 420))
            x, y = (index % 4) * 310 + 10, (index // 4) * 460 + 25
            contact.paste(thumbnail, (x, y))
            draw.text((x, y - 17), path.stem, fill='black')
    contact.save(ROOT / 'research-pdf-preview/contact-sheet.jpg')


if __name__ == '__main__':
    main()
