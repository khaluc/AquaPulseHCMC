"""Package the complete LaTeX source and four figures for a web TeX editor."""
from pathlib import Path
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'AquaPulse-Research-Report.tex'
ARCHIVE = ROOT / 'AquaPulse-LaTeX-Overleaf.zip'
BUNDLE = ROOT / 'AquaPulse-LaTeX-Overleaf'

README = '''# AquaPulse — hướng dẫn mở LaTeX trên web

1. Trên Overleaf, tạo dự án bằng cách tải lên file AquaPulse-LaTeX-Overleaf.zip.
2. Dùng trình biên dịch mặc định **pdfLaTeX**; XeLaTeX và LuaLaTeX cũng được hỗ trợ.
3. Chọn file chính **main.tex**, rồi bấm **Recompile**.
4. Chỉnh bài trực tiếp trong main.tex; tải PDF sau khi biên dịch.

Gói gồm toàn bộ bài nghiên cứu tiếng Anh, 4 hình, 5 bảng, 2 công thức và 20 tài liệu tham khảo.
Nội dung được căn đều hai lề. Bảng có đường kẻ ngang theo kiểu booktabs.
Đã bỏ tiêu đề chạy ở đầu trang, chỉ giữ số trang ở chân trang.
Hình dùng vị trí linh hoạt và giới hạn chiều cao để giảm khoảng trắng lớn.

Giữ thư mục figures/ cùng main.tex để các hình hiển thị đúng.
Mã nguồn đã được kiểm tra nội dung và đường dẫn hình; chưa biên dịch thành PDF tại máy này.
'''


def main():
    document = SOURCE.read_text(encoding='utf-8')
    linked = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', document)
    assert len(linked) == 4
    contents = {}
    for reference in linked:
        image = (ROOT / reference).resolve()
        assert image.is_file() and image.suffix == '.png'
        target = 'figures/' + image.name
        contents[target] = image.read_bytes()
        document = document.replace('{' + reference + '}', '{' + target + '}')
    document = document.replace(
        '% Generated from AquaPulse-Research-Report.en.md. Edit the Markdown and regenerate.\n'
        '% Compile from docs/submission with pdfLaTeX, XeLaTeX, or LuaLaTeX.',
        '% Complete editable research report for Overleaf or another web LaTeX editor.\n'
        '% Select main.tex as the main document; pdfLaTeX, XeLaTeX, and LuaLaTeX are supported.'
    )
    assert document.count(r'\begin{table}') == 5
    assert document.count(r'\begin{figure}') == 4
    assert document.count(r'\bibitem{') == 20
    assert document.count(r'\begin{equation}') == 2
    assert r'\pagestyle{plain}' in document
    assert 'fancyhdr' not in document
    assert 'AquaPulse | Technical research report' not in document
    contents['main.tex'] = document.encode('utf-8')
    contents['README.md'] = README.encode('utf-8')
    for name, data in contents.items():
        destination = BUNDLE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(contents)
        for reference in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', document):
            assert reference in archive.namelist()
        for name, data in contents.items():
            assert archive.read(name) == data
    print(json.dumps({'archive': str(ARCHIVE), 'folder': str(BUNDLE), 'bytes': ARCHIVE.stat().st_size,
                      'files': len(contents), 'figures': 4, 'tables': 5, 'references': 20,
                      'compiler': 'pdfLaTeX / XeLaTeX / LuaLaTeX', 'compiled_locally': False}, indent=2))


if __name__ == '__main__':
    main()
