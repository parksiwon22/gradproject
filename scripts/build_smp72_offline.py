"""Produce a self-contained, network-independent presentation artifact."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
docs=root/'docs'
html=(docs/'index.html').read_text()
css=(docs/'smp72_demo.css').read_text()
js=(docs/'smp72_demo.js').read_text()
data=(docs/'demo/smp72-2025.json').read_text().replace('</','<\\/')
html=html.replace('<link rel="stylesheet" href="./smp72_demo.css?v=20261120-ev1"><script defer src="./smp72_demo.js?v=20261120-ev1"></script>',f'<style>{css}</style>')
html=html.replace('<script defer src="./smp72_live.js?v=20261120-ev1"></script>', '').replace('<script defer src="./smp72_charging.js?v=20261120-ev1"></script>', '')
live_path=docs/'demo/forecast-20261120.json'
if live_path.exists():
    live=live_path.read_text().replace('</','<\\/')
    live_js=(docs/'smp72_charging.js').read_text()+'\n'+(docs/'smp72_live.js').read_text()
    html=html.replace('</body>',f'<script id="live-json" type="application/json">{live}</script><script>{live_js}</script></body>')
html=html.replace('</body>',f'<script id="demo-json" type="application/json">{data}</script><script>{js}</script></body>')
html=html.replace('<a href="./downloads/smp72-offline.html" download="smp72-offline.html">오프라인 시연 파일 ↓</a>', '<span>오프라인 시연 파일 실행 중</span>')
html=html.replace('<a href="./demo/smp72-2025.json" download>검증 데이터 JSON ↓</a>', '')
html=html.replace('href="./legacy-v20.html"','href="https://parksiwon22.github.io/gradproject/legacy-v20.html"')
out=docs/'downloads/smp72-offline.html';out.parent.mkdir(exist_ok=True);out.write_text(html)
print(f'{out}: {out.stat().st_size:,} bytes')
