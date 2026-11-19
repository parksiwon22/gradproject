"""Self-contained 2025 replay and EV simulation, no 2026 payload."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1];docs=root/'docs'
html=(docs/'index.html').read_text()
html=re.sub(r'<link rel="stylesheet" href="./smp72_demo.css[^"]*">',lambda m:'<style>'+(docs/'smp72_demo.css').read_text()+'</style>',html)
html=re.sub(r'<script defer src="./smp72_[^"]+"></script>','',html)
data=(docs/'demo/smp72-2025.json').read_text().replace('</','<\\/')
scripts=''.join('<script>'+(docs/name).read_text()+'</script>' for name in ['smp72_demo.js','smp72_charging.js','smp72_weather.js','smp72_live.js'])
latest=(docs/'demo/forecast-20261120.json').read_text().replace('</','<\\/')
html=html.replace('</body>',f'<script id="latest-json" type="application/json">{latest}</script></body>')
html=html.replace('</body>',f'<script id="demo-json" type="application/json">{data}</script>{scripts}</body>')
html=html.replace('<a href="./downloads/smp72-offline.html" download="smp72-offline.html">오프라인 시연 파일 ↓</a>','<span>오프라인 시연 실행 중</span>')
html=html.replace('<a href="./demo/smp72-2025.json" download>검증 데이터 JSON ↓</a>','')
out=docs/'downloads/smp72-offline.html';out.parent.mkdir(exist_ok=True);out.write_text(html)
print(f'{out}: {out.stat().st_size:,} bytes')
