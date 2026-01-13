async function analyzeDay() {
  const inputs = document.querySelectorAll('.lumi-in');
  const lines = Array.from(inputs).map(i => i.value);
  const btn = document.getElementById('mainBtn');
  const orb = document.getElementById('orb');
  const result = document.getElementById('result');
  const cand = document.getElementById('candidates');

  btn.innerText = 'Processing...';
  btn.disabled = true;
  try {
    const res = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lines })
    });
    const data = await res.json();

    result.style.display = 'block';
    document.getElementById('emotionText').innerText = data.emotion;
    document.getElementById('confText').innerText = `Confidence: ${data.confidence}`;
    document.getElementById('methodText').innerText = `Method: ${data.method}`;

    const color = data.hue !== null ? `hsl(${data.hue},85%,65%)` : '#e9edf2';
    orb.style.background = color;
    orb.style.boxShadow = `0 16px 50px ${color}33`;

    cand.innerHTML = '';
    if (Array.isArray(data.candidates)) {
      data.candidates.slice(0,6).forEach(c => {
        const el = document.createElement('div');
        el.className = 'cand';
        const dot = document.createElement('div');
        dot.className = 'dot';
        dot.style.background = c.hue !== null && c.hue !== undefined ? `hsl(${c.hue},85%,60%)` : '#e9edf2';
        const lbl = document.createElement('div'); lbl.innerText = c.label.split('/')[0]; lbl.style.fontWeight='600'; lbl.style.fontSize='12px';
        const sc = document.createElement('div'); sc.innerText = `${Math.round(c.score*100)}%`; sc.style.color='#8b8b92'; sc.style.fontSize='11px';
        el.appendChild(dot); el.appendChild(lbl); el.appendChild(sc); cand.appendChild(el);
      })
    }

  } catch (e) {
    alert('Make sure the backend is running at http://127.0.0.1:8000');
  } finally {
    btn.disabled = false; btn.innerText = 'Reveal My Color';
  }
}