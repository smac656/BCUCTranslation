const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
const caption = document.getElementById('caption');
const langSelect = document.getElementById('lang');


ws.onopen = () => console.log('ws open');
ws.onmessage = (ev) => {
try {
const data = JSON.parse(ev.data);
if (data.type === 'translation_packet') {
const lang = langSelect.value || 'en';
if (lang === 'en') {
caption.innerText = data.english || '';
} else {
caption.innerText = (data.translations && data.translations[lang]) || '—';
}
}
} catch (e) {
console.error(e);
}
};


ws.onclose = () => caption.innerText = 'Connection closed';