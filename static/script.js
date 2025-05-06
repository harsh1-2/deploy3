
let selectedColor = 0;

function refreshPaintboard() {
    const paintboard = document.getElementById('paintboard');
    paintboard.src = "{{ url_for('paintboard_feed') }}?t=" + new Date().getTime();
}

setInterval(refreshPaintboard, 100);
function selectColor(colorIndex) {
selectedColor = colorIndex;
fetch('/select_color/${colorIndex}', { method: "POST" });
}

