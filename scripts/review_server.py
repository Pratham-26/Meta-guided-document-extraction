"""HITL (Human-in-the-Loop) Gold Standard Review Server.

A lightweight Flask-free HTTP server for reviewing, editing, approving,
and rejecting Scout-generated Gold Standards.  Run with::

    python scripts/review_server.py [--port 8111]

Then open http://localhost:8111 in a browser.
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure the project root is on sys.path so ``src`` imports work when
# the script is executed directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.loader import list_category_configs, load_category_config
from src.schemas.gold_standard import ApprovalStatus
from src.storage.fs_store import (
    approve_gold_standard,
    list_gold_standards,
    load_gold_standard,
    reject_gold_standard,
    save_gold_standard,
)
from src.storage import paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_source_document(gs) -> str:
    """Try to read the source document text for display."""
    src_path = Path(gs.source_document_uri)
    if not src_path.exists():
        return f"[Source file not found: {src_path}]"
    try:
        if src_path.suffix.lower() == ".pdf":
            return f"[PDF document: {src_path.name}]"
        return src_path.read_text(encoding="utf-8", errors="replace")[:50_000]
    except Exception as e:
        return f"[Error reading source: {e}]"


def _gs_to_json(gs) -> dict:
    """Serialize a GoldStandard for the API."""
    return {
        "id": gs.id,
        "category": gs.category,
        "input_modality": gs.input_modality,
        "source_document_uri": str(gs.source_document_uri),
        "extraction": gs.extraction,
        "approved_by": gs.approved_by,
        "created_at": gs.created_at.isoformat(),
        "supersedes": gs.supersedes,
        "approval_status": gs.approval_status.value,
    }


# ---------------------------------------------------------------------------
# HTML UI (single-page app embedded as a string)
# ---------------------------------------------------------------------------

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Gold Standard Review</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #232733;
    --border: #2d3141; --text: #e0e1e6; --text2: #9196a8;
    --accent: #7c6aef; --accent-hover: #9080ff;
    --green: #34d399; --green-bg: rgba(52,211,153,.12);
    --red: #f87171; --red-bg: rgba(248,113,113,.12);
    --yellow: #fbbf24; --yellow-bg: rgba(251,191,36,.12);
    --radius: 10px; --mono: 'SF Mono', 'Fira Code', monospace;
  }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  /* Layout */
  .app { display: flex; height: 100vh; }
  .sidebar { width: 300px; min-width: 260px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* Sidebar */
  .sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); }
  .sidebar-header h1 { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
  .sidebar-header select { width: 100%; padding: 8px 10px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface2); color: var(--text); font-size: 13px; margin-bottom: 8px; outline: none; }
  .sidebar-header select:focus { border-color: var(--accent); }
  .gs-list { flex: 1; overflow-y: auto; padding: 8px; }
  .gs-item { padding: 10px 12px; border-radius: var(--radius); cursor: pointer; margin-bottom: 4px; transition: background .15s; display: flex; align-items: center; gap: 10px; }
  .gs-item:hover { background: var(--surface2); }
  .gs-item.active { background: var(--accent); color: #fff; }
  .gs-item.active .gs-status { opacity: 1; }
  .gs-id { font-weight: 600; font-size: 13px; }
  .gs-status { font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500; text-transform: uppercase; letter-spacing: .3px; }
  .status-pending_review { background: var(--yellow-bg); color: var(--yellow); }
  .status-approved { background: var(--green-bg); color: var(--green); }
  .status-rejected { background: var(--red-bg); color: var(--red); }

  /* Main panels */
  .toolbar { padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; background: var(--surface); }
  .toolbar h2 { font-size: 15px; font-weight: 600; flex: 1; }
  .btn { padding: 7px 16px; border-radius: var(--radius); border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: background .15s, transform .1s; }
  .btn:active { transform: scale(.97); }
  .btn-approve { background: var(--green); color: #000; }
  .btn-approve:hover { background: #4ade80; }
  .btn-reject { background: var(--red); color: #fff; }
  .btn-reject:hover { background: #fca5a5; }
  .btn-save { background: var(--accent); color: #fff; }
  .btn-save:hover { background: var(--accent-hover); }

  .panels { flex: 1; display: flex; overflow: hidden; }
  .panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .panel + .panel { border-left: 1px solid var(--border); }
  .panel-header { padding: 10px 16px; background: var(--surface2); border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; color: var(--text2); }
  .panel-body { flex: 1; overflow-y: auto; padding: 16px; }
  .source-doc { white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px; line-height: 1.65; color: var(--text); }
  .json-editor { width: 100%; height: 100%; border: none; background: transparent; color: var(--text); font-family: var(--mono); font-size: 12.5px; line-height: 1.65; resize: none; outline: none; }

  .empty-state { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text2); font-size: 14px; }
  .toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; border-radius: var(--radius); font-size: 13px; font-weight: 600; color: #fff; z-index: 100; animation: slidein .25s ease-out; }
  .toast-ok { background: var(--green); color: #000; }
  .toast-err { background: var(--red); }
  @keyframes slidein { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

  /* Stats bar */
  .stats { padding: 12px 20px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text2); display: flex; gap: 16px; }
  .stats span { display: inline-flex; align-items: center; gap: 4px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-pending { background: var(--yellow); }
  .dot-approved { background: var(--green); }
  .dot-rejected { background: var(--red); }
</style>
</head>
<body>
<div class="app">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <h1>📋 Gold Standard Review</h1>
      <select id="selCategory"><option value="">Select category…</option></select>
      <select id="selModality">
        <option value="text">text</option>
        <option value="pdf">pdf</option>
      </select>
    </div>
    <div class="gs-list" id="gsList"></div>
    <div class="stats" id="statsBar"></div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="toolbar" id="toolbar">
      <h2 id="toolbarTitle">Select a Gold Standard</h2>
      <button class="btn btn-approve" id="btnApprove" style="display:none" onclick="doApprove()">✓ Approve</button>
      <button class="btn btn-reject"  id="btnReject"  style="display:none" onclick="doReject()">✗ Reject</button>
      <button class="btn btn-save"    id="btnSave"    style="display:none" onclick="doSave()">Save Edits</button>
    </div>
    <div class="panels" id="panels">
      <div class="empty-state" id="emptyState">Choose a category and gold standard to begin</div>
    </div>
  </div>
</div>

<script>
const API = '';
let currentGS = null;

// --- Data fetching ---------------------------------------------------
async function loadCategories() {
  const res = await fetch(`${API}/api/categories`);
  const cats = await res.json();
  const sel = document.getElementById('selCategory');
  sel.innerHTML = '<option value="">Select category…</option>';
  cats.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
}

async function loadGoldStandards() {
  const cat = document.getElementById('selCategory').value;
  const mod = document.getElementById('selModality').value;
  if (!cat) return;
  const res = await fetch(`${API}/api/gold_standards?category=${cat}&modality=${mod}`);
  const items = await res.json();
  renderList(items);
}

function renderList(items) {
  const list = document.getElementById('gsList');
  list.innerHTML = '';
  let pending=0, approved=0, rejected=0;
  items.forEach(gs => {
    if (gs.approval_status === 'pending_review') pending++;
    else if (gs.approval_status === 'approved') approved++;
    else rejected++;
    const div = document.createElement('div');
    div.className = `gs-item ${currentGS && currentGS.id === gs.id ? 'active' : ''}`;
    div.innerHTML = `<span class="gs-id">${gs.id}</span><span class="gs-status status-${gs.approval_status}">${gs.approval_status.replace('_',' ')}</span>`;
    div.onclick = () => selectGS(gs);
    list.appendChild(div);
  });
  document.getElementById('statsBar').innerHTML =
    `<span><span class="dot dot-pending"></span> ${pending} pending</span>` +
    `<span><span class="dot dot-approved"></span> ${approved} approved</span>` +
    `<span><span class="dot dot-rejected"></span> ${rejected} rejected</span>`;
}

async function selectGS(gs) {
  currentGS = gs;
  const cat = document.getElementById('selCategory').value;
  const mod = document.getElementById('selModality').value;

  // Fetch source
  const srcRes = await fetch(`${API}/api/source?category=${cat}&modality=${mod}&id=${gs.id}`);
  const srcData = await srcRes.json();

  document.getElementById('toolbarTitle').textContent = `${gs.id}  ·  ${gs.approval_status.replace('_',' ')}`;
  document.getElementById('btnApprove').style.display = '';
  document.getElementById('btnReject').style.display = '';
  document.getElementById('btnSave').style.display = '';
  document.getElementById('emptyState')?.remove();

  const panels = document.getElementById('panels');
  panels.innerHTML = `
    <div class="panel"><div class="panel-header">Source Document</div><div class="panel-body"><pre class="source-doc" id="sourceDoc"></pre></div></div>
    <div class="panel"><div class="panel-header">Extraction JSON (editable)</div><div class="panel-body"><textarea class="json-editor" id="jsonEditor"></textarea></div></div>
  `;
  document.getElementById('sourceDoc').textContent = srcData.content;
  document.getElementById('jsonEditor').value = JSON.stringify(gs.extraction, null, 2);

  // highlight active
  document.querySelectorAll('.gs-item').forEach(el => el.classList.remove('active'));
  event?.target?.closest('.gs-item')?.classList.add('active');
}

// --- Actions ---------------------------------------------------------
async function doApprove() {
  if (!currentGS) return;
  // save edits first
  await doSave(true);
  const cat = document.getElementById('selCategory').value;
  const mod = document.getElementById('selModality').value;
  const res = await fetch(`${API}/api/approve`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({category: cat, modality: mod, id: currentGS.id})
  });
  if (res.ok) { toast('Approved ✓', 'ok'); await loadGoldStandards(); }
  else toast('Approve failed', 'err');
}

async function doReject() {
  if (!currentGS) return;
  const cat = document.getElementById('selCategory').value;
  const mod = document.getElementById('selModality').value;
  const res = await fetch(`${API}/api/reject`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({category: cat, modality: mod, id: currentGS.id})
  });
  if (res.ok) { toast('Rejected', 'ok'); await loadGoldStandards(); }
  else toast('Reject failed', 'err');
}

async function doSave(silent) {
  if (!currentGS) return;
  const cat = document.getElementById('selCategory').value;
  const mod = document.getElementById('selModality').value;
  let extraction;
  try { extraction = JSON.parse(document.getElementById('jsonEditor').value); }
  catch(e) { toast('Invalid JSON', 'err'); return; }

  const res = await fetch(`${API}/api/update_extraction`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({category: cat, modality: mod, id: currentGS.id, extraction})
  });
  if (res.ok) {
    currentGS.extraction = extraction;
    if (!silent) toast('Saved', 'ok');
  } else if (!silent) toast('Save failed', 'err');
}

function toast(msg, type) {
  const d = document.createElement('div');
  d.className = `toast toast-${type}`;
  d.textContent = msg;
  document.body.appendChild(d);
  setTimeout(() => d.remove(), 2500);
}

// --- Init ------------------------------------------------------------
document.getElementById('selCategory').onchange = loadGoldStandards;
document.getElementById('selModality').onchange = loadGoldStandards;
loadCategories();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ReviewHandler(SimpleHTTPRequestHandler):
    """Serves the single-page review UI and a small JSON API."""

    def log_message(self, format, *args):
        # Quieter logs
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    # --- GET routes ----------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            body = _HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/categories":
            self._send_json(list_category_configs())
            return

        if path == "/api/gold_standards":
            cat = qs.get("category", [""])[0]
            mod = qs.get("modality", ["text"])[0]
            gs_list = list_gold_standards(cat, mod)
            self._send_json([_gs_to_json(gs) for gs in gs_list])
            return

        if path == "/api/source":
            cat = qs.get("category", [""])[0]
            mod = qs.get("modality", ["text"])[0]
            gs_id = qs.get("id", [""])[0]
            try:
                gs = load_gold_standard(cat, mod, gs_id)
                content = _read_source_document(gs)
                self._send_json({"content": content})
            except Exception as e:
                self._send_json({"content": f"[Error: {e}]"}, 404)
            return

        self.send_error(404)

    # --- POST routes ---------------------------------------------------

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/approve":
            data = self._read_body()
            try:
                gs = approve_gold_standard(
                    data["category"], data["modality"], data["id"],
                    approved_by="human",
                )
                self._send_json(_gs_to_json(gs))
            except Exception as e:
                self._send_json({"error": str(e)}, 400)
            return

        if path == "/api/reject":
            data = self._read_body()
            try:
                gs = reject_gold_standard(
                    data["category"], data["modality"], data["id"],
                )
                self._send_json(_gs_to_json(gs))
            except Exception as e:
                self._send_json({"error": str(e)}, 400)
            return

        if path == "/api/update_extraction":
            data = self._read_body()
            try:
                gs = load_gold_standard(
                    data["category"], data["modality"], data["id"]
                )
                gs.extraction = data["extraction"]
                save_gold_standard(data["category"], data["modality"], gs)
                self._send_json(_gs_to_json(gs))
            except Exception as e:
                self._send_json({"error": str(e)}, 400)
            return

        self.send_error(404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Start the HITL Gold Standard Review Server"
    )
    parser.add_argument("--port", type=int, default=8111, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), ReviewHandler)
    print(f"🔍 HITL Review Server running at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
