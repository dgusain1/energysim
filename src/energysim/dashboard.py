# -*- coding: utf-8 -*-
"""
Interactive HTML Result Explorer for energysim.

Generates a self-contained HTML file (Plotly.js via CDN, or embedded with
``offline=True``) that lets users:
    * Browse all simulator variables in a tree
    * Create / rename / delete custom subplots
    * Assign any variable to any subplot (drag-and-drop or click)
    * Remove variables from subplots individually
    * Choose 1- or 2-column layout
    * Toggle line style per trace
    * Export the figure as PNG or standalone HTML
    * Quick-view: auto-plot everything ("Overview" button)
"""
from __future__ import annotations


import html
import json
import os
import tempfile
import warnings
import webbrowser
from datetime import datetime

import numpy as np

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.0.min.js"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _time_scale(max_span):
    """Pick a display unit for a time span in seconds -> (divisor, label)."""
    if max_span >= 7200:
        return 3600.0, 'Time (h)'
    if max_span >= 120:
        return 60.0, 'Time (min)'
    return 1.0, 'Time (s)'


def _decimate_indices(y, max_points):
    """Indices of a min/max-preserving decimation of *y* to about *max_points* points.

    The series is split into ``max_points // 2`` buckets; each bucket keeps
    the positions of its minimum and maximum, so peaks survive.  The first and
    last points are always kept.  Non-numeric data falls back to a stride.
    """
    n = len(y)
    if max_points is None or n <= max_points:
        return None
    nb = max(1, int(max_points) // 2)
    edges = np.linspace(0, n, nb + 1).astype(np.int64)
    try:
        yf = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        yf = None
    if yf is None:
        idx = np.unique(np.concatenate([np.linspace(0, n - 1, max_points).astype(np.int64)]))
        return idx
    keep = [0, n - 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg = yf[lo:hi]
        if np.all(np.isnan(seg)):
            keep.append(lo)
            continue
        keep.append(lo + int(np.nanargmin(seg)))
        keep.append(lo + int(np.nanargmax(seg)))
    return np.unique(np.asarray(keep, dtype=np.int64))


def _to_json_list(arr):
    a = np.asarray(arr)
    if a.dtype.kind == 'b':
        return a.astype(int).tolist()
    if a.dtype.kind in 'iuf':
        return a.tolist()
    return [None if v is None or (isinstance(v, float) and v != v) else
            (v.item() if isinstance(v, np.generic) else (v if isinstance(v, (int, float, str)) else str(v)))
            for v in a]


def _iter_tables(results):
    """Yield (sim_name, time_seconds ndarray, DataFrame without time) without mutating inputs."""
    from .results import Results
    if isinstance(results, Results):
        for sim in results.sims:
            df = results._raw(sim)
            yield sim, df.index.to_numpy(dtype=np.float64), df
        return
    for sim_name, df in results.items():
        if 'time' in df.columns:
            t = df['time'].to_numpy()
            data = df.drop(columns=['time'])
        else:
            t = df.index.to_numpy()
            data = df
        if np.issubdtype(np.asarray(t).dtype, np.datetime64):
            t = (t - t[0]) / np.timedelta64(1, 's') if len(t) else np.asarray([], dtype=np.float64)
        try:
            t = np.asarray(t, dtype=np.float64)
        except (TypeError, ValueError):
            t = np.arange(len(df), dtype=np.float64)
        yield sim_name, t, data


def _build_traces_json(results, max_points=None, divisor=1.0):
    """Convert results ({sim: DataFrame} or Results) -> JSON-ready list of sim descriptors.

    Time is divided by *divisor*; series longer than *max_points* are
    decimated (min/max preserving) and then carry their own x values in
    ``xs[var]``.  The inputs are never modified.
    """
    sims = []
    for sim_name, t, df in _iter_tables(results):
        cols = [c for c in df.columns if str(c).lower() != "time"]
        if not cols:
            continue
        tt = np.asarray(t, dtype=np.float64) / float(divisor)
        data, xs = {}, {}
        for col in cols:
            y = df[col].to_numpy()
            idx = _decimate_indices(y, max_points)
            if idx is None:
                data[str(col)] = _to_json_list(y)
            else:
                data[str(col)] = _to_json_list(y[idx])
                xs[str(col)] = tt[idx].tolist()
        entry = {
            "name": str(sim_name),
            "variables": [str(c) for c in cols],
            "time": tt.tolist() if len(xs) < len(cols) else [],
            "data": data,
        }
        if xs:
            entry["xs"] = xs
        sims.append(entry)
    return sims


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_JS_ESCAPES = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    " ": "\\u2028",
    " ": "\\u2029",
}


def _json_for_script(obj):
    """Serialise *obj* to JSON that is safe to embed inside a ``<script>`` element.

    ``<``, ``>``, ``&`` and the JS line separators U+2028/U+2029 are
    replaced by ``\\uXXXX`` escapes, so values such as ``</script>`` can
    never terminate the script block.  NaN/Infinity become ``null``.
    """
    def _clean(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    s = json.dumps(_clean(obj), allow_nan=False)
    for ch, rep in _JS_ESCAPES.items():
        s = s.replace(ch, rep)
    return s


def _plotly_script_tag(plotly_js=None):
    if plotly_js is None:
        return '<script src="' + _PLOTLY_CDN + '" charset="utf-8"></script>'
    # never let the embedded library terminate the script element early
    safe = plotly_js.replace('</script', '<\\/script').replace('</SCRIPT', '<\\/SCRIPT')
    return '<script charset="utf-8">' + safe + '</script>'


def _generate_html(sims_json, title="energysim Result Explorer", time_label="Time (h)", plotly_js=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sims_js = _json_for_script(sims_json)
    escaped_title = html.escape(str(title), quote=True)

    # The entire dashboard is a single HTML string with embedded CSS + JS.
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>""" + escaped_title + r"""</title>
""" + _plotly_script_tag(plotly_js) + r"""
<style>
:root {
    --bg:          #121220;
    --bg-panel:    #1a1a2e;
    --bg-card:     #222240;
    --bg-hover:    #2a2a4a;
    --accent:      #7c6ff7;
    --accent2:     #38bdf8;
    --accent-dim:  rgba(124,111,247,.25);
    --danger:      #f87171;
    --text:        #e0e0f0;
    --text2:       #9090b0;
    --border:      #333355;
    --radius:      8px;
}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;overflow:hidden;
  font-family:'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);}

/* ---------- Layout ---------- */
.app{display:flex;height:100%;}

/* Left panel: variable browser */
.panel-left{
    width:280px;min-width:280px;
    background:var(--bg-panel);
    border-right:1px solid var(--border);
    display:flex;flex-direction:column;
}
.panel-left .hdr{
    padding:16px;border-bottom:1px solid var(--border);
}
.panel-left .hdr h1{font-size:16px;color:var(--accent);margin-bottom:2px;}
.panel-left .hdr small{font-size:11px;color:var(--text2);}
.panel-left .tree{flex:1;overflow-y:auto;padding:8px;}
.panel-left .tree::-webkit-scrollbar{width:5px;}
.panel-left .tree::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

/* Right panel: subplot manager */
.panel-right{
    width:300px;min-width:300px;
    background:var(--bg-panel);
    border-left:1px solid var(--border);
    display:flex;flex-direction:column;
}
.panel-right .hdr{
    padding:14px 16px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between;
}
.panel-right .hdr h2{font-size:14px;font-weight:600;}
.panel-right .sp-list{flex:1;overflow-y:auto;padding:8px;}
.panel-right .sp-list::-webkit-scrollbar{width:5px;}
.panel-right .sp-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

/* Centre: toolbar + plot */
.centre{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.toolbar{
    display:flex;align-items:center;gap:10px;
    padding:10px 18px;
    border-bottom:1px solid var(--border);
    background:var(--bg-panel);
    flex-wrap:wrap;
}
.toolbar .sep{width:1px;height:24px;background:var(--border);}

.plot-wrap{flex:1;overflow:auto;padding:12px;}
.plot-wrap::-webkit-scrollbar{width:6px;}
.plot-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
#plotArea{width:100%;min-height:500px;}

/* ---------- Components ---------- */
.btn{
    padding:5px 14px;border-radius:var(--radius);border:1px solid var(--border);
    background:transparent;color:var(--text2);cursor:pointer;font-size:12px;
    transition:.15s;display:inline-flex;align-items:center;gap:5px;
}
.btn:hover{border-color:var(--accent);color:var(--accent);}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff;}
.btn.primary:hover{background:#6b5ce6;}
.btn.sm{padding:2px 8px;font-size:11px;border-radius:4px;}
.btn.danger{border-color:var(--danger);color:var(--danger);}
.btn.danger:hover{background:rgba(248,113,113,.15);}

/* Sim tree */
.sim-node{margin-bottom:6px;}
.sim-hdr{
    display:flex;align-items:center;gap:6px;
    padding:7px 10px;border-radius:6px;cursor:pointer;
    background:var(--bg-card);user-select:none;transition:.12s;
}
.sim-hdr:hover{background:var(--bg-hover);}
.sim-hdr .dot{width:9px;height:9px;border-radius:50%;}
.sim-hdr .name{font-size:13px;font-weight:600;flex:1;}
.sim-hdr .cnt{font-size:11px;color:var(--text2);}
.sim-hdr .chev{font-size:10px;color:var(--text2);transition:transform .2s;}
.sim-hdr .chev.open{transform:rotate(90deg);}
.sim-vars{display:none;padding:4px 0 0 0;}
.sim-vars.open{display:block;}
.var-row{
    display:flex;align-items:center;gap:6px;
    padding:4px 8px 4px 24px;border-radius:4px;cursor:grab;
    font-size:12px;transition:.1s;
}
.var-row:hover{background:rgba(124,111,247,.08);}
.var-row.dragging{opacity:.4;}
.var-row .vname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.var-row .add-btn{
    opacity:0;font-size:16px;color:var(--accent);cursor:pointer;
    transition:.15s;padding:0 4px;
}
.var-row:hover .add-btn{opacity:1;}

/* Subplot cards */
.sp-card{
    background:var(--bg-card);border-radius:var(--radius);
    margin-bottom:8px;border:2px solid transparent;
    transition:border-color .15s;
}
.sp-card.drop-target{border-color:var(--accent);}
.sp-card-hdr{
    display:flex;align-items:center;gap:6px;
    padding:8px 10px;border-bottom:1px solid var(--border);cursor:pointer;
}
.sp-card-hdr .sp-color{width:8px;height:8px;border-radius:50%;}
.sp-card-hdr .sp-title{
    flex:1;font-size:13px;font-weight:600;
    background:none;border:none;color:var(--text);
    cursor:text;outline:none;
    padding:2px 4px;border-radius:3px;
}
.sp-card-hdr .sp-title:focus{background:var(--bg-hover);}
.sp-card-hdr .sp-del{
    font-size:14px;color:var(--text2);cursor:pointer;padding:0 4px;
    transition:.15s;
}
.sp-card-hdr .sp-del:hover{color:var(--danger);}
.sp-card-body{padding:6px 10px;min-height:32px;}
.sp-card-body.empty-hint{
    display:flex;align-items:center;justify-content:center;
    color:var(--text2);font-size:11px;font-style:italic;
    min-height:40px;
}
.sp-var{
    display:flex;align-items:center;gap:4px;
    padding:3px 6px;margin:2px 0;border-radius:4px;font-size:12px;
    background:var(--bg-hover);
}
.sp-var .sv-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sp-var .sv-rm{
    font-size:13px;cursor:pointer;color:var(--text2);padding:0 3px;
    transition:.15s;
}
.sp-var .sv-rm:hover{color:var(--danger);}

/* Empty state */
.empty-state{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:100%;color:var(--text2);gap:10px;
}
.empty-state svg{opacity:.35;}
.empty-state p{font-size:14px;}
.empty-state .ht{font-size:11px;opacity:.6;}

/* Column toggle highlight */
.active-col{background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important;}
</style>
</head>
<body>
<div class="app">

<!-- ===== LEFT: Variable Browser ===== -->
<div class="panel-left">
    <div class="hdr">
        <h1>&#9889; energysim</h1>
        <small>Result Explorer &middot; """ + timestamp + r"""</small>
    </div>
    <div class="tree" id="varTree"></div>
</div>

<!-- ===== CENTRE: Toolbar + Plot ===== -->
<div class="centre">
    <div class="toolbar">
        <button class="btn primary" onclick="autoOverview()"
         title="Plot all variables grouped by simulator">&#9733; Overview</button>
        <div class="sep"></div>
        <span style="font-size:12px;color:var(--text2)">Layout:</span>
        <button class="btn sm" id="btnCol1" onclick="setColumns(1)">1 col</button>
        <button class="btn sm active-col" id="btnCol2" onclick="setColumns(2)">2 col</button>
        <div class="sep"></div>
        <button class="btn sm" onclick="exportPNG()" title="Download as PNG">&#128247; PNG</button>
        <button class="btn sm" onclick="exportHTML()" title="Save standalone HTML">&#128190; HTML</button>
        <div class="sep"></div>
        <button class="btn sm danger" onclick="clearAll()" title="Remove all subplots">&#10006; Clear</button>
        <div style="flex:1"></div>
        <span id="statusText" style="font-size:11px;color:var(--text2)"></span>
    </div>
    <div class="plot-wrap">
        <div id="plotArea"></div>
    </div>
</div>

<!-- ===== RIGHT: Subplot Manager ===== -->
<div class="panel-right">
    <div class="hdr">
        <h2>Subplots</h2>
        <button class="btn sm primary" onclick="addSubplot()">+ New</button>
    </div>
    <div class="sp-list" id="spList"></div>
</div>

</div>

<script>
// ==================================================================
//  DATA (injected by Python)
// ==================================================================
const SIMS = """ + sims_js + r""";

const PALETTE = [
    '#7c6ff7','#4ade80','#f97316','#38bdf8','#f472b6','#fbbf24',
    '#a78bfa','#34d399','#fb7185','#60a5fa','#c084fc','#2dd4bf',
    '#e879f9','#facc15','#22d3ee','#f43f5e','#84cc16','#6366f1',
];
const LINE_COLORS = [
    '#7c6ff7','#38bdf8','#4ade80','#f97316','#f472b6','#fbbf24',
    '#a78bfa','#34d399','#fb7185','#60a5fa','#c084fc','#2dd4bf',
    '#e879f9','#facc15','#22d3ee','#f43f5e','#84cc16','#6366f1',
    '#818cf8','#67e8f9','#86efac','#fdba74','#f9a8d4','#fde047',
];

// ==================================================================
//  STATE
// ==================================================================
let subplots = [];          // [{id, title, color, vars:[{sim,varName}]}]
let spIdCounter = 0;
let numColumns = 2;
let dragPayload = null;     // {sim, varName}
let activeSpId = null;      // currently-focused subplot for click-to-add

// helper: HTML-escape any text that Plotly (which renders a subset of
// HTML in titles / legends / annotations) might interpret.
function escapeHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// helper: build a DOM element; text is always set via textContent
function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
}

// ==================================================================
//  VARIABLE TREE (left panel)
// ==================================================================
function buildVarTree() {
    var tree = document.getElementById('varTree');
    tree.textContent = '';
    SIMS.forEach(function(sim, si) {
        var col = PALETTE[si % PALETTE.length];
        var node = el('div', 'sim-node');
        var hdr = el('div', 'sim-hdr');
        hdr.addEventListener('click', function(){ toggleSim(hdr); });
        var dot = el('span', 'dot');
        dot.style.background = col;
        hdr.appendChild(dot);
        hdr.appendChild(el('span', 'name', sim.name));
        hdr.appendChild(el('span', 'cnt', sim.variables.length));
        hdr.appendChild(el('span', 'chev', '▶'));
        node.appendChild(hdr);
        var vars = el('div', 'sim-vars');
        sim.variables.forEach(function(v) {
            var row = el('div', 'var-row');
            row.draggable = true;
            row.dataset.sim = sim.name;
            row.dataset.var = v;
            row.addEventListener('dragstart', onDragStart);
            row.addEventListener('dragend', onDragEnd);
            var vn = el('span', 'vname', v);
            vn.title = sim.name + '.' + v;
            row.appendChild(vn);
            var add = el('span', 'add-btn', '+');
            add.addEventListener('click', function(){ clickAdd(sim.name, v); });
            row.appendChild(add);
            vars.appendChild(row);
        });
        node.appendChild(vars);
        tree.appendChild(node);
    });
}

function toggleSim(hdr) {
    const vars = hdr.nextElementSibling;
    const chev = hdr.querySelector('.chev');
    vars.classList.toggle('open');
    chev.classList.toggle('open');
}

// ==================================================================
//  DRAG & DROP
// ==================================================================
function onDragStart(e) {
    const row = e.currentTarget;
    dragPayload = { sim: row.dataset.sim, varName: row.dataset.var };
    row.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('text/plain', '');
}
function onDragEnd(e) {
    e.currentTarget.classList.remove('dragging');
    dragPayload = null;
    document.querySelectorAll('.sp-card').forEach(function(c){ c.classList.remove('drop-target'); });
}
function onSpDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drop-target');
}
function onSpDragLeave(e) {
    e.currentTarget.classList.remove('drop-target');
}
function onSpDrop(e, spId) {
    e.preventDefault();
    e.currentTarget.classList.remove('drop-target');
    if (!dragPayload) return;
    addVarToSubplot(spId, dragPayload.sim, dragPayload.varName);
    dragPayload = null;
}

// ==================================================================
//  CLICK-TO-ADD  (+ button in variable tree)
// ==================================================================
function clickAdd(sim, varName) {
    if (activeSpId !== null && subplots.find(function(s){return s.id===activeSpId;})) {
        addVarToSubplot(activeSpId, sim, varName);
    } else if (subplots.length > 0) {
        addVarToSubplot(subplots[subplots.length - 1].id, sim, varName);
    } else {
        var sp = addSubplot();
        addVarToSubplot(sp.id, sim, varName);
    }
}

// ==================================================================
//  SUBPLOT MANAGEMENT
// ==================================================================
function addSubplot(title) {
    var id = ++spIdCounter;
    var col = PALETTE[(id - 1) % PALETTE.length];
    var sp = { id: id, title: title || 'Subplot ' + id, color: col, vars: [] };
    subplots.push(sp);
    activeSpId = id;
    renderSubplots();
    updatePlot();
    return sp;
}

function removeSubplot(id) {
    subplots = subplots.filter(function(s){return s.id !== id;});
    if (activeSpId === id) activeSpId = subplots.length ? subplots[subplots.length-1].id : null;
    renderSubplots();
    updatePlot();
}

function renameSubplot(id, newTitle) {
    var sp = subplots.find(function(s){return s.id===id;});
    if (sp) { sp.title = newTitle || sp.title; updatePlot(); }
}

function addVarToSubplot(spId, sim, varName) {
    var sp = subplots.find(function(s){return s.id===spId;});
    if (!sp) return;
    if (sp.vars.some(function(v){return v.sim===sim && v.varName===varName;})) return;
    sp.vars.push({ sim: sim, varName: varName });
    renderSubplots();
    updatePlot();
}

function removeVarFromSubplot(spId, sim, varName) {
    var sp = subplots.find(function(s){return s.id===spId;});
    if (!sp) return;
    sp.vars = sp.vars.filter(function(v){return !(v.sim===sim && v.varName===varName);});
    renderSubplots();
    updatePlot();
}

function setActiveSubplot(id) {
    activeSpId = id;
    renderSubplots();
}

// ==================================================================
//  RENDER SUBPLOT CARDS (right panel)
// ==================================================================
function renderSubplots() {
    var el = document.getElementById('spList');
    if (subplots.length === 0) {
        el.innerHTML = '<div class="empty-state" style="padding:40px 0">' +
            '<p>No subplots yet</p>' +
            '<div class="ht">Click "+ New" or drag a variable here</div></div>';
        return;
    }
    el.textContent = '';
    subplots.forEach(function(sp) {
        var isActive = sp.id === activeSpId;
        var card = document.createElement('div');
        card.className = 'sp-card' + (isActive ? ' drop-target' : '');
        if (isActive) card.style.borderColor = 'var(--accent)';
        card.addEventListener('click', function(){ setActiveSubplot(sp.id); });
        card.addEventListener('dragover', onSpDragOver);
        card.addEventListener('dragleave', onSpDragLeave);
        card.addEventListener('drop', function(e){ onSpDrop(e, sp.id); });

        var hdr = document.createElement('div');
        hdr.className = 'sp-card-hdr';
        var dot = document.createElement('span');
        dot.className = 'sp-color';
        dot.style.background = sp.color;
        hdr.appendChild(dot);
        var inp = document.createElement('input');
        inp.className = 'sp-title';
        inp.value = sp.title;
        inp.addEventListener('change', function(){ renameSubplot(sp.id, inp.value); });
        inp.addEventListener('click', function(e){ e.stopPropagation(); });
        hdr.appendChild(inp);
        var del = document.createElement('span');
        del.className = 'sp-del';
        del.title = 'Delete subplot';
        del.textContent = '×';
        del.addEventListener('click', function(e){ e.stopPropagation(); removeSubplot(sp.id); });
        hdr.appendChild(del);
        card.appendChild(hdr);

        var body = document.createElement('div');
        body.className = 'sp-card-body' + (sp.vars.length === 0 ? ' empty-hint' : '');
        if (sp.vars.length === 0) {
            body.textContent = 'Drag variables here or click + in the tree';
        } else {
            sp.vars.forEach(function(v) {
                var row = document.createElement('div');
                row.className = 'sp-var';
                var nm = document.createElement('span');
                nm.className = 'sv-name';
                nm.title = v.sim + '.' + v.varName;
                nm.textContent = v.sim + '.' + v.varName;
                row.appendChild(nm);
                var rm = document.createElement('span');
                rm.className = 'sv-rm';
                rm.textContent = '×';
                rm.addEventListener('click', function(e){
                    e.stopPropagation();
                    removeVarFromSubplot(sp.id, v.sim, v.varName);
                });
                row.appendChild(rm);
                body.appendChild(row);
            });
        }
        card.appendChild(body);
        el.appendChild(card);
    });
}

// ==================================================================
//  TIME LABEL  (injected by Python based on auto-detected unit)
// ==================================================================
var TIME_LABEL = escapeHtml(""" + _json_for_script(str(time_label)) + r""");

// ==================================================================
//  PLOTTING
// ==================================================================
function updatePlot() {
    var plotDiv = document.getElementById('plotArea');
    var activeSubplots = subplots.filter(function(s){return s.vars.length > 0;});
    var status = document.getElementById('statusText');
    var totalVars = activeSubplots.reduce(function(s,sp){return s + sp.vars.length;}, 0);
    status.textContent = subplots.length + ' subplot' + (subplots.length!==1?'s':'') +
                         ', ' + totalVars + ' variable' + (totalVars!==1?'s':'');

    if (activeSubplots.length === 0) {
        Plotly.purge(plotDiv);
        plotDiv.innerHTML = '<div class="empty-state">' +
            '<svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
            '<path d="M3 3v18h18"/><path d="M7 16l4-8 4 5 5-9"/></svg>' +
            '<p>Create a subplot and add variables to start plotting</p>' +
            '<div class="ht">Use the left panel to browse, right panel to manage subplots</div></div>';
        return;
    }

    var n = activeSubplots.length;
    var rows = numColumns === 1 ? n : Math.ceil(n / 2);
    var cols = numColumns === 1 ? 1 : Math.min(n, 2);

    var traces = [];
    var colorIdx = 0;

    activeSubplots.forEach(function(sp, si) {
        var axIdx = si + 1;
        var xRef = si === 0 ? 'x' : 'x' + axIdx;
        var yRef = si === 0 ? 'y' : 'y' + axIdx;

        sp.vars.forEach(function(v) {
            var simData = SIMS.find(function(s){return s.name === v.sim;});
            if (!simData) return;
            traces.push({
                x: (simData.xs && simData.xs[v.varName]) || simData.time,
                y: simData.data[v.varName],
                type: 'scatter',
                mode: 'lines',
                name: escapeHtml(v.sim + '.' + v.varName),
                legendgroup: 'sp' + sp.id,
                xaxis: xRef,
                yaxis: yRef,
                line: { color: LINE_COLORS[colorIdx % LINE_COLORS.length], width: 1.6 },
            });
            colorIdx++;
        });
    });

    var layout = {
        paper_bgcolor: '#121220',
        plot_bgcolor:  '#121220',
        font: { color: '#e0e0f0', size: 11 },
        margin: { l: 55, r: 20, t: 35, b: 40 },
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            bgcolor: 'rgba(26,26,46,0.9)',
            bordercolor: '#333355',
            borderwidth: 1,
            font: { size: 10 },
        },
        grid: {
            rows: rows,
            columns: cols,
            pattern: 'independent',
            roworder: 'top to bottom',
            xgap: 0.06,
            ygap: 0.08,
        },
        height: Math.max(420, rows * 280),
    };

    activeSubplots.forEach(function(sp, si) {
        var axIdx = si + 1;
        var xKey = si === 0 ? 'xaxis' : 'xaxis' + axIdx;
        var yKey = si === 0 ? 'yaxis' : 'yaxis' + axIdx;
        var isLastRow = (numColumns === 1)
            ? si === activeSubplots.length - 1
            : Math.floor(si / 2) === rows - 1;
        layout[xKey] = {
            gridcolor: '#2a2a48', zerolinecolor: '#2a2a48',
            title: isLastRow ? TIME_LABEL : '',
            showticklabels: true,
        };
        layout[yKey] = {
            gridcolor: '#2a2a48', zerolinecolor: '#2a2a48',
            title: escapeHtml(sp.title),
        };
    });

    // Subplot title annotations
    layout.annotations = activeSubplots.map(function(sp, si) {
        var axIdx = si + 1;
        var xRef = si === 0 ? 'x' : 'x' + axIdx;
        var yRef = si === 0 ? 'y' : 'y' + axIdx;
        return {
            text: '<b>' + escapeHtml(sp.title) + '</b>',
            xref: xRef + ' domain', yref: yRef + ' domain',
            x: 0, y: 1.06,
            showarrow: false,
            font: { size: 12, color: sp.color },
        };
    });

    Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}

// ==================================================================
//  TOOLBAR ACTIONS
// ==================================================================
function setColumns(n) {
    numColumns = n;
    document.getElementById('btnCol1').classList.toggle('active-col', n===1);
    document.getElementById('btnCol2').classList.toggle('active-col', n===2);
    updatePlot();
}

function autoOverview() {
    subplots = [];
    spIdCounter = 0;
    SIMS.forEach(function(sim, si) {
        var sp = { id: ++spIdCounter, title: sim.name, color: PALETTE[si % PALETTE.length], vars: [] };
        sim.variables.forEach(function(v){ sp.vars.push({ sim: sim.name, varName: v }); });
        subplots.push(sp);
    });
    activeSpId = subplots.length ? subplots[0].id : null;
    renderSubplots();
    updatePlot();
    document.querySelectorAll('.sim-vars').forEach(function(v){ v.classList.add('open'); });
    document.querySelectorAll('.chev').forEach(function(c){ c.classList.add('open'); });
}

function clearAll() {
    subplots = [];
    spIdCounter = 0;
    activeSpId = null;
    renderSubplots();
    updatePlot();
}

function exportPNG() {
    Plotly.downloadImage(document.getElementById('plotArea'),
        { format: 'png', width: 1800, height: 900, filename: 'energysim_results' });
}

function safeJson(o) {
    return JSON.stringify(o).replace(/</g,'\\u003c').replace(/>/g,'\\u003e')
        .replace(/&/g,'\\u0026').replace(/\u2028/g,'\\u2028').replace(/\u2029/g,'\\u2029');
}

function exportHTML() {
    var plotDiv = document.getElementById('plotArea');
    var d = plotDiv.data;
    var l = plotDiv.layout;
    if (!d || d.length === 0) { alert('Nothing to export.'); return; }
    var blob = new Blob([
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>energysim export</title>' +
        '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"><\/script></head>' +
        '<body style="margin:0;background:#121220"><div id="p"></div><script>' +
        'Plotly.newPlot("p",' + safeJson(d) + ',' + safeJson(l) + ',{responsive:true});' +
        '<\/script></body></html>'
    ], { type: 'text/html' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'energysim_results.html';
    a.click();
}

// ==================================================================
//  INIT
// ==================================================================
buildVarTree();
renderSubplots();
autoOverview();

// Drop on empty subplot list => create new subplot
document.getElementById('spList').addEventListener('dragover', function(e){ e.preventDefault(); });
document.getElementById('spList').addEventListener('drop', function(e) {
    e.preventDefault();
    if (!dragPayload) return;
    if (e.target.id === 'spList' || e.target.closest('.empty-state')) {
        var sp = addSubplot();
        addVarToSubplot(sp.id, dragPayload.sim, dragPayload.varName);
    }
    dragPayload = null;
});
</script>
</body>
</html>"""
    return html_content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _offline_plotly_js():
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        warnings.warn("offline=True needs the 'plotly' package (pip install plotly); "
                      "falling back to the Plotly CDN.", UserWarning, stacklevel=3)
        return None
    return get_plotlyjs()


def generate_dashboard(results, output_path=None, auto_open=False,
                       title="energysim Result Explorer", time_label=None,
                       max_points=5000, offline=False, time_unit='auto'):
    """Generate an interactive HTML result explorer.

    Parameters
    ----------
    results : energysim.results.Results or dict
        A :class:`~energysim.results.Results` or ``{sim_name: DataFrame}``
        (time in seconds, as a ``time`` column or the index).  Inputs are
        never modified.
    output_path : str or None
        Where to write the HTML file.  If *None*, a temp file is created.
    auto_open : bool
        Open the dashboard in the default web browser (default False).
    title : str
        Page title.
    time_label : str or None
        X-axis label; by default derived from the chosen time unit.
    max_points : int or None
        Series longer than this are decimated (min/max preserving).
    offline : bool
        Embed plotly.js (from the ``plotly`` Python package) instead of
        loading it from the CDN.
    time_unit : 'auto', 's', 'min' or 'h'
        Display unit of the time axis ('auto' picks it from the data span).

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    units = {'s': (1.0, 'Time (s)'), 'min': (60.0, 'Time (min)'), 'h': (3600.0, 'Time (h)')}
    if time_unit == 'auto':
        span = 0.0
        for _, t, _df in _iter_tables(results):
            if len(t):
                span = max(span, float(np.nanmax(t)) - min(0.0, float(np.nanmin(t))))
        divisor, label = _time_scale(span)
    elif time_unit in units:
        divisor, label = units[time_unit]
    else:
        raise ValueError(f"time_unit must be 'auto', 's', 'min' or 'h', got {time_unit!r}.")
    if time_label is not None:
        label = time_label

    sims_json = _build_traces_json(results, max_points=max_points, divisor=divisor)
    plotly_js = _offline_plotly_js() if offline else None
    html_text = _generate_html(sims_json, title=title, time_label=label, plotly_js=plotly_js)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="energysim_dashboard_")
        os.close(fd)
    else:
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    if auto_open:
        webbrowser.open("file://" + os.path.abspath(output_path).replace("\\", "/"))

    return os.path.abspath(output_path)
