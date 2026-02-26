# -*- coding: utf-8 -*-
"""
Interactive HTML Result Explorer for energysim.

Generates a self-contained HTML file (Plotly.js via CDN) that lets users:
    * Browse all simulator variables in a tree
    * Create / rename / delete custom subplots
    * Assign any variable to any subplot (drag-and-drop or click)
    * Remove variables from subplots individually
    * Choose 1- or 2-column layout
    * Toggle line style per trace
    * Export the figure as PNG or standalone HTML
    * Quick-view: auto-plot everything ("Overview" button)
"""

import json
import os
import tempfile
import webbrowser
from datetime import datetime


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_traces_json(results_dict):
    """Convert {sim_name: DataFrame} -> JSON-ready list of sim descriptors."""
    sims = []
    for sim_name, df in results_dict.items():
        cols = [c for c in df.columns if c.lower() != "time"]
        if not cols:
            continue
        time_vals = df["time"].tolist() if "time" in df.columns else list(range(len(df)))
        data = {col: df[col].tolist() for col in cols}
        sims.append({
            "name": sim_name,
            "variables": cols,
            "time": time_vals,
            "data": data,
        })
    return sims


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _generate_html(sims_json, title="energysim Result Explorer", time_label="Time (h)"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sims_js = json.dumps(sims_json)

    # The entire dashboard is a single HTML string with embedded CSS + JS.
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>""" + title + r"""</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" charset="utf-8"></script>
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

// helper: escape single quotes for inline onclick attributes
function esc(s) { return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

// ==================================================================
//  VARIABLE TREE (left panel)
// ==================================================================
function buildVarTree() {
    const el = document.getElementById('varTree');
    let h = '';
    SIMS.forEach((sim, si) => {
        const col = PALETTE[si % PALETTE.length];
        h += '<div class="sim-node">' +
            '<div class="sim-hdr" onclick="toggleSim(this)">' +
                '<span class="dot" style="background:'+col+'"></span>' +
                '<span class="name">'+sim.name+'</span>' +
                '<span class="cnt">'+sim.variables.length+'</span>' +
                '<span class="chev">&#9654;</span>' +
            '</div>' +
            '<div class="sim-vars" id="sv-'+sim.name+'">';
        sim.variables.forEach(function(v) {
            h += '<div class="var-row" draggable="true"' +
                 ' data-sim="'+sim.name+'" data-var="'+v.replace(/"/g,'&quot;')+'"' +
                 ' ondragstart="onDragStart(event)"' +
                 ' ondragend="onDragEnd(event)">' +
                '<span class="vname" title="'+sim.name+'.'+v+'">'+v+'</span>' +
                '<span class="add-btn" onclick="clickAdd(\''+esc(sim.name)+'\',\''+esc(v)+'\')">+</span>' +
            '</div>';
        });
        h += '</div></div>';
    });
    el.innerHTML = h;
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
    var h = '';
    subplots.forEach(function(sp) {
        var isActive = sp.id === activeSpId;
        h += '<div class="sp-card'+(isActive?' drop-target':'')+'"' +
             ' onclick="setActiveSubplot('+sp.id+')"' +
             ' ondragover="onSpDragOver(event)"' +
             ' ondragleave="onSpDragLeave(event)"' +
             ' ondrop="onSpDrop(event,'+sp.id+')"' +
             ' style="'+(isActive?'border-color:var(--accent)':'')+'">' +
            '<div class="sp-card-hdr">' +
                '<span class="sp-color" style="background:'+sp.color+'"></span>' +
                '<input class="sp-title" value="'+sp.title.replace(/"/g,'&quot;')+'"' +
                '  onchange="renameSubplot('+sp.id+',this.value)"' +
                '  onclick="event.stopPropagation()">' +
                '<span class="sp-del" ' +
                'onclick="event.stopPropagation();removeSubplot('+sp.id+')" ' +
                'title="Delete subplot">&times;</span>' +
            '</div>' +
            '<div class="sp-card-body'+(sp.vars.length===0?' empty-hint':'')+'">';
        if (sp.vars.length === 0) {
            h += 'Drag variables here or click + in the tree';
        } else {
            sp.vars.forEach(function(v) {
                h += '<div class="sp-var">' +
                    '<span class="sv-name" title="'+v.sim+'.'+v.varName+'">'+v.sim+'.'+v.varName+'</span>' +
                    '<span class="sv-rm" ' +
                    'onclick="event.stopPropagation();removeVarFromSubplot(' +
                    sp.id+',\''+esc(v.sim)+'\',\''+esc(v.varName) +
                    '\')">&times;</span>' +
                '</div>';
            });
        }
        h += '</div></div>';
    });
    el.innerHTML = h;
}

// ==================================================================
//  TIME LABEL  (injected by Python based on auto-detected unit)
// ==================================================================
var TIME_LABEL = """ + json.dumps(time_label) + r""";

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
                x: simData.time,
                y: simData.data[v.varName],
                type: 'scatter',
                mode: 'lines',
                name: v.sim + '.' + v.varName,
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
            title: sp.title,
        };
    });

    // Subplot title annotations
    layout.annotations = activeSubplots.map(function(sp, si) {
        var axIdx = si + 1;
        var xRef = si === 0 ? 'x' : 'x' + axIdx;
        var yRef = si === 0 ? 'y' : 'y' + axIdx;
        return {
            text: '<b>' + sp.title + '</b>',
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

function exportHTML() {
    var plotDiv = document.getElementById('plotArea');
    var d = plotDiv.data;
    var l = plotDiv.layout;
    if (!d || d.length === 0) { alert('Nothing to export.'); return; }
    var blob = new Blob([
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>energysim export</title>' +
        '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"><\/script></head>' +
        '<body style="margin:0;background:#121220"><div id="p"></div><script>' +
        'Plotly.newPlot("p",' + JSON.stringify(d) + ',' + JSON.stringify(l) + ',{responsive:true});' +
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
    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dashboard(results_dict, output_path=None, auto_open=True,
                       title="energysim Result Explorer", time_label="Time (h)"):
    """Generate an interactive HTML result explorer from simulation results.

    Parameters
    ----------
    results_dict : dict
        Dictionary of ``{sim_name: pandas.DataFrame}`` as returned by
        ``world.results()``.  The *time* column should already be
        converted to the desired display unit.
    output_path : str or None
        Where to write the HTML file.  If *None*, a temp file is created.
    auto_open : bool
        If True, opens the dashboard in the default web browser.
    title : str
        Page title.
    time_label : str
        X-axis label for time, e.g. ``'Time (h)'`` or ``'Time (s)'``.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    sims_json = _build_traces_json(results_dict)
    html = _generate_html(sims_json, title=title, time_label=time_label)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="energysim_dashboard_")
        os.close(fd)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    if auto_open:
        webbrowser.open("file://" + os.path.abspath(output_path).replace("\\", "/"))

    return os.path.abspath(output_path)
