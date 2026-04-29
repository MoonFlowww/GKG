from __future__ import annotations
import json
from dataclasses import asdict
from enum import Enum
from gkg import Graph, Level


_LEVEL_COLOR  = {"MACRO": "#4a90d9", "MESO": "#7ec87e", "MICRO": "#999"}
_LEVEL_RADIUS = {"MACRO": 18, "MESO": 10, "MICRO": 5}
_EDGE_COLOR   = {
    "DEPENDS_ON": "#e06c75",
    "CALLS":      "#e5c07b",
    "IMPLEMENTS": "#c678dd",
    "OWN":        "#666",
    "SEND":       "#56b6c2",
}


def _graph_data(g: Graph) -> dict:
    nodes = []
    for n in g.nodes.values():
        lvl = n.level.value
        payload_d: dict = {}
        if n.payload:
            try:
                from enum import Enum as _Enum
                def _serial(o):
                    return o.value if isinstance(o, _Enum) else o
                from dataclasses import asdict as _asdict
                payload_d = {k: _serial(v) for k, v in _asdict(n.payload).items()}
            except Exception:
                pass
        nodes.append({
            "id":       n.id,
            "name":     n.name.split("::")[-1] or n.name,   # short display name
            "fullname": n.name,
            "level":    lvl,
            "status":   n.status.value,
            "intent":   n.intent,
            "parent":   n.parent,
            "color":    _LEVEL_COLOR[lvl],
            "radius":   _LEVEL_RADIUS[lvl],
            "payload":  payload_d,
        })
    links = []
    for e in g.edges.values():
        kind = e.kind.value
        links.append({
            "id":     e.id,
            "source": e.src,
            "target": e.dst,
            "kind":   kind,
            "color":  _EDGE_COLOR.get(kind, "#888"),
            "order":  e.order,
        })
    return {"nodes": nodes, "links": links}


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GKG Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1e1e2e; color: #cdd6f4; font-family: monospace; display: flex; height: 100vh; overflow: hidden; }
  #sidebar { width: 280px; min-width: 280px; background: #181825; border-right: 1px solid #313244; display: flex; flex-direction: column; }
  #controls { padding: 12px; border-bottom: 1px solid #313244; }
  #controls h2 { font-size: 13px; color: #89b4fa; margin-bottom: 10px; }
  .ctrl-row { display: flex; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
  button { background: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px 8px; font-size: 11px; cursor: pointer; border-radius: 3px; }
  button:hover { background: #45475a; }
  button.active { background: #89b4fa; color: #1e1e2e; }
  #legend { padding: 10px 12px; border-bottom: 1px solid #313244; }
  #legend h3 { font-size: 11px; color: #a6adc8; margin-bottom: 6px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; margin-bottom: 3px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .line-seg { width: 20px; height: 2px; flex-shrink: 0; }
  #stats { padding: 8px 12px; font-size: 10px; color: #6c7086; border-bottom: 1px solid #313244; }
  #detail { padding: 12px; flex: 1; overflow-y: auto; font-size: 11px; }
  #detail h3 { color: #89b4fa; margin-bottom: 8px; font-size: 12px; }
  .detail-row { margin-bottom: 4px; }
  .detail-key { color: #a6adc8; }
  .detail-val { color: #cdd6f4; word-break: break-all; }
  #graph-area { flex: 1; position: relative; overflow: hidden; }
  svg { width: 100%; height: 100%; }
  .node circle { stroke-width: 1.5px; cursor: pointer; }
  .node text { font-size: 9px; fill: #cdd6f4; pointer-events: none; text-anchor: middle; dominant-baseline: middle; }
  .link { stroke-opacity: 0.6; fill: none; }
  .link.highlighted { stroke-opacity: 1; stroke-width: 2.5px !important; }
  .node.dimmed circle { opacity: 0.2; }
  .node.dimmed text { opacity: 0.2; }
  .link.dimmed { stroke-opacity: 0.08; }
  marker path { stroke: none; }
</style>
</head>
<body>
<div id="sidebar">
  <div id="controls">
    <h2>GKG Visualizer</h2>
    <div class="ctrl-row">
      <button id="btn-macro" class="active" onclick="toggleLevel('MACRO',this)">MACRO</button>
      <button id="btn-meso"  class="active" onclick="toggleLevel('MESO',this)">MESO</button>
      <button id="btn-micro" onclick="toggleLevel('MICRO',this)">MICRO</button>
    </div>
    <div class="ctrl-row">
      <button onclick="resetZoom()">Reset zoom</button>
      <button onclick="reheat()">Reheat</button>
    </div>
  </div>
  <div id="legend">
    <h3>Nodes</h3>
    <div class="legend-item"><div class="dot" style="background:#4a90d9"></div>MACRO (module)</div>
    <div class="legend-item"><div class="dot" style="background:#7ec87e"></div>MESO (class)</div>
    <div class="legend-item"><div class="dot" style="background:#999"></div>MICRO (function)</div>
    <h3 style="margin-top:8px">Edges</h3>
    <div class="legend-item"><div class="line-seg" style="background:#e06c75"></div>DEPENDS_ON</div>
    <div class="legend-item"><div class="line-seg" style="background:#e5c07b"></div>CALLS</div>
    <div class="legend-item"><div class="line-seg" style="background:#c678dd"></div>IMPLEMENTS</div>
    <div class="legend-item"><div class="line-seg" style="background:#56b6c2"></div>SEND</div>
    <div class="legend-item"><div class="line-seg" style="background:#666"></div>OWN</div>
  </div>
  <div id="stats"></div>
  <div id="detail"><h3>Click a node</h3></div>
</div>
<div id="graph-area">
  <svg id="svg"></svg>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const RAW = __GRAPH_DATA__;

let visibleLevels = new Set(["MACRO","MESO"]);

function activeNodes() {
  return RAW.nodes.filter(n => visibleLevels.has(n.level));
}
function activeLinks() {
  const ids = new Set(activeNodes().map(n=>n.id));
  return RAW.links.filter(l => ids.has(l.source.id||l.source) && ids.has(l.target.id||l.target));
}

document.getElementById('stats').textContent =
  `nodes: ${RAW.nodes.length}  edges: ${RAW.links.length}`;

const svg = d3.select("#svg");
const g   = svg.append("g");

// arrowhead markers per edge kind
const kinds = ["DEPENDS_ON","CALLS","IMPLEMENTS","OWN","SEND"];
const edgeColor = {"DEPENDS_ON":"#e06c75","CALLS":"#e5c07b","IMPLEMENTS":"#c678dd","OWN":"#666","SEND":"#56b6c2"};
const defs = svg.append("defs");
kinds.forEach(k => {
  defs.append("marker")
    .attr("id","arrow-"+k)
    .attr("viewBox","0 -4 8 8")
    .attr("refX",12).attr("refY",0)
    .attr("markerWidth",6).attr("markerHeight",6)
    .attr("orient","auto")
    .append("path")
      .attr("d","M0,-4L8,0L0,4")
      .attr("fill", edgeColor[k]||"#888");
});

let linkSel, nodeSel, sim;

function build() {
  g.selectAll("*").remove();
  const nodes = activeNodes().map(d=>({...d}));
  const links = activeLinks().map(d=>({...d}));

  linkSel = g.append("g").selectAll("line")
    .data(links).join("line")
    .attr("class","link")
    .attr("stroke", d=>d.color)
    .attr("stroke-width", d=> d.kind==="DEPENDS_ON"?2:1)
    .attr("marker-end", d=>`url(#arrow-${d.kind})`);

  nodeSel = g.append("g").selectAll(".node")
    .data(nodes).join("g")
    .attr("class","node")
    .call(d3.drag()
      .on("start",(e,d)=>{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on("drag", (e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on("end",  (e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; })
    )
    .on("click", (e,d)=>{ e.stopPropagation(); showDetail(d); highlightNode(d); })
    .on("mouseenter",(e,d)=>highlightNode(d))
    .on("mouseleave",()=>clearHighlight());

  nodeSel.append("circle")
    .attr("r", d=>d.radius)
    .attr("fill", d=>d.color)
    .attr("stroke", d=>d3.color(d.color).darker(0.8));

  nodeSel.filter(d=>d.level!=="MICRO")
    .append("text")
    .attr("dy","0.1em")
    .text(d=>d.name.length>18?d.name.slice(0,16)+"…":d.name);

  sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d=>d.id)
      .distance(d=>{
        const sl=d.source.level||"MICRO", tl=d.target.level||"MICRO";
        if(sl==="MACRO"||tl==="MACRO") return 160;
        if(sl==="MESO"||tl==="MESO") return 80;
        return 40;
      })
    )
    .force("charge", d3.forceManyBody().strength(d=>d.level==="MACRO"?-400:d.level==="MESO"?-120:-30))
    .force("center", d3.forceCenter(
      document.getElementById("graph-area").clientWidth/2,
      document.getElementById("graph-area").clientHeight/2))
    .force("collide", d3.forceCollide(d=>d.radius+4))
    .on("tick",()=>{
      linkSel
        .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
        .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
      nodeSel.attr("transform",d=>`translate(${d.x},${d.y})`);
    });
}

// zoom
svg.call(d3.zoom().scaleExtent([0.1,8]).on("zoom",e=>g.attr("transform",e.transform)));
svg.on("click",()=>clearHighlight());

function highlightNode(d) {
  if(!linkSel||!nodeSel) return;
  const connectedIds = new Set([d.id]);
  linkSel.each(ld=>{
    const s=ld.source.id||ld.source, t=ld.target.id||ld.target;
    if(s===d.id||t===d.id){ connectedIds.add(s); connectedIds.add(t); }
  });
  nodeSel.classed("dimmed", n=>!connectedIds.has(n.id));
  linkSel.classed("dimmed", ld=>{
    const s=ld.source.id||ld.source, t=ld.target.id||ld.target;
    return s!==d.id && t!==d.id;
  });
  linkSel.classed("highlighted", ld=>{
    const s=ld.source.id||ld.source, t=ld.target.id||ld.target;
    return s===d.id||t===d.id;
  });
}
function clearHighlight() {
  if(!nodeSel||!linkSel) return;
  nodeSel.classed("dimmed",false);
  linkSel.classed("dimmed",false).classed("highlighted",false);
}
function showDetail(d) {
  const el = document.getElementById("detail");
  const p = d.payload||{};
  let rows = [
    ["level", d.level],
    ["status", d.status],
    ["intent", d.intent||"—"],
    ["fullname", d.fullname],
  ];
  if(p.inputs?.length)    rows.push(["inputs",  p.inputs.join(", ")]);
  if(p.outputs?.length)   rows.push(["outputs", p.outputs.join(", ")]);
  if(p.behaviors?.length) rows.push(["behaviors", p.behaviors.join(", ")]);
  if(p.language)          rows.push(["language", p.language]);
  if(p.design_pattern && p.design_pattern!=="PAT_NONE") rows.push(["pattern", p.design_pattern]);
  el.innerHTML = `<h3>${d.name}</h3>` +
    rows.map(([k,v])=>`<div class="detail-row"><span class="detail-key">${k}: </span><span class="detail-val">${v}</span></div>`).join("");
}
function toggleLevel(lvl, btn) {
  if(visibleLevels.has(lvl)) visibleLevels.delete(lvl);
  else visibleLevels.add(lvl);
  btn.classList.toggle("active");
  build();
}
function resetZoom() { svg.transition().call(d3.zoom().transform, d3.zoomIdentity); }
function reheat() { if(sim) sim.alpha(0.5).restart(); }

build();
</script>
</body>
</html>
"""


def render_html(g: Graph, output_path: str = "gkg_graph.html") -> str:
    data = _graph_data(g)
    html = _HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(data))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
