/* ============================================================
   vqc-molecule-gym · dashboard rendering (ECharts, light theme)
   Reads assets/data/*.json produced by
   scripts/build_dashboard_data.py and renders the figure panels.
   ============================================================ */
(function () {
  "use strict";

  // ---- Palette (mirrors CSS custom properties; muted academic) ----
  var C = {
    beam: "#1b6ca8",     /* beam_search — steel blue */
    greedy: "#b8791f",   /* greedy — ochre */
    random: "#8a94a3",   /* random — grey */
    ref: "#1b2230",      /* exact reference — near-black, dashed */
    accent2: "#5a8fb0",  /* secondary steel blue */
    good: "#4b7d52",     /* under chemical accuracy */
    warn: "#b8791f",     /* intermediate error */
    bad: "#a14040",      /* large error */
    ink: "#1b2230",
    muted: "#6b7280",
    line: "#e7eaef",
    panel: "#ffffff"
  };
  var AGENT_COLOR = { random: C.random, greedy: C.greedy, beam_search: C.beam };
  var AGENT_LABEL = { random: "Random", greedy: "Greedy", beam_search: "Beam search" };
  var FONT = "Inter, system-ui, sans-serif";

  var charts = []; // for resize

  // ---- ECharts light theme ----
  function registerTheme() {
    echarts.registerTheme("qchem", {
      color: [C.beam, C.greedy, C.random, C.accent2, C.ref],
      backgroundColor: "transparent",
      textStyle: { color: C.ink, fontFamily: FONT },
      title: { textStyle: { color: C.ink } },
      legend: { textStyle: { color: C.ink_2 || C.ink }, inactiveColor: "#c2c8d0" },
      tooltip: {
        backgroundColor: "rgba(255,255,255,0.98)",
        borderColor: C.line,
        textStyle: { color: C.ink, fontFamily: FONT },
        extraCssText: "border-radius:6px;box-shadow:0 4px 16px rgba(16,24,40,0.12);"
      },
      categoryAxis: axisTheme(),
      valueAxis: axisTheme(),
      logAxis: axisTheme(),
      grid: { borderColor: C.line }
    });
  }
  function axisTheme() {
    return {
      axisLine: { lineStyle: { color: "#c9ced6" } },
      axisTick: { lineStyle: { color: "#c9ced6" } },
      axisLabel: { color: C.muted, fontFamily: FONT },
      splitLine: { lineStyle: { color: "#eef1f4" } },
      nameTextStyle: { color: C.muted }
    };
  }

  function mk(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    var c = echarts.init(el, "qchem", { renderer: "canvas" });
    charts.push(c);
    return c;
  }
  function empty(id, msg) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = '<div class="chart-empty">' + (msg || "no data") + "</div>";
  }

  var BASE_GRID = { left: 56, right: 18, top: 36, bottom: 42, containLabel: true };

  // ===========================================================
  // Hero stat row
  // ===========================================================
  function renderSummary(s) {
    if (!s) return;
    set("stat-eval", s.total_evaluations);
    set("stat-tasks", s.n_tasks);
    set("stat-bench", s.n_benchmarks);
    set("stat-agents", s.n_agents);
    set("stat-chemacc", s.chem_acc_evaluations);
  }
  function set(id, v) {
    var el = document.getElementById(id);
    if (el && v != null) el.textContent = v;
  }

  // ===========================================================
  // Dashboard 1 — Potential energy surfaces
  // ===========================================================
  var pesData = null, pesEnergy = null, pesError = null, pesActive = null;

  function renderPES(data) {
    pesData = data;
    var keys = Object.keys(data.benchmarks);
    if (!keys.length) { empty("pes-energy"); empty("pes-error"); return; }

    var tabs = document.getElementById("pes-tabs");
    keys.forEach(function (k) {
      var b = document.createElement("button");
      b.className = "tab";
      b.type = "button";
      b.setAttribute("role", "tab");
      b.innerHTML = data.benchmarks[k].label;
      b.addEventListener("click", function () { selectPES(k); });
      b.dataset.key = k;
      tabs.appendChild(b);
    });

    pesEnergy = mk("pes-energy");
    pesError = mk("pes-error");
    selectPES(keys[0]);
  }

  function selectPES(key) {
    pesActive = key;
    var tabs = document.querySelectorAll("#pes-tabs .tab");
    tabs.forEach(function (t) {
      var on = t.dataset.key === key;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    var b = pesData.benchmarks[key];
    var thr = pesData.chem_acc_mha;
    var sub = document.getElementById("pes-energy-sub");
    if (sub) sub.textContent = b.scan_axis;

    var xs = b.points.map(function (p) { return p.scan_value; });
    var discovered = b.points.map(function (p) { return [p.scan_value, p.energy]; });
    var reference = b.points.map(function (p) { return [p.scan_value, p.reference]; });

    pesEnergy.setOption({
      grid: BASE_GRID,
      legend: { top: 4, right: 8, data: ["Discovered", "Exact reference"] },
      tooltip: {
        trigger: "axis",
        valueFormatter: function (v) { return (v == null ? "" : v.toFixed(5) + " Ha"); }
      },
      xAxis: { type: "value", name: b.scan_axis, nameLocation: "middle", nameGap: 28, scale: true, min: "dataMin", max: "dataMax" },
      yAxis: { type: "value", name: "Energy (Ha)", scale: true },
      series: [
        {
          name: "Discovered", type: "line", data: discovered, smooth: true,
          symbol: "circle", symbolSize: 7, lineStyle: { width: 2.5, color: C.beam },
          itemStyle: { color: C.beam },
          areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: "rgba(27,108,168,0.14)" }, { offset: 1, color: "rgba(27,108,168,0)" }]) }
        },
        {
          name: "Exact reference", type: "line", data: reference, smooth: true,
          symbol: "emptyCircle", symbolSize: 6,
          lineStyle: { width: 2, color: C.ref, type: "dashed" }, itemStyle: { color: C.ref }
        }
      ]
    }, true);

    var errPts = b.points.map(function (p) {
      return {
        value: [p.scan_value, Math.max(p.error_mha, 1e-4)],
        itemStyle: { color: p.error_mha < thr ? C.good : (p.error_mha < 10 ? C.warn : C.bad) }
      };
    });

    pesError.setOption({
      grid: BASE_GRID,
      tooltip: {
        trigger: "axis",
        valueFormatter: function (v) { return v.toFixed(3) + " mHa"; }
      },
      xAxis: { type: "value", name: b.scan_axis, nameLocation: "middle", nameGap: 28, scale: true, min: "dataMin", max: "dataMax" },
      yAxis: { type: "log", name: "Error (mHa)" },
      series: [{
        name: "Energy error", type: "line", data: errPts, smooth: false,
        symbolSize: 8, lineStyle: { width: 2, color: "rgba(125,138,153,0.6)" },
        markLine: {
          silent: true, symbol: "none",
          lineStyle: { color: C.good, type: "dashed", width: 1.5 },
          label: { color: C.good, formatter: "chem. acc. 1.6 mHa", position: "insideEndTop" },
          data: [{ yAxis: thr }]
        }
      }]
    }, true);
  }

  // ===========================================================
  // Dashboard 2 — Algorithm comparison
  // ===========================================================
  function renderAlgo(data) {
    var labels = data.benchmarks.map(function (b) { return b.label; });
    var agents = data.agents;

    var errSeries = agents.map(function (a) {
      return {
        name: AGENT_LABEL[a], type: "bar",
        itemStyle: { color: AGENT_COLOR[a], borderRadius: [3, 3, 0, 0] },
        data: data.benchmarks.map(function (b) {
          var x = b.by_agent[a];
          return x ? Math.max(x.median_err, 1e-4) : null;
        })
      };
    });

    var ce = mk("algo-error");
    if (ce) ce.setOption({
      grid: BASE_GRID,
      legend: { top: 4, data: agents.map(function (a) { return AGENT_LABEL[a]; }) },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        valueFormatter: function (v) { return v == null ? "—" : v.toFixed(3) + " mHa"; } },
      xAxis: { type: "category", data: labels, axisLabel: { interval: 0 } },
      yAxis: { type: "log", name: "Median error (mHa)" },
      series: errSeries
    });

    var hitSeries = agents.map(function (a) {
      return {
        name: AGENT_LABEL[a], type: "bar",
        itemStyle: { color: AGENT_COLOR[a], borderRadius: [3, 3, 0, 0] },
        data: data.benchmarks.map(function (b) {
          var x = b.by_agent[a];
          return x && x.n_tasks ? Math.round((x.chem_acc_hits / x.n_tasks) * 100) : 0;
        })
      };
    });

    var ch = mk("algo-hits");
    if (ch) ch.setOption({
      grid: BASE_GRID,
      legend: { top: 4, data: agents.map(function (a) { return AGENT_LABEL[a]; }) },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        valueFormatter: function (v) { return v + "%"; } },
      xAxis: { type: "category", data: labels, axisLabel: { interval: 0 } },
      yAxis: { type: "value", name: "Hit rate (%)", max: 100, min: 0 },
      series: hitSeries
    });
  }

  // ===========================================================
  // Dashboard 3 — Reward & circuit metrics
  // ===========================================================
  function renderRewards(data) {
    var keys = data.component_keys;
    var compLabel = {
      accuracy: "Energy error", chemical_accuracy: "Chem-acc",
      depth: "Depth", shots: "Shots", compactness: "Compactness"
    };
    var compColor = [C.beam, C.good, C.accent2, C.greedy, C.ref];

    // best (max reward) point per benchmark
    var best = {};
    var order = [];
    data.points.forEach(function (p) {
      if (!best[p.benchmark] || p.reward > best[p.benchmark].reward) {
        if (!best[p.benchmark]) order.push(p.benchmark);
        best[p.benchmark] = p;
      }
    });
    var benches = order;
    var labels = benches.map(function (k) { return best[k].label; });

    var series = keys.map(function (k, i) {
      return {
        name: compLabel[k] || k, type: "bar", stack: "reward",
        itemStyle: { color: compColor[i % compColor.length] },
        emphasis: { focus: "series" },
        data: benches.map(function (b) { return best[b].components[k]; })
      };
    });

    var rc = mk("reward-components");
    if (rc) rc.setOption({
      grid: { left: 56, right: 18, top: 44, bottom: 36, containLabel: true },
      legend: { top: 6 },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        valueFormatter: function (v) { return v == null ? "" : v.toFixed(3); } },
      xAxis: { type: "category", data: labels, axisLabel: { interval: 0 } },
      yAxis: { type: "value", name: "Reward" },
      series: series
    });

    // Pareto scatter: depth (x, log) vs error (y, log) per agent
    var byAgent = {};
    data.points.forEach(function (p) {
      if (p.depth == null) return;
      (byAgent[p.agent] = byAgent[p.agent] || []).push({
        value: [Math.max(p.depth, 1), Math.max(p.error_mha, 1e-3)],
        bench: p.label, agent: p.agent, reward: p.reward,
        gates: p.gate_count, ops: p.num_operators
      });
    });
    var scSeries = Object.keys(byAgent).map(function (a) {
      return {
        name: AGENT_LABEL[a] || a, type: "scatter", symbolSize: 11,
        itemStyle: { color: AGENT_COLOR[a], opacity: 0.82, borderColor: "rgba(0,0,0,0.3)" },
        data: byAgent[a]
      };
    });

    var rp = mk("reward-pareto");
    if (rp) rp.setOption({
      grid: BASE_GRID,
      legend: { top: 4 },
      tooltip: {
        trigger: "item",
        formatter: function (o) {
          var d = o.data;
          return "<b>" + d.bench + "</b> · " + (AGENT_LABEL[d.agent] || d.agent) +
            "<br/>depth " + d.value[0] + " · " + d.gates + " gates · " + d.ops + " ops" +
            "<br/>error " + d.value[1].toFixed(3) + " mHa · reward " + d.reward.toFixed(3);
        }
      },
      xAxis: { type: "log", name: "Circuit depth", nameLocation: "middle", nameGap: 28 },
      yAxis: { type: "log", name: "Error (mHa)" },
      series: scSeries
    });
  }

  // ===========================================================
  // Dashboard 4 — Benchmark / curriculum overview
  // ===========================================================
  function renderOverview(data) {
    var rows = data.ceiling;
    if (rows && rows.length) renderHeatmap(rows);
    if (data.curriculum) renderCurriculum(data.curriculum);
  }

  function renderHeatmap(rows) {
    var cols = ["Min", "Median", "Max"];
    var yLabels = rows.map(function (r) { return r.label; });
    var cells = [];
    var maxLog = -Infinity, minLog = Infinity;
    rows.forEach(function (r, yi) {
      [r.min_err, r.median_err, r.max_err].forEach(function (v, xi) {
        var lv = Math.log10(Math.max(v, 0.001));
        maxLog = Math.max(maxLog, lv); minLog = Math.min(minLog, lv);
        cells.push({ value: [xi, yi, lv], real: v });
      });
    });

    var hm = mk("overview-heatmap");
    if (!hm) return;
    hm.setOption({
      grid: { left: 96, right: 28, top: 22, bottom: 70, containLabel: true },
      tooltip: {
        position: "top",
        formatter: function (o) {
          return rows[o.data.value[1]].label + " · " + cols[o.data.value[0]] +
            "<br/><b>" + o.data.real.toFixed(3) + " mHa</b>";
        }
      },
      xAxis: { type: "category", data: cols, splitArea: { show: false }, axisLine: { show: false }, axisTick: { show: false } },
      yAxis: { type: "category", data: yLabels, inverse: true, splitArea: { show: false }, axisLine: { show: false }, axisTick: { show: false } },
      visualMap: {
        min: minLog, max: maxLog, calculable: true, orient: "horizontal",
        left: "center", bottom: 6, dimension: 2,
        text: ["higher error", "lower error"],
        textStyle: { color: C.muted },
        inRange: { color: ["#fff7ec", "#fee8c8", "#fdbb84", "#e34a33", "#b30000"] }
      },
      series: [{
        type: "heatmap", data: cells,
        label: {
          show: true, fontFamily: FONT, fontWeight: 600, fontSize: 11,
          color: function (o) {
            var span = (maxLog - minLog) || 1;
            var n = (o.data.value[2] - minLog) / span;
            return n > 0.55 ? "#ffffff" : "#1b2230";
          },
          formatter: function (o) { return o.data.real.toFixed(o.data.real < 10 ? 2 : 0); }
        },
        itemStyle: { borderColor: C.panel, borderWidth: 3, borderRadius: 2 },
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: "rgba(16,24,40,0.25)" } }
      }]
    });
  }

  function renderCurriculum(tiers) {
    var host = document.getElementById("curriculum");
    if (!host) return;
    var cls = { Easy: "easy", Medium: "medium", Hard: "hard" };
    host.innerHTML = "";
    tiers.forEach(function (t) {
      var card = document.createElement("div");
      card.className = "tier-card " + (cls[t.tier] || "");
      var mols = t.benchmarks.map(function (b) { return "<span>" + b.label + "</span>"; }).join("");
      card.innerHTML =
        '<span class="badge">' + t.tier + " curriculum</span>" +
        '<div class="mols">' + mols + "</div>" +
        "<p>" + t.note + "</p>";
      host.appendChild(card);
    });
  }

  // ===========================================================
  // Molecule gallery — lightweight SVG ball-and-stick
  // ===========================================================
  var EL = {
    H:  { r: 0.30, fill: "#d4d9e0", light: "#f3f5f8", stroke: "#aab2bd" },
    C:  { r: 0.46, fill: "#3a3f4a", light: "#5b616d", stroke: "#23272e" },
    N:  { r: 0.46, fill: "#2c5aa0", light: "#5180c8", stroke: "#1d3c6e" },
    O:  { r: 0.44, fill: "#c0392b", light: "#e0594b", stroke: "#8c2a20" },
    Li: { r: 0.52, fill: "#9b59b6", light: "#bd80d3", stroke: "#6f3f86" }
  };

  // atoms: [element, x, y, z] (Angstrom). bonds: [i, j, order] (0 = H-bond, dashed).
  var MOLECULES = [
    { id: "h2", label: "H₂", sub: "bond-length scan",
      atoms: [["H", -0.37, 0, 0], ["H", 0.37, 0, 0]], bonds: [[0, 1, 1]] },
    { id: "lih", label: "LiH", sub: "bond-length scan",
      atoms: [["Li", -0.80, 0, 0], ["H", 0.80, 0, 0]], bonds: [[0, 1, 1]] },
    { id: "n2", label: "N₂", sub: "bond-length scan",
      atoms: [["N", -0.55, 0, 0], ["N", 0.55, 0, 0]], bonds: [[0, 1, 3]] },
    { id: "h4", label: "H₄", sub: "linear, bond-length scan",
      atoms: [["H", -1.5, 0, 0], ["H", -0.5, 0, 0], ["H", 0.5, 0, 0], ["H", 1.5, 0, 0]],
      bonds: [[0, 1, 1], [1, 2, 1], [2, 3, 1]] },
    { id: "h2o", label: "H₂O", sub: "H-O-H angle scan",
      atoms: [["O", 0, 0, 0], ["H", 0.757, -0.587, 0], ["H", -0.757, -0.587, 0]],
      bonds: [[0, 1, 1], [0, 2, 1]] },
    { id: "c2h6", label: "C₂H₆", sub: "torsion scan",
      atoms: [["C", -0.77, 0, 0], ["C", 0.77, 0, 0],
        ["H", -1.134, 0.514, 0.890], ["H", -1.134, -1.028, 0], ["H", -1.134, 0.514, -0.890],
        ["H", 1.134, 1.028, 0], ["H", 1.134, -0.514, 0.890], ["H", 1.134, -0.514, -0.890]],
      bonds: [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1]] },
    { id: "h2o_dimer", label: "(H₂O)₂", sub: "O···O distance scan",
      atoms: [["O", -1.45, 0, 0], ["O", 1.45, 0, 0],
        ["H", -2.21, 0.588, 0], ["H", -2.21, -0.588, 0],
        ["H", 0.49, 0, 0], ["H", 2.04, 0.759, 0]],
      bonds: [[0, 2, 1], [0, 3, 1], [1, 4, 1], [1, 5, 1], [4, 0, 0]] }
  ];

  function moleculeSVG(mol, w, h) {
    var b = 26 * Math.PI / 180, a = 12 * Math.PI / 180;
    var cb = Math.cos(b), sb = Math.sin(b), ca = Math.cos(a), sa = Math.sin(a);
    var P = mol.atoms.map(function (at, i) {
      var x = at[1], y = at[2], z = at[3];
      var x1 = x * cb + z * sb, z1 = -x * sb + z * cb;
      var y2 = y * ca - z1 * sa, z2 = y * sa + z1 * ca;
      return { i: i, el: at[0], X: x1, Y: y2, Z: z2 };
    });
    var maxR = Math.max.apply(null, mol.atoms.map(function (at) { return EL[at[0]].r; }));
    var xs = P.map(function (p) { return p.X; }), ys = P.map(function (p) { return p.Y; });
    var minX = Math.min.apply(null, xs) - maxR, maxX = Math.max.apply(null, xs) + maxR;
    var minY = Math.min.apply(null, ys) - maxR, maxY = Math.max.apply(null, ys) + maxR;
    var spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    var pad = 10;
    var scale = Math.min((w - 2 * pad) / spanX, (h - 2 * pad) / spanY);
    var offX = (w - spanX * scale) / 2, offY = (h - spanY * scale) / 2;
    function SX(X) { return offX + (X - minX) * scale; }
    function SY(Y) { return offY + (maxY - Y) * scale; }

    var used = {};
    P.forEach(function (p) { used[p.el] = true; });
    var defs = "<defs>" + Object.keys(used).map(function (el) {
      var e = EL[el];
      return '<radialGradient id="mg-' + mol.id + '-' + el + '" cx="34%" cy="30%" r="72%">' +
        '<stop offset="0%" stop-color="' + e.light + '"/>' +
        '<stop offset="100%" stop-color="' + e.fill + '"/></radialGradient>';
    }).join("") + "</defs>";

    var bw = Math.max(2, scale * 0.10);
    var bonds = mol.bonds.map(function (bd) {
      var p1 = P[bd[0]], p2 = P[bd[1]];
      var x1 = SX(p1.X), y1 = SY(p1.Y), x2 = SX(p2.X), y2 = SY(p2.Y);
      var order = bd[2];
      if (order === 0) {
        return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 +
          '" stroke="#9aa3b0" stroke-width="' + Math.max(1.5, bw * 0.7) +
          '" stroke-dasharray="3 3" stroke-linecap="round"/>';
      }
      var dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      var ux = -dy / len, uy = dx / len, off = bw * 1.5;
      var offsets = order === 1 ? [0] : order === 2 ? [-off, off] : [-off * 1.4, 0, off * 1.4];
      return offsets.map(function (o) {
        return '<line x1="' + (x1 + ux * o) + '" y1="' + (y1 + uy * o) +
          '" x2="' + (x2 + ux * o) + '" y2="' + (y2 + uy * o) +
          '" stroke="#aeb6c0" stroke-width="' + bw + '" stroke-linecap="round"/>';
      }).join("");
    }).join("");

    var atoms = P.slice().sort(function (p, q) { return p.Z - q.Z; }).map(function (p) {
      var e = EL[p.el];
      return '<circle cx="' + SX(p.X) + '" cy="' + SY(p.Y) + '" r="' + (e.r * scale) +
        '" fill="url(#mg-' + mol.id + '-' + p.el + ')" stroke="' + e.stroke + '" stroke-width="1"/>';
    }).join("");

    return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h +
      '" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="' + mol.label +
      ' structure">' + defs + bonds + atoms + "</svg>";
  }

  function renderMolecules() {
    var host = document.getElementById("molecule-gallery");
    if (!host) return;
    host.innerHTML = MOLECULES.map(function (mol) {
      return '<div class="mol-card"><div class="mol-view">' + moleculeSVG(mol, 172, 132) +
        '</div><div class="mol-meta"><strong>' + mol.label + "</strong><span>" + mol.sub +
        "</span></div></div>";
    }).join("");
  }

  // ===========================================================
  // Boot
  // ===========================================================
  function load(name) {
    // Prefer the inlined bundle (works when the page is opened from disk, where
    // fetch of local JSON is blocked); fall back to fetch when served over HTTP.
    var key = name.replace(/\.json$/, "");
    if (window.DASH_DATA && window.DASH_DATA[key]) {
      return Promise.resolve(window.DASH_DATA[key]);
    }
    return fetch("assets/data/" + name, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(name + " " + r.status);
      return r.json();
    });
  }

  function boot() {
    if (typeof echarts === "undefined") {
      ["pes-energy", "pes-error", "algo-error", "algo-hits", "reward-components", "reward-pareto", "overview-heatmap"]
        .forEach(function (id) { empty(id, "chart library failed to load"); });
      return;
    }
    registerTheme();
    renderMolecules();

    load("summary.json").then(renderSummary).catch(function () {});
    load("pes.json").then(renderPES).catch(function (e) { empty("pes-energy", "data unavailable"); empty("pes-error", "data unavailable"); console.error(e); });
    load("algo_comparison.json").then(renderAlgo).catch(function (e) { empty("algo-error", "data unavailable"); empty("algo-hits", "data unavailable"); console.error(e); });
    load("reward_metrics.json").then(renderRewards).catch(function (e) { empty("reward-components", "data unavailable"); empty("reward-pareto", "data unavailable"); console.error(e); });
    load("overview.json").then(renderOverview).catch(function (e) { empty("overview-heatmap", "data unavailable"); console.error(e); });

    var t;
    window.addEventListener("resize", function () {
      clearTimeout(t);
      t = setTimeout(function () { charts.forEach(function (c) { c.resize(); }); }, 120);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
