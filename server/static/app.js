// RouteDiff frontend - a calm cartographic instrument.
// Page 1 drives the route-evolution canvas (a tour emerging from noise as clean
// lines, no glow). Page 2 holds the analysis: benchmark bars, training loss,
// and the edge-confidence heatmap. Pages are toggled client-side. D3 v7.

(function () {
  "use strict";

  // Palette pulled from CSS custom properties so JS and CSS never drift.
  var css = getComputedStyle(document.documentElement);
  function v(name, fallback) {
    return (css.getPropertyValue(name) || "").trim() || fallback;
  }
  var C_NOISE = v("--noise-edge", "#9aa68c");
  var C_TOUR = v("--tour", "#ea580c");
  var C_ACCENT = v("--accent", "#c2410c");
  var C_NODE = v("--node", "#2d2a22");
  var C_NODE_RING = v("--node-ring", "#f5ecd8");
  var C_TERRAIN = v("--terrain", "#4a5d3a");
  var C_SLATE = v("--slate", "#5b6b7a");
  var C_LINE = v("--line", "#cdba94");
  var C_MUTED = v("--text-muted", "#7a6f57");

  var NN_REF = 4.018; // nearest-neighbor reference (benchmarks/compare.py)
  var TWO_OPT_REF = 3.493; // 2-opt reference

  // Speed presets: [dwell per step ms, transition ms].
  var SPEEDS = {
    slow: [1100, 720],
    normal: [620, 420],
    fast: [300, 220],
  };

  // Confidence -> color: faint contour gray-green candidate edges that warm
  // through to a confident burnt-orange tour line.
  var edgeColor = d3
    .scaleLinear()
    .domain([0, 0.55, 1])
    .range([C_NOISE, "#cf7a3c", C_TOUR])
    .interpolate(d3.interpolateRgb)
    .clamp(true);
  var heatColor = d3
    .scaleLinear()
    .domain([0, 1])
    .range(["#e6d8ba", C_TOUR])
    .interpolate(d3.interpolateRgb)
    .clamp(true);

  // Substantial, drawn-feeling geometry.
  var TOUR_W = 3.6; // final, fully inked tour stroke
  var FIRM_W = 2.6; // an edge firming up mid-denoise
  var NODE_R = 4.4; // survey point
  var NODE_R_PULSE = 7.2; // momentary scale as the path reaches it
  var SHADOW_T = 0.6; // confidence above which an edge gets ink depth

  function edgeOpacity(p, isFinal) {
    if (isFinal) return p > 0.5 ? 0.97 : 0.0;
    // Noise stays faint; firming edges climb toward solid so they read as ink.
    return Math.max(0, Math.min(0.92, Math.pow(p, 1.6) * 1.05));
  }
  function edgeWidth(p, isFinal) {
    if (isFinal) return TOUR_W;
    return 0.8 + Math.pow(p, 1.4) * (FIRM_W - 0.8);
  }

  // ---- DOM --------------------------------------------------------------

  var el = function (id) { return document.getElementById(id); };
  var citiesInput = el("cities");
  var citiesValue = el("cities-value");
  var seedInput = el("seed");
  var generateBtn = el("generate");
  var playBtn = el("play");
  var playLabel = el("play-label");
  var restartBtn = el("restart");
  var slider = el("step-slider");
  var stepLabel = el("step-label");
  var speedGroup = el("speed");
  var statusEl = el("status");
  var hint = el("canvas-hint");
  var resLength = el("res-length");
  var resTime = el("res-time");
  var aboutLength = el("about-length");
  var aboutGap = el("about-gap");
  var heatCap = el("heatmap-cap");
  var toAbout = el("to-about");
  var toDemo = el("to-demo");
  var pageDemo = el("page-demo");
  var pageAbout = el("page-about");

  // ---- SVG scaffolding (no glow, just a faint node halo) ----------------

  var svg = d3.select("#route-canvas");

  // A soft warm drop shadow so the inked route sits on the paper with depth.
  var defs = svg.append("defs");
  var inkShadow = defs
    .append("filter")
    .attr("id", "ink-shadow")
    .attr("x", "-50%")
    .attr("y", "-50%")
    .attr("width", "200%")
    .attr("height", "200%");
  inkShadow
    .append("feDropShadow")
    .attr("dx", 0.5)
    .attr("dy", 1.4)
    .attr("stdDeviation", 1.6)
    .attr("flood-color", "#7a370f")
    .attr("flood-opacity", 0.42);

  var terrainG = svg.append("g").attr("class", "terrain"); // faint contour hint
  var haloG = svg.append("g").attr("class", "halos");
  var edgesG = svg.append("g").attr("class", "edges");
  var nodesG = svg.append("g").attr("class", "nodes");

  var heatSvg = d3.select("#heatmap");
  var heatG = heatSvg.append("g");
  var benchSvg = d3.select("#benchmark");
  var benchG = benchSvg.append("g");
  var lossSvg = d3.select("#loss-curve");

  // ---- State ------------------------------------------------------------

  var state = {
    timeline: null,
    candidates: [],
    tourSet: new Set(),
    step: 0,
    maxT: 1,
    playing: false,
    timer: null,
    busy: false,
    dwellMs: SPEEDS.normal[0],
    transMs: SPEEDS.normal[1],
    heatN: -1,
    lossData: null,
  };

  var bench = [
    { name: "Diffusion (ours)", value: null, color: C_TOUR },
    { name: "Nearest-neighbor", value: NN_REF, color: C_TERRAIN },
    { name: "2-opt", value: TWO_OPT_REF, color: C_SLATE },
  ];

  // ---- Geometry ---------------------------------------------------------

  function scaler() {
    var node = svg.node();
    var w = node.clientWidth || 600;
    var h = node.clientHeight || 600;
    var pad = 48;
    var size = Math.max(10, Math.min(w, h) - 2 * pad);
    var ox = (w - size) / 2;
    var oy = (h - size) / 2;
    return {
      x: function (t) { return ox + t * size; },
      y: function (t) { return oy + t * size; },
    };
  }

  function buildCandidates(n) {
    var out = [];
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) out.push({ i: i, j: j, key: i + "-" + j });
    }
    return out;
  }

  // ---- Terrain hint (extremely faint contour lines under everything) ----

  function renderTerrain() {
    var node = svg.node();
    var w = node.clientWidth || 600;
    var h = node.clientHeight || 600;
    var lines = 7;
    var lg = d3.line().x(function (d) { return d[0]; }).y(function (d) { return d[1]; })
      .curve(d3.curveBasis);
    var data = [];
    for (var k = 0; k < lines; k++) {
      var y0 = ((k + 0.5) / lines) * h;
      var amp = 9 + (k % 3) * 6;
      var phase = k * 0.9;
      var pts = [];
      for (var x = 0; x <= w; x += Math.max(22, w / 22)) {
        pts.push([x, y0 + Math.sin(x / 130 + phase) * amp + Math.cos(x / 270 + phase) * amp * 0.45]);
      }
      data.push(pts);
    }
    var sel = terrainG.selectAll("path.contour").data(data);
    sel.enter().append("path").attr("class", "contour")
      .attr("fill", "none").attr("stroke", C_LINE).attr("stroke-width", 1)
      .attr("stroke-opacity", 0.11)
      .merge(sel)
      .attr("d", function (d) { return lg(d); });
    sel.exit().remove();
  }

  // Walk the decoded cycle so the tour can be inked in path order, with each
  // node and edge keyed to its position along the trail.
  function tourSchedule() {
    var tl = state.timeline;
    var n = tl.num_cities;
    var adj = {};
    tl.tour_edges.forEach(function (e) {
      (adj[e[0]] = adj[e[0]] || []).push(e[1]);
      (adj[e[1]] = adj[e[1]] || []).push(e[0]);
    });
    var path = [];
    var seen = {};
    var cur = 0;
    var prev = -1;
    for (var k = 0; k < n; k++) {
      path.push(cur);
      seen[cur] = true;
      var nbrs = adj[cur] || [];
      var nxt = nbrs[0] === prev ? nbrs[1] : nbrs[0];
      if (nxt == null || seen[nxt]) {
        nxt = nbrs.filter(function (x) { return !seen[x]; })[0];
        if (nxt == null) break;
      }
      prev = cur;
      cur = nxt;
    }
    var pos = {};
    var edgeOrder = {};
    for (var i = 0; i < path.length; i++) pos[path[i]] = i;
    for (var j = 0; j < path.length; j++) {
      var a = path[j];
      var b = path[(j + 1) % path.length];
      edgeOrder[a < b ? a + "-" + b : b + "-" + a] = j;
    }
    return { pos: pos, edgeOrder: edgeOrder, count: path.length };
  }

  // ---- Route rendering --------------------------------------------------

  function renderRoute(animate) {
    var tl = state.timeline;
    if (!tl) return;
    var step = tl.steps[state.step];
    var isFinal = !!step.final;
    var sc = scaler();
    var coords = tl.coords;
    renderTerrain();

    // Noise weight: 1 at the noisy start, 0 at the clean end. Confidence colors
    // each edge; visibility is blended toward the sampled noisy edge set early,
    // so the run opens as a faint web of scratches and resolves to a clean line.
    var w = isFinal ? 0 : Math.max(0, Math.min(1, step.t / (state.maxT || 1)));
    var noiseSet = null;
    if (w > 0) {
      noiseSet = new Set(step.edges.map(function (e) { return e[0] + "-" + e[1]; }));
    }

    var data = state.candidates.map(function (e) {
      var conf = isFinal ? (state.tourSet.has(e.key) ? 1 : 0) : step.probs[e.i][e.j];
      var present = noiseSet && noiseSet.has(e.key) ? 0.4 : 0;
      // Let noise recede faster than confidence rises so the tour wins contrast.
      var vis = isFinal ? conf : w * w * present + (1 - w) * conf;
      return {
        key: e.key, conf: conf, vis: vis,
        x1: sc.x(coords[e.i][0]), y1: sc.y(coords[e.i][1]),
        x2: sc.x(coords[e.j][0]), y2: sc.y(coords[e.j][1]),
        len: Math.hypot(sc.x(coords[e.j][0]) - sc.x(coords[e.i][0]),
                        sc.y(coords[e.j][1]) - sc.y(coords[e.i][1])),
      };
    });

    var drawIn = isFinal && animate; // the satisfying ink-in moment
    var dur = animate ? state.transMs : 0;
    var sched = drawIn ? tourSchedule() : null;
    var stagger = drawIn
      ? Math.max(55, Math.min(150, 1500 / Math.max(1, tl.num_cities)))
      : 0;
    var drawDur = 460;

    var sel = edgesG.selectAll("line.edge").data(data, function (d) { return d.key; });
    var entered = sel.enter()
      .append("line")
      .attr("class", "edge")
      .attr("stroke-linecap", "round")
      .attr("stroke-opacity", 0);
    var all = entered.merge(sel)
      .attr("x1", function (d) { return d.x1; })
      .attr("y1", function (d) { return d.y1; })
      .attr("x2", function (d) { return d.x2; })
      .attr("y2", function (d) { return d.y2; })
      .attr("stroke", function (d) { return edgeColor(d.conf); })
      .attr("filter", function (d) {
        return d.conf > SHADOW_T && d.vis > 0.45 ? "url(#ink-shadow)" : null;
      })
      .interrupt();

    if (drawIn) {
      all.each(function (d) {
        var line = d3.select(this);
        if (state.tourSet.has(d.key)) {
          var delay = (sched.edgeOrder[d.key] || 0) * stagger;
          line
            .attr("stroke-width", TOUR_W)
            .attr("stroke-opacity", 0.97)
            .attr("stroke-dasharray", d.len + " " + (d.len + 1))
            .attr("stroke-dashoffset", d.len)
            .transition().delay(delay).duration(drawDur).ease(d3.easeCubicOut)
            .attr("stroke-dashoffset", 0)
            .transition().duration(150).ease(d3.easeSinOut) // small settle
            .attr("stroke-width", TOUR_W * 1.16)
            .transition().duration(190).ease(d3.easeSinIn)
            .attr("stroke-width", TOUR_W);
        } else {
          line.transition().duration(dur)
            .attr("stroke-opacity", 0)
            .attr("stroke-width", 0.8);
        }
      });
    } else {
      all.transition().duration(dur)
        .attr("stroke-dasharray", null)
        .attr("stroke-dashoffset", null)
        .attr("stroke-width", function (d) { return edgeWidth(d.conf, isFinal); })
        .attr("stroke-opacity", function (d) { return edgeOpacity(d.vis, isFinal); });
    }
    sel.exit().remove();

    var nodes = coords.map(function (c, i) {
      return { i: i, x: sc.x(c[0]), y: sc.y(c[1]) };
    });
    // Very subtle halo behind each pinpoint (calm, not glowing).
    var hsel = haloG.selectAll("circle.halo").data(nodes, function (d) { return d.i; });
    hsel.enter().append("circle").attr("class", "halo").attr("r", 9)
      .attr("fill", C_TOUR).attr("fill-opacity", 0.07)
      .merge(hsel)
      .attr("cx", function (d) { return d.x; })
      .attr("cy", function (d) { return d.y; });
    hsel.exit().remove();

    var nsel = nodesG.selectAll("circle.city").data(nodes, function (d) { return d.i; });
    var nmerged = nsel.enter().append("circle").attr("class", "city").attr("r", 0)
      .attr("fill", C_NODE).attr("stroke", C_NODE_RING).attr("stroke-width", 1.5)
      .merge(nsel)
      .attr("cx", function (d) { return d.x; })
      .attr("cy", function (d) { return d.y; })
      .interrupt();

    if (drawIn) {
      nmerged.attr("r", NODE_R)
        .transition()
        .delay(function (d) { return (sched.pos[d.i] || 0) * stagger + drawDur * 0.55; })
        .duration(150).ease(d3.easeSinOut)
        .attr("r", NODE_R_PULSE)
        .transition().duration(230).ease(d3.easeBackOut.overshoot(1.4))
        .attr("r", NODE_R);
    } else {
      nmerged.transition().duration(dur).attr("r", NODE_R);
    }
    nsel.exit().remove();

    var last = tl.steps.length - 1;
    slider.value = String(state.step);
    stepLabel.textContent = state.step + " / " + last;
  }

  // ---- Heatmap (final-step confidence, shown on the about page) ---------

  function renderHeatmap(stepIndex) {
    var tl = state.timeline;
    if (!tl) return;
    var step = tl.steps[stepIndex == null ? tl.steps.length - 1 : stepIndex];
    var n = tl.num_cities;
    var size = 210;
    var cell = size / n;
    heatSvg.attr("viewBox", "0 0 " + size + " " + size);

    if (state.heatN !== n) {
      heatG.selectAll("rect").remove();
      var cells = [];
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) cells.push({ i: i, j: j });
      }
      heatG.selectAll("rect").data(cells).enter().append("rect")
        .attr("x", function (d) { return d.j * cell; })
        .attr("y", function (d) { return d.i * cell; })
        .attr("width", cell - 0.7)
        .attr("height", cell - 0.7)
        .attr("rx", 1);
      state.heatN = n;
    }

    heatG.selectAll("rect")
      .attr("fill", function (d) {
        return d.i === d.j ? "#d8c8a4" : heatColor(step.probs[d.i][d.j]);
      });
    heatSvg.classed("is-visible", true);
    heatCap.textContent = "confidence at t = 0 (final tour)";
  }

  // ---- Results ----------------------------------------------------------

  function renderResults(length, timeMs) {
    resLength.textContent = length.toFixed(3);
    resTime.textContent = Math.round(timeMs) + " ms";
    aboutLength.textContent = length.toFixed(3);
    var gap = ((length - NN_REF) / NN_REF) * 100;
    var sign = gap > 0 ? "+" : "";
    aboutGap.textContent = sign + gap.toFixed(1) + "%";
  }

  // ---- Benchmark bars ---------------------------------------------------

  function renderBenchmark(animate) {
    var node = benchSvg.node();
    var w = node.clientWidth || 600;
    var h = node.clientHeight || 168;
    benchSvg.attr("viewBox", "0 0 " + w + " " + h);
    var m = { top: 8, right: 56, bottom: 10, left: 120 };
    var iw = Math.max(10, w - m.left - m.right);
    var ih = Math.max(10, h - m.top - m.bottom);
    benchG.attr("transform", "translate(" + m.left + "," + m.top + ")");

    var maxv = Math.max(4.2, d3.max(bench, function (d) { return d.value || 0; })) * 1.06;
    var x = d3.scaleLinear().domain([0, maxv]).range([0, iw]);
    var y = d3.scaleBand().domain(bench.map(function (d) { return d.name; })).range([0, ih]).padding(0.36);
    var dur = animate ? 600 : 0;

    var rows = benchG.selectAll("g.bar").data(bench, function (d) { return d.name; });
    var enter = rows.enter().append("g").attr("class", "bar");
    enter.append("rect").attr("rx", 3).attr("width", 0);
    enter.append("text").attr("class", "bar-label");
    enter.append("text").attr("class", "bar-value");
    var merged = enter.merge(rows);

    merged.select("rect")
      .attr("y", function (d) { return y(d.name); })
      .attr("height", y.bandwidth())
      .attr("fill", function (d) { return d.color; })
      .attr("fill-opacity", function (d) { return d.value == null ? 0.2 : 0.9; })
      .interrupt().transition().duration(dur)
      .attr("width", function (d) { return x(d.value || 0); });

    merged.select("text.bar-label")
      .attr("x", -10)
      .attr("y", function (d) { return y(d.name) + y.bandwidth() / 2; })
      .attr("dy", "0.32em").attr("text-anchor", "end")
      .text(function (d) { return d.name; });

    merged.select("text.bar-value")
      .attr("y", function (d) { return y(d.name) + y.bandwidth() / 2; })
      .attr("dy", "0.32em").attr("text-anchor", "start")
      .interrupt().transition().duration(dur)
      .attr("x", function (d) { return x(d.value || 0) + 7; })
      .tween("text", function (d) {
        return function () { this.textContent = d.value == null ? "-" : d.value.toFixed(3); };
      });
  }

  // ---- Loss curve -------------------------------------------------------

  function renderLoss(steps, loss) {
    var node = lossSvg.node();
    var w = node.clientWidth || 600;
    var h = node.clientHeight || 168;
    lossSvg.attr("viewBox", "0 0 " + w + " " + h);
    var m = { top: 12, right: 14, bottom: 22, left: 40 };
    var iw = Math.max(10, w - m.left - m.right);
    var ih = Math.max(10, h - m.top - m.bottom);

    var x = d3.scaleLinear().domain(d3.extent(steps)).range([0, iw]);
    var y = d3.scaleLinear().domain([0, d3.max(loss)]).nice().range([ih, 0]);
    var line = d3.line()
      .x(function (d, i) { return x(steps[i]); })
      .y(function (d) { return y(d); })
      .curve(d3.curveMonotoneX);

    lossSvg.selectAll("*").remove();
    var g = lossSvg.append("g").attr("transform", "translate(" + m.left + "," + m.top + ")");
    var yAxis = d3.axisLeft(y).ticks(3).tickSize(0).tickPadding(6);
    g.append("g").attr("class", "loss-axis").call(yAxis).select(".domain").remove();
    g.append("path").datum(loss).attr("class", "loss-line").attr("d", line);
    g.append("text").attr("class", "axis-label")
      .attr("x", 0).attr("y", ih + 16).attr("text-anchor", "start").text("epoch 1");
    g.append("text").attr("class", "axis-label")
      .attr("x", iw).attr("y", ih + 16).attr("text-anchor", "end")
      .text("epoch " + steps[steps.length - 1]);
  }

  function loadLossCurve() {
    fetch("/api/loss-curve")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d && d.loss && d.loss.length) {
          state.lossData = d;
          if (pageAbout.classList.contains("is-active")) renderLoss(d.steps, d.loss);
        }
      })
      .catch(function () { /* loss curve is non-critical */ });
  }

  // ---- Page toggle ------------------------------------------------------

  function renderAboutVisuals() {
    renderBenchmark(false);
    if (state.lossData) renderLoss(state.lossData.steps, state.lossData.loss);
    if (state.timeline) renderHeatmap();
  }

  function showPage(name) {
    var demo = name === "demo";
    pageDemo.classList.toggle("is-active", demo);
    pageAbout.classList.toggle("is-active", !demo);
    toAbout.hidden = !demo;
    toDemo.hidden = demo;
    if (demo) {
      if (state.timeline) requestAnimationFrame(function () { renderRoute(false); });
    } else {
      // Let the page lay out before measuring SVG widths.
      requestAnimationFrame(renderAboutVisuals);
    }
  }

  // ---- Playback ---------------------------------------------------------

  function gotoStep(n, animate) {
    if (!state.timeline) return;
    var last = state.timeline.steps.length - 1;
    state.step = Math.max(0, Math.min(last, n));
    renderRoute(animate !== false);
  }

  function atEnd() {
    return !state.timeline || state.step >= state.timeline.steps.length - 1;
  }

  function stopTimer() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function setPlaying(p) {
    state.playing = p;
    playBtn.dataset.playing = p ? "true" : "false";
    playLabel.textContent = p ? "Pause" : "Play";
    playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
    stopTimer();
    if (p) {
      setStatus("playing");
      state.timer = setInterval(function () {
        if (atEnd()) { setPlaying(false); setStatus("complete"); return; }
        gotoStep(state.step + 1, true);
      }, state.dwellMs);
    } else if (state.timeline && !atEnd()) {
      setStatus("paused");
    }
  }

  function togglePlay() {
    if (!state.timeline) return;
    if (state.playing) { setPlaying(false); return; }
    if (atEnd()) gotoStep(0, false);
    setPlaying(true);
  }

  // ---- Status / controls ------------------------------------------------

  function setStatus(text) { statusEl.textContent = text; }

  function setControlsEnabled(on) {
    state.busy = !on;
    generateBtn.disabled = !on;
    var has = !!state.timeline;
    playBtn.disabled = !on || !has;
    restartBtn.disabled = !on || !has;
    slider.disabled = !on || !has;
  }

  // ---- Run --------------------------------------------------------------

  function generate() {
    if (state.busy) return;
    setPlaying(false);
    setControlsEnabled(false);
    setStatus("sampling");

    var body = { num_cities: parseInt(citiesInput.value, 10) };
    var seedRaw = seedInput.value.trim();
    if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);

    var t0 = performance.now();
    fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data && data.detail ? data.detail : "HTTP " + res.status);
          return data;
        });
      })
      .then(function (tl) {
        var elapsed = performance.now() - t0;
        state.timeline = tl;
        state.maxT = tl.steps[0] ? tl.steps[0].t : 1;
        state.candidates = buildCandidates(tl.num_cities);
        state.tourSet = new Set(tl.tour_edges.map(function (e) { return e[0] + "-" + e[1]; }));
        state.heatN = -1;
        state.step = 0;
        slider.min = "0";
        slider.max = String(tl.steps.length - 1);
        slider.value = "0";
        hint.classList.add("is-hidden");
        renderResults(Number(tl.length), elapsed);
        bench[0].value = Number(tl.length);
        setControlsEnabled(true);
        renderRoute(false);
        setPlaying(true);
      })
      .catch(function (err) {
        setControlsEnabled(true);
        setStatus("error: " + err.message);
      });
  }

  // ---- Wiring -----------------------------------------------------------

  citiesInput.addEventListener("input", function () {
    citiesValue.textContent = citiesInput.value;
  });
  generateBtn.addEventListener("click", generate);
  playBtn.addEventListener("click", togglePlay);
  restartBtn.addEventListener("click", function () {
    if (!state.timeline) return;
    setPlaying(false);
    gotoStep(0, true);
    setStatus("ready");
  });
  slider.addEventListener("input", function () {
    if (!state.timeline) return;
    setPlaying(false);
    gotoStep(parseInt(slider.value, 10), true);
  });
  speedGroup.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".seg");
    if (!btn) return;
    var preset = SPEEDS[btn.dataset.speed] || SPEEDS.normal;
    state.dwellMs = preset[0];
    state.transMs = preset[1];
    speedGroup.querySelectorAll(".seg").forEach(function (b) {
      b.classList.toggle("is-active", b === btn);
    });
    if (state.playing) setPlaying(true);
  });
  toAbout.addEventListener("click", function () { showPage("about"); });
  toDemo.addEventListener("click", function () { showPage("demo"); });

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (pageDemo.classList.contains("is-active")) {
        if (state.timeline) renderRoute(false);
      } else {
        renderAboutVisuals();
      }
    }, 120);
  });

  // ---- Init -------------------------------------------------------------

  setControlsEnabled(true);
  loadLossCurve();
  if (window.location.hash === "#diag") {
    window.addEventListener("error", function (e) { heatCap.textContent = "JSERR: " + e.message; });
    var origThen = generate;
    generate();
    var iv = setInterval(function () {
      if (state.timeline) { clearInterval(iv); showPage("about"); }
    }, 200);
  }
})();
