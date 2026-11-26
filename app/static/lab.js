// The lab page: explore thresholds, and see what breaks when a model moves.
// Every number comes from lab-data.json, built by study/lab_data.py.

const SVG = "http://www.w3.org/2000/svg";
const METHODS = {
  none: "No update",
  em: "Label-free (EM)",
  intercept: "Shift only",
  platt: "Shift and slope",
  refit: "New local model",
  rate: "Local rate only",
};
const MODEL_NAMES = {
  heart: "Heart disease", diabetes: "Diabetes", liver: "Liver disease",
  dengue: "Dengue severity", malaria: "Malaria cell",
};
const PLACES = {
  "heart: cleveland": "Cleveland", "heart: hungarian": "Hungary", "heart: switzerland": "Switzerland",
  "heart: va": "VA Long Beach", "dengue: center_west": "Center-West", "dengue: north": "North",
  "dengue: northeast": "Northeast", "dengue: south": "South", "dengue: southeast": "Southeast",
  "dengue: 2025": "Year 2025",
};
const HYPOTHESES = {
  H1: "Rates break more often than ranking (exploratory)",
  H2: "A single shift repairs most transfers",
  H3: "At the smallest sample, shift alone beats shift and slope",
  H4: "Updating the old model beats a new local model at every size",
  H5: "The label-free repair makes things worse in at least 2 targets",
};

const $ = (selector) => document.querySelector(selector);
let data = null;

function html(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const percent = (value, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "none");
// Charts scale with their box, so a narrow screen needs a narrower drawing to
// keep its text readable.
const wideChart = () => (matchMedia("(max-width: 760px)").matches ? 360 : 560);
const count = (value) => value.toLocaleString("en");
const place = (transfer) => PLACES[transfer.name] || transfer.name;

// One tooltip for every chart. Marks carry their text, so keyboard and
// screen reader users get the same words through the <title> element.
function describe(node, text) {
  node.append(svg("title", {}, text));
  const tip = $("#lab-tip");
  node.addEventListener("pointerenter", (event) => {
    tip.textContent = text;
    tip.hidden = false;
    const left = Math.min(event.clientX + 12, innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, event.clientY - tip.offsetHeight - 12)}px`;
  });
  node.addEventListener("pointerleave", () => { tip.hidden = true; });
  return node;
}

function frame(label, { width = 340, height = 260, x, y, xTicks, yTicks, xLabel, yLabel, xText, yText }) {
  const pad = { left: 52, right: 24, top: 12, bottom: 42 };
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label, class: "plot" });
  const px = (v) => pad.left + ((v - x[0]) / (x[1] - x[0])) * (width - pad.left - pad.right);
  const py = (v) => height - pad.bottom - ((v - y[0]) / (y[1] - y[0])) * (height - pad.top - pad.bottom);
  for (const tick of yTicks) {
    root.append(svg("line", { x1: pad.left, x2: width - pad.right, y1: py(tick), y2: py(tick), class: "grid" }));
    root.append(svg("text", { x: pad.left - 7, y: py(tick) + 4, "text-anchor": "end" }, yText(tick)));
  }
  for (const tick of xTicks) {
    root.append(svg("text", { x: px(tick), y: height - pad.bottom + 16, "text-anchor": "middle" }, xText(tick)));
  }
  root.append(svg("text", { x: (pad.left + width - pad.right) / 2, y: height - 6, "text-anchor": "middle" }, xLabel));
  root.append(svg("text", {
    x: -(height - pad.bottom + pad.top) / 2, y: 12, transform: "rotate(-90)", "text-anchor": "middle",
  }, yLabel));
  return { root, px, py };
}

const path = (points) => points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");

function niceTicks(low, high, target = 4) {
  const raw = (high - low) / target || 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

// Threshold explorer

export function operatingPoint(model, index) {
  const tp = model.tp[index];
  const fp = model.fp[index];
  const fn = model.positives - tp;
  const tn = model.negatives - fp;
  return { tp, fp, fn, tn, sensitivity: tp / model.positives, specificity: tn / model.negatives };
}

export function precisionAt(point, prevalence) {
  const hits = point.sensitivity * prevalence;
  const alarms = (1 - point.specificity) * (1 - prevalence);
  return hits + alarms ? hits / (hits + alarms) : NaN;
}

function drawHistogram(model, cutoff) {
  const { edges, positive, negative } = model.histogram;
  const share = (counts, total) => counts.map((c) => c / total);
  const pos = share(positive, model.positives);
  const neg = share(negative, model.negatives);
  const top = Math.max(...pos, ...neg) * 1.05;
  const last = edges.at(-1);
  const chart = frame("Scores of people with and without the condition", {
    x: [0, last], y: [0, top], xTicks: niceTicks(0, last), yTicks: niceTicks(0, top),
    xLabel: "Model score", yLabel: "Share of group",
    xText: (v) => percent(v, last < 0.1 ? 1 : 0), yText: (v) => percent(v),
  });
  for (const [shares, name, cls] of [[neg, "without", "fill-neg"], [pos, "with", "fill-pos"]]) {
    shares.forEach((value, i) => {
      const bar = svg("rect", {
        x: chart.px(edges[i]) + 0.5, width: Math.max(1, chart.px(edges[i + 1]) - chart.px(edges[i]) - 1),
        y: chart.py(value), height: chart.py(0) - chart.py(value), class: cls,
      });
      chart.root.append(describe(bar, `${percent(value, 1)} of those ${name} the condition score `
        + `${percent(edges[i], 1)} to ${percent(edges[i + 1], 1)}`));
    });
  }
  const at = chart.px(Math.min(cutoff, last));
  chart.root.append(svg("line", { x1: at, x2: at, y1: 12, y2: chart.py(0), class: "cutline" }));
  return chart.root;
}

function drawRoc(model, index) {
  const chart = frame(`ROC curve, AUC ${model.auc.toFixed(3)}`, {
    x: [0, 1], y: [0, 1], xTicks: [0, 0.5, 1], yTicks: [0, 0.25, 0.5, 0.75, 1],
    xLabel: "False alarm rate", yLabel: "Sensitivity", xText: (v) => percent(v), yText: (v) => percent(v),
  });
  chart.root.append(svg("path", { d: path([[chart.px(0), chart.py(0)], [chart.px(1), chart.py(1)]]), class: "chance" }));
  const points = model.grid.map((_, i) => [chart.px(model.fp[i] / model.negatives), chart.py(model.tp[i] / model.positives)]);
  chart.root.append(svg("path", { d: path(points), class: "curve" }));
  const now = operatingPoint(model, index);
  chart.root.append(describe(svg("circle", {
    cx: chart.px(1 - now.specificity), cy: chart.py(now.sensitivity), r: 6, class: "marker",
  }), `Sensitivity ${percent(now.sensitivity)}, false alarms ${percent(1 - now.specificity)}`));
  return chart.root;
}

function drawCalibration(model) {
  const { bins } = model.calibration;
  const chart = frame(`Calibration, expected error ${percent(model.calibration.ece, 1)}`, {
    x: [0, 1], y: [0, 1], xTicks: [0, 0.5, 1], yTicks: [0, 0.25, 0.5, 0.75, 1],
    xLabel: "Predicted risk", yLabel: "Observed rate", xText: (v) => percent(v), yText: (v) => percent(v),
  });
  chart.root.append(svg("path", { d: path([[chart.px(0), chart.py(0)], [chart.px(1), chart.py(1)]]), class: "chance" }));
  const most = Math.max(...bins.map((b) => b.rows));
  for (const bin of bins) {
    chart.root.append(describe(svg("circle", {
      cx: chart.px(bin.mean_predicted), cy: chart.py(bin.observed_rate),
      r: 4 + 5 * Math.sqrt(bin.rows / most), class: "marker marker--soft",
    }), `Predicted ${percent(bin.mean_predicted, 1)}, observed ${percent(bin.observed_rate, 1)}, ${count(bin.rows)} rows`));
  }
  return chart.root;
}

function figure(title, note, chart, legend) {
  const node = html("figure", "lab-figure");
  const caption = html("figcaption");
  caption.append(html("strong", "", title), html("span", "", note));
  node.append(caption);
  if (legend) node.append(legend);
  node.append(chart);
  return node;
}

function swatches(entries) {
  const list = html("ul", "legend");
  for (const [cls, label] of entries) {
    const item = html("li");
    const mark = svg("svg", { viewBox: "0 0 24 10", "aria-hidden": "true", class: "key" });
    mark.append(cls.startsWith("fill")
      ? svg("rect", { x: 6, y: 0, width: 12, height: 10, class: cls })
      : svg("path", { d: "M0 5L24 5", class: cls }));
    item.append(mark, html("span", "", label));
    list.append(item);
  }
  return list;
}

function table(caption, head, rows) {
  const wrap = html("div", "table-scroll");
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", caption);
  const node = html("table", "lab-table");
  node.append(html("caption", "visually-hidden", caption));
  const top = html("tr");
  head.forEach((text) => { const th = html("th", "", text); th.scope = "col"; top.append(th); });
  const thead = html("thead");
  thead.append(top);
  const body = html("tbody");
  for (const row of rows) {
    const tr = html("tr");
    row.forEach((text, i) => {
      const cell = html(i ? "td" : "th", "", text);
      if (!i) cell.scope = "row";
      tr.append(cell);
    });
    body.append(tr);
  }
  node.append(thead, body);
  wrap.append(node);
  return wrap;
}

function renderModel() {
  const model = data.models[$("#lab-model").value];
  const slider = $("#lab-cutoff");
  slider.max = model.grid.length - 1;
  slider.value = model.grid.indexOf(model.threshold);
  const natural = model.positives / (model.positives + model.negatives);
  const prevalence = $("#lab-prevalence");
  prevalence.value = (natural * 100).toFixed(1);
  $("#lab-families").replaceChildren(table(
    "Model families compared by cross-validation", ["Family", "Mean AUC", "Fold-to-fold SD"],
    model.families.map((f) => [f.family, f.roc_auc.toFixed(3), f.roc_auc_std.toFixed(3)]),
  ));
  $("#lab-inputs").replaceChildren(model.importance.length
    ? table("Inputs that matter most", ["Input", "AUC lost when shuffled"],
      model.importance.map((f) => [f.label, f.auc_drop.toFixed(3)]))
    : html("p", "lab-note", "Not measured for this model: its 181 image measurements overlap too much to rank one by one."));
  renderPoint();
}

function renderPoint() {
  const model = data.models[$("#lab-model").value];
  const slider = $("#lab-cutoff");
  const index = Number(slider.value);
  const cutoff = model.grid[index];
  const point = operatingPoint(model, index);
  const prevalence = Math.min(99.9, Math.max(0.1, Number($("#lab-prevalence").value) || 0.1)) / 100;
  const precision = precisionAt(point, prevalence);
  const served = cutoff === model.threshold ? " (the one this site uses)" : "";
  slider.setAttribute("aria-valuetext", percent(cutoff, 1));
  $("#lab-cutoff-value").textContent = `${percent(cutoff, 1)}${served}`;

  const tiles = [
    ["Cases caught", percent(point.sensitivity), `${count(point.tp)} of ${count(model.positives)}`],
    ["Cases missed", count(point.fn), "below the cutoff"],
    ["False alarms", percent(1 - point.specificity), `${count(point.fp)} of ${count(model.negatives)}`],
    ["Flags that are right", percent(precision), `if ${percent(prevalence, 1)} have the condition`],
  ].map(([label, value, note]) => {
    const tile = html("div", "lab-tile");
    tile.append(html("span", "lab-tile-label", label), html("strong", "", value), html("small", "", note));
    return tile;
  });
  $("#lab-tiles").replaceChildren(...tiles);
  $("#lab-charts").replaceChildren(
    figure("Scores", "The dashed line is the cutoff. Overlap is what no cutoff can fix.", drawHistogram(model, cutoff),
      swatches([["fill-pos", "Has the condition"], ["fill-neg", "Does not"]])),
    figure("Ranking", `AUC ${model.auc.toFixed(3)} (95% interval ${model.auc_ci[0].toFixed(3)} to ${model.auc_ci[1].toFixed(3)}).`,
      drawRoc(model, index)),
    figure("Calibration", "Dots on the diagonal mean honest probabilities. Bigger dots hold more rows.", drawCalibration(model)),
  );
}

// Transfer study

function drawRates(transfers) {
  const rowHeight = 30;
  const height = transfers.length * rowHeight + 54;
  const top = Math.max(...transfers.flatMap((t) => [t.mean_predicted, t.target_rate])) * 1.12;
  const width = wideChart();
  const left = 120;
  const x = (v) => left + (v / top) * (width - left - 20);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", class: "plot",
    "aria-label": "Predicted against real rate at each target" });
  for (const tick of niceTicks(0, top)) {
    root.append(svg("line", { x1: x(tick), x2: x(tick), y1: 8, y2: height - 44, class: "grid" }));
    root.append(svg("text", { x: x(tick), y: height - 26, "text-anchor": "middle" }, percent(tick)));
  }
  transfers.forEach((t, i) => {
    const y = 24 + i * rowHeight;
    root.append(svg("text", { x: left - 10, y: y + 4, "text-anchor": "end" }, place(t)));
    root.append(svg("line", { x1: x(t.mean_predicted), x2: x(t.target_rate), y1: y, y2: y, class: "gap" }));
    root.append(describe(svg("circle", { cx: x(t.mean_predicted), cy: y, r: 6, class: "predicted" }),
      `${place(t)}: predicted ${percent(t.mean_predicted, 1)}`));
    root.append(describe(svg("circle", { cx: x(t.target_rate), cy: y, r: 6, class: "observed" }),
      `${place(t)}: real ${percent(t.target_rate, 1)}`));
  });
  root.append(svg("text", { x: (left + width) / 2, y: height - 6, "text-anchor": "middle" }, "Share with the condition"));
  return root;
}

function drawCurves(transfer) {
  const { sizes } = transfer;
  const all = Object.values(transfer.curves).flatMap((c) => sizes.map((n) => c[n].log_loss.mean));
  // A local model fit on a handful of patients can be far off the scale. Cap
  // the axis so the rest stays readable, and say so under the chart.
  const base = transfer.curves.none[sizes[0]].log_loss.mean;
  const floor = Math.min(...all);
  const cap = Math.max(base, floor) + Math.max(base - floor, base * 0.15) * 1.2;
  const high = Math.min(Math.max(...all), cap) * 1.01;
  const low = floor - (high - floor) * 0.08;
  const clipped = all.some((v) => v > high);
  const chart = frame(`Log loss by number of local patients, ${place(transfer)}`, {
    width: wideChart(), height: 320, x: [0, sizes.length - 1], y: [low, high], xTicks: sizes.map((_, i) => i),
    yTicks: niceTicks(low, high), xLabel: "Local labelled patients", yLabel: "Log loss (lower is better)",
    xText: (i) => count(sizes[i]), yText: (v) => v.toFixed(high - low < 0.05 ? 3 : 2),
  });
  for (const [method, label] of Object.entries(METHODS)) {
    const values = sizes.map((n) => transfer.curves[method][n].log_loss.mean);
    const points = values.map((v, i) => [chart.px(i), chart.py(Math.min(v, high))]);
    chart.root.append(svg("path", { d: path(points), class: `line line--${method}` }));
    values.forEach((v, i) => {
      const at = transfer.curves[method][sizes[i]].log_loss;
      chart.root.append(describe(svg("circle", {
        cx: points[i][0], cy: points[i][1], r: 4.5, class: `dot dot--${method}`,
      }), `${label}, ${count(sizes[i])} patients: ${v.toFixed(3)} (90% of draws ${at.p5.toFixed(3)} to ${at.p95.toFixed(3)})`));
    });
  }
  return { root: chart.root, clipped };
}

function renderTransfer() {
  const transfer = data.transfers.find((t) => t.name === $("#lab-target").value);
  const { root, clipped } = drawCurves(transfer);
  const repair = transfer.repair_size.intercept;
  const note = repair === null
    ? `At no tested size did the one-number shift help in 90% of draws.`
    : `From ${count(repair)} local patients, the one-number shift helped in at least 90% of draws.`;
  $("#lab-curves").replaceChildren(figure(
    `Repairing the model at ${place(transfer)}`,
    note + (clipped ? " Points above the axis are drawn at its top edge; the table has their values." : ""),
    root,
    swatches(Object.entries(METHODS).map(([key, label]) => [`line line--${key}`, label])),
  ));
  $("#lab-curve-table").replaceChildren(table(
    `Mean log loss at ${place(transfer)}`, ["Method", ...transfer.sizes.map((n) => `${count(n)} patients`)],
    Object.entries(METHODS).map(([key, label]) => [label,
      ...transfer.sizes.map((n) => transfer.curves[key][n].log_loss.mean.toFixed(4))]),
  ));
}

function renderStudy() {
  const heart = data.transfers.filter((t) => t.family === "heart");
  const dengue = data.transfers.filter((t) => t.family === "dengue");
  const key = () => swatchesForRates();
  $("#lab-rates").replaceChildren(
    figure("Heart disease", "Each hospital was left out of training, then predicted.", drawRates(heart), key()),
    figure("Dengue severity", "Each region, and a later year, was left out of training.", drawRates(dengue), key()),
  );
  $("#lab-transfer-table").replaceChildren(table("Ten transfers",
    ["Target", "Rows", "AUC at home", "AUC there", "Predicted", "Real", "Calibration slope", "Cases caught at the 90% cutoff"],
    data.transfers.map((t) => [
      `${capital(t.family)}, ${place(t)}`, count(t.target_rows), t.source_auc.toFixed(3), t.target_auc.toFixed(3),
      percent(t.mean_predicted, 1), percent(t.target_rate, 1), t.calibration_slope.toFixed(2), percent(t.sensitivity),
    ])));
  $("#lab-verdicts").replaceChildren(...Object.entries(HYPOTHESES).map(([id, text]) => {
    const v = data.verdicts[id];
    const item = html("li");
    const detail = id === "H1"
      ? `Rates broke in ${v.rates_broke.length} of ${v.of} targets, ranking in ${v.ranking_broke.length}.`
      : `True in ${v.count} of ${v.of} targets. The bar set in advance was ${v.needed}.`;
    item.append(html("strong", v.supported ? "verdict is-yes" : "verdict is-no", v.supported ? "Supported" : "Not supported"),
      html("span", "", `${id}. ${text}. `), html("small", "", detail));
    return item;
  }));
  const { missingness, site } = data.audits;
  $("#lab-site-audit").replaceChildren(table("Site shortcut audit",
    ["Data", "Pooled AUC", "Site's rate alone", "Within-site AUC"],
    Object.entries(site).map(([name, a]) => [MODEL_NAMES[name], a.pooled_auc.toFixed(3),
      a.site_rate_alone_auc.toFixed(3), a.within_site_auc.toFixed(3)])));
  $("#lab-missing-audit").replaceChildren(table("Missing value audit",
    ["Data", "Values missing", "Predicts the label (AUC)", "Predicts the hospital (AUC)"],
    Object.entries(missingness).map(([name, a]) => [
      name === "heart_raw_13_inputs" ? "Heart, all 13 raw inputs" : MODEL_NAMES[name],
      percent(a.missing_share, 1), a.predicts_label_auc.toFixed(3),
      a.predicts_site_auc === undefined ? "not applicable" : a.predicts_site_auc.toFixed(3),
    ])));
  $("#lab-target").replaceChildren(...data.transfers.map((t) => {
    const option = html("option", "", `${capital(t.family)}: ${place(t)}`);
    option.value = t.name;
    return option;
  }));
  renderTransfer();
}

const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// State study

const STATE_TESTS = {
  S1: "Draws with under 10 local cases hurt more often than draws with 20 or more",
  S2: "Repair sizes agree better in local cases than in patients",
  S3: "The input-shift score warns where the label-free repair fails",
  S4: "A warning cut-off fitted on 2023 works on 2024 and 2025",
  R1: "A single shift with 500 local patients helps (replication)",
  R2: "The label-free repair makes things worse (replication)",
};
const ESTIMATORS = [["em", "EM's estimate", "dot--em"], ["bbse", "BBSE estimate", "dot--refit"]];
const RATIO_RANGE = [0.05, 20];
const signed = (value) => `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
const ratioText = (value) => `${value < 1 ? value.toFixed(2) : value.toFixed(1)} times`;

function helped(targets, key) {
  return targets.filter((t) => t.change[key] < 0).length;
}

function stateTiles(states) {
  const { targets } = states;
  const of = targets.length;
  return [
    ["Label-free repair (EM)", `${of - helped(targets, "em")} of ${of}`, "made predictions worse"],
    ["EM's correction, given the true rate", `${helped(targets, "true_prior")} of ${of}`, "made them better"],
    ["BBSE, another label-free estimate", `${helped(targets, "bbse")} of ${of}`, "made them better"],
    ["One-number shift, labelled patients", `${helped(targets, "intercept")} of ${of}`, "made them better"],
  ].map(([label, value, note]) => {
    const tile = html("div", "lab-tile");
    tile.append(html("span", "lab-tile-label", label), html("strong", "", value), html("small", "", note));
    return tile;
  });
}

function dotKey(entries) {
  const list = html("ul", "legend");
  for (const [, label, cls] of entries) {
    const item = html("li");
    const mark = svg("svg", { viewBox: "0 0 14 14", "aria-hidden": "true", class: "key key--dot" });
    mark.append(svg("circle", { cx: 7, cy: 7, r: 5, class: `dot ${cls}` }));
    item.append(mark, html("span", "", label));
    list.append(item);
  }
  return list;
}

function drawRateErrors(targets) {
  const clampRatio = (v) => Math.min(RATIO_RANGE[1], Math.max(RATIO_RANGE[0], v));
  const changes = targets.flatMap((t) => ESTIMATORS.map(([key]) => t.change[key]));
  const low = Math.min(0, ...changes);
  const high = Math.max(...changes);
  const pad = (high - low) * 0.06;
  const chart = frame("How far each estimate of the local rate was off, against the change in log loss", {
    width: wideChart(), height: 340, x: RATIO_RANGE.map(Math.log10), y: [low - pad, high + pad],
    xTicks: [0.1, 0.5, 1, 2, 5, 10].map(Math.log10), yTicks: niceTicks(low - pad, high + pad),
    xLabel: "Estimated rate divided by the true rate (log scale)", yLabel: "Change in log loss",
    xText: (v) => `${Number((10 ** v).toPrecision(2))}x`, yText: (v) => v.toFixed(2),
  });
  const { root, px, py } = chart;
  root.append(svg("line", { x1: px(0), x2: px(0), y1: py(high + pad), y2: py(low - pad), class: "chance" }));
  root.append(svg("line", { x1: px(RATIO_RANGE.map(Math.log10)[0]), x2: px(RATIO_RANGE.map(Math.log10)[1]),
    y1: py(0), y2: py(0), class: "chance" }));
  for (const [key, label, cls] of ESTIMATORS) {
    for (const t of targets) {
      const ratio = t[`${key}_rate_over_true`];
      const node = svg("circle", { cx: px(Math.log10(clampRatio(ratio))), cy: py(t.change[key]), r: 5,
        class: `dot ${cls} dot--ringed` });
      root.append(describe(node, `${t.name}, ${label}: ${ratioText(ratio)} the true rate, log loss ${signed(t.change[key])}`));
    }
  }
  return root;
}

function renderStates() {
  const states = data.states;
  $("#lab-state-tiles").replaceChildren(...stateTiles(states));
  $("#lab-state-scatter").replaceChildren(figure(
    "A wrong guess of the rate is what hurts",
    "Each dot is one state-year. Dots right of the dashed line guessed too high, dots above the other line did harm. "
      + "Ratios beyond the axis are drawn at its edge.",
    drawRateErrors(states.targets),
    dotKey(ESTIMATORS),
  ));
  $("#lab-state-verdicts").replaceChildren(...Object.entries(STATE_TESTS).map(([id, text]) => {
    const v = states.verdicts[id];
    let detail;
    if (id === "S2") detail = `Ratio ${v.cv_ratio.toFixed(2)}, 95% interval ${v.ci[0].toFixed(2)} to ${v.ci[1].toFixed(2)}. It needed to stay below 1.`;
    else if (id === "S3") detail = `Spearman ${v.spearman.toFixed(2)}, 95% interval ${v.ci[0].toFixed(2)} to ${v.ci[1].toFixed(2)}. It needed to stay above 0.`;
    else if (id === "S4") detail = `The fitted cut-off was never to use EM, so it could not help in any of ${v.of} targets.`;
    else detail = `True in ${v.count} of ${v.of} targets. The bar set in advance was ${percent(v.needed_share)}.`;
    if (id === "S1") detail += " Flawed as written: it pools sample sizes, so it mostly compares small samples with large.";
    const item = html("li");
    item.append(html("strong", v.supported ? "verdict is-yes" : "verdict is-no", v.supported ? "Supported" : "Not supported"),
      html("span", "", `${id}. ${text}. `), html("small", "", detail));
    return item;
  }));
  $("#lab-state-table").replaceChildren(table("Every state-year",
    ["Target", "Rows", "AUC at home", "AUC there", "Real over predicted", "Local patients", "EM", "True rate", "BBSE", "Labelled shift"],
    states.targets.map((t) => [
      t.name, count(t.rows), t.source_auc.toFixed(3), t.target_auc.toFixed(3), t.oe.toFixed(2), count(t.largest_n),
      signed(t.change.em), signed(t.change.true_prior), signed(t.change.bbse), signed(t.change.intercept),
    ])));
}

function swatchesForRates() {
  const list = html("ul", "legend");
  for (const [cls, label] of [["predicted", "Average prediction"], ["observed", "Real rate"]]) {
    const item = html("li");
    const mark = svg("svg", { viewBox: "0 0 14 14", "aria-hidden": "true", class: "key key--dot" });
    mark.append(svg("circle", { cx: 7, cy: 7, r: 5, class: cls }));
    item.append(mark, html("span", "", label));
    list.append(item);
  }
  return list;
}

export async function loadLab() {
  if (data) return;
  const status = $("#lab-status");
  try {
    const response = await fetch("/static/lab-data.json");
    if (!response.ok) throw new Error(`status ${response.status}`);
    data = await response.json();
  } catch (failure) {
    status.textContent = `The lab data did not load (${failure.message}). Reload the page to try again.`;
    return;
  }
  $("#lab-model").replaceChildren(...Object.keys(data.models).map((name) => {
    const option = html("option", "", MODEL_NAMES[name]);
    option.value = name;
    return option;
  }));
  $("#lab-model").addEventListener("change", renderModel);
  $("#lab-cutoff").addEventListener("input", renderPoint);
  $("#lab-prevalence").addEventListener("input", renderPoint);
  $("#lab-target").addEventListener("change", renderTransfer);
  $("#lab-reset").addEventListener("click", renderModel);
  renderModel();
  renderStudy();
  renderStates();
  status.hidden = true;
  $("#lab-body").hidden = false;
}
