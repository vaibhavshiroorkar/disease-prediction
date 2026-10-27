const $ = (selector) => document.querySelector(selector);
const fields = $("#input-fields");
const form = $("#prediction-form");
const results = $("#results");
const submit = $("#submit-button");
const error = $("#form-error");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const state = {
  model: "symptom", ready: false, symptoms: [], selected: new Set(),
  details: null, loading: false, file: null, preview: null, load: null, run: null, revision: 0,
};

const models = {
  symptom: {
    title: "Rank conditions from symptoms",
    description: "Pick the symptoms to include. The model ranks the conditions in its training data that share them.",
    scope: "Scores rank conditions against each other. They are not the chance of having one.",
  },
  heart: {
    title: "Coronary artery disease",
    description: "Enter clinical measurements. The model estimates how likely an angiogram would show a major artery more than half blocked.",
    scope: "Accuracy and calibration differ between the four hospitals it learned from.",
  },
  diabetes: {
    title: "Diabetes within five years",
    description: "Enter glucose and other measurements. The model estimates the chance of a diabetes diagnosis within five years.",
    scope: "It only saw Pima women aged 21 and over. Nobody has checked it on any other population.",
  },
  liver: {
    title: "Recorded liver patient",
    description: "Enter age and blood test results. The model estimates whether the record belongs to a liver patient.",
    scope: "It learned from records in Andhra Pradesh. Its cutoff flags many people who are not liver patients.",
  },
  dengue: {
    title: "Severity of confirmed dengue",
    description: "Describe a lab-confirmed case. The model estimates the chance it becomes dengue with warning signs or severe dengue.",
    scope: "Use it only for lab-confirmed dengue. It learned from Brazil's 2023 records and scored worse on 2025 records.",
  },
  malaria: {
    title: "Parasite in a blood cell",
    description: "Upload one cropped red blood cell from a Giemsa-stained thin smear. The model estimates whether it holds a malaria parasite.",
    scope: "It learned from single cells from one hospital. It cannot read whole slides, rapid test strips or other images.",
  },
};

const examples = {
  symptom: ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
  heart: { age: 61, sex: 1, cp: 4, trestbps: 140, chol: 240, thalach: 115, exang: 1, oldpeak: 2.5 },
  diabetes: { pregnancies: 2, age: 45, glucose: 140, blood_pressure: 80, bmi: 30, pedigree: 0.5 },
  liver: { age: 45, sex: 1, total_bilirubin: 1.2, direct_bilirubin: 0.4, alt: 35, ast: 40, albumin: 4 },
  dengue: { age: 34, sex: 0, pregnant: 0, days_since_onset: 3, fever: 1, headache: 1, myalgia: 1 },
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function showError(message = "") {
  error.textContent = message;
  error.hidden = !message;
}

function busy(value) {
  submit.disabled = value || !state.ready;
  submit.textContent = value ? "Running..." : "Run prediction";
  results.setAttribute("aria-busy", String(value));
}

// Builds the 0 to 100% scale with the cutoff line. The marker is placed after
// the first paint so that it slides from zero to the score.
function ruler(cut, score) {
  const node = element("div", "ruler");
  node.style.setProperty("--cut", String(cut));
  const ends = element("div", "ruler-ends");
  ends.append(element("span", "", "0%"), element("span", "", "100%"));
  const track = element("div", "ruler-track");
  track.append(element("span", "ruler-cut"));
  if (score === undefined) {
    node.classList.add("is-empty");
    node.setAttribute("aria-hidden", "true");
  } else {
    track.append(element("span", "ruler-marker"));
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", `Score ${pct(score)} on a scale from 0 to 100%, cutoff at ${pct(cut)}`);
    requestAnimationFrame(() => requestAnimationFrame(() => node.style.setProperty("--pos", String(score))));
  }
  const label = element("span", "ruler-cut-label", score === undefined ? "Cutoff chosen in training" : `Cutoff ${pct(cut)}`);
  node.append(ends, track, label);
  return node;
}

function emptyResult() {
  const box = element("div", "empty-result");
  if (state.model === "symptom") {
    const ranks = element("div", "empty-ranks");
    ranks.setAttribute("aria-hidden", "true");
    ranks.append(element("span"), element("span"), element("span"));
    box.append(ranks, element("h2", "", "Your results will appear here"),
      element("p", "", "The model lists the three conditions in its training data that best match your symptoms, with the symptoms they share."));
  } else {
    box.append(ruler(0.5), element("h2", "", "Your results will appear here"),
      element("p", "", "The model places its score on this scale. Anything past the cutoff line counts as flagged. The cutoff was chosen during training."));
  }
  return box;
}

function invalidate() {
  state.revision += 1;
  state.run?.abort();
  state.run = null;
  results.replaceChildren(emptyResult());
  showError();
  busy(false);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options, cache: "no-store",
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(30000)])
      : AbortSignal.timeout(30000),
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`The service sent an unreadable response (status ${response.status}). Try again shortly.`);
  }
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map((item) => `${item.loc.at(-1)}: ${item.msg}`).join(". ")
        : detail?.message || "The request could not be completed. Check your inputs and try again.";
    throw new Error(message);
  }
  return body;
}

function explainFailure(failure) {
  if (failure.name === "TimeoutError") return "The request took too long. Try again shortly.";
  if (failure instanceof TypeError) return "Could not reach the service. Check your connection and try again.";
  return failure.message;
}

// Symptom picker

function selectedSymptoms() {
  const selected = $("#selected-symptoms");
  selected.replaceChildren();
  for (const symptom of state.symptoms.filter((item) => state.selected.has(item.key))) {
    const chip = element("button", "symptom-chip", symptom.label);
    chip.type = "button";
    chip.setAttribute("aria-label", `Remove ${symptom.label}`);
    chip.insertAdjacentHTML("beforeend",
      '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8"/></svg>');
    chip.addEventListener("click", () => {
      state.selected.delete(symptom.key);
      invalidate();
      selectedSymptoms();
      symptomOptions();
      $("#symptom-search").focus();
    });
    selected.append(chip);
  }
  if (!state.selected.size) selected.append(element("span", "selected-empty", "No symptoms chosen yet."));
  $("#input-count").textContent = `${state.selected.size} selected`;
}

function symptomOptions() {
  const options = $("#symptom-options");
  const query = $("#symptom-search").value.trim().toLowerCase();
  const matches = state.symptoms.filter((item) => item.label.toLowerCase().includes(query));
  options.replaceChildren();
  for (const symptom of matches) {
    const label = element("label", "symptom-option");
    const checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.value = symptom.key;
    checkbox.checked = state.selected.has(symptom.key);
    checkbox.addEventListener("change", () => {
      invalidate();
      if (checkbox.checked && state.selected.size >= 50) {
        checkbox.checked = false;
        showError("Choose up to 50 symptoms per prediction.");
        return;
      }
      if (checkbox.checked) state.selected.add(symptom.key);
      else state.selected.delete(symptom.key);
      selectedSymptoms();
    });
    label.append(checkbox, document.createTextNode(symptom.label));
    options.append(label);
  }
  if (!matches.length) options.append(element("p", "no-matches", "No symptom matches that search. Try a shorter word."));
}

function renderSymptoms() {
  const label = element("label", "", `Search ${state.symptoms.length} symptoms`);
  label.htmlFor = "symptom-search";
  const search = element("input");
  search.type = "search";
  search.id = "symptom-search";
  search.placeholder = "Headache, fatigue, itching...";
  search.autocomplete = "off";
  search.addEventListener("input", symptomOptions);
  const selected = element("div", "selected-symptoms");
  selected.id = "selected-symptoms";
  selected.setAttribute("aria-label", "Chosen symptoms");
  const options = element("div", "symptom-options");
  options.id = "symptom-options";
  options.setAttribute("role", "group");
  options.setAttribute("aria-label", "Available symptoms");
  fields.replaceChildren(label, search, selected, options);
  selectedSymptoms();
  symptomOptions();
}

// Clinical forms, built from the input specs the model publishes

function clinicalField(spec) {
  const wrapper = element("div", "field");
  const label = element("label", "", spec.unit ? `${spec.label} (${spec.unit})` : spec.label);
  label.htmlFor = `field-${spec.name}`;
  let control;
  if (spec.choices) {
    control = element("select");
    const empty = element("option", "", spec.required ? "Choose an option" : "Not provided");
    empty.value = "";
    control.append(empty);
    for (const [value, description] of Object.entries(spec.choices)) {
      const option = element("option", "", capitalize(description));
      option.value = value;
      control.append(option);
    }
  } else {
    control = element("input");
    control.type = "number";
    control.step = "any";
    control.inputMode = "decimal";
    if (spec.minimum !== undefined) control.min = spec.minimum;
    if (spec.maximum !== undefined) control.max = spec.maximum;
    control.placeholder = spec.minimum !== undefined && spec.maximum !== undefined
      ? `${spec.minimum} to ${spec.maximum}` : spec.required ? "Required" : "Optional";
  }
  control.id = `field-${spec.name}`;
  control.name = spec.name;
  control.required = Boolean(spec.required);
  wrapper.append(label, control);
  if (spec.description && !(spec.choices && /^\d/.test(spec.description))) {
    const help = element("small", "", spec.description);
    help.id = `help-${spec.name}`;
    control.setAttribute("aria-describedby", help.id);
    wrapper.append(help);
  }
  return wrapper;
}

function renderClinical() {
  const required = element("div", "field-grid");
  const optional = element("div", "field-grid");
  for (const spec of state.details.inputs) {
    (spec.required ? required : optional).append(clinicalField(spec));
  }
  const optionalCount = state.details.inputs.filter((item) => !item.required).length;
  const children = [required];
  if (optionalCount) {
    const details = element("details", "optional-inputs");
    details.append(element("summary", "", `More measurements (${optionalCount} optional)`), optional);
    children.push(details);
  }
  fields.replaceChildren(...children);
  const requiredCount = state.details.inputs.length - optionalCount;
  $("#input-count").textContent = `${requiredCount} required`;
}

// Malaria image upload

function releaseImage() {
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.preview = null;
  state.file = null;
}

function renderUpload() {
  const zone = element("div", "upload-zone");
  const text = element("div");
  const label = element("label", "", "Blood cell image");
  label.htmlFor = "cell-image";
  const input = element("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg";
  input.id = "cell-image";
  input.setAttribute("aria-describedby", "cell-image-help");
  const help = element("p", "", "PNG or JPEG, up to 2 MB. Crop it to one cell. You can also drop the file here.");
  help.id = "cell-image-help";
  text.append(label, input, help);
  const preview = element("img", "image-preview");
  preview.alt = "The blood cell image you chose";
  preview.hidden = true;
  input.addEventListener("change", () => {
    invalidate();
    releaseImage();
    preview.hidden = true;
    preview.removeAttribute("src");
    const file = input.files[0];
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type) || file.size > 2_000_000) {
      input.value = "";
      showError("Choose a PNG or JPEG image smaller than 2 MB.");
      return;
    }
    state.file = file;
    state.preview = URL.createObjectURL(file);
    preview.src = state.preview;
    preview.hidden = false;
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    if (!event.dataTransfer.files.length) return;
    input.files = event.dataTransfer.files;
    input.dispatchEvent(new Event("change"));
  });
  zone.append(text, preview);
  fields.replaceChildren(zone);
  $("#input-count").textContent = "One cell per image";
}

async function chooseModel(model) {
  state.load?.abort();
  const controller = new AbortController();
  state.load = controller;
  state.model = model;
  state.ready = false;
  state.loading = true;
  releaseImage();
  invalidate();
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.model === model));
  });
  const { title, description, scope } = models[model];
  $("#model-title").textContent = title;
  $("#model-description").textContent = description;
  $("#scope-note").textContent = scope;
  $("#input-count").textContent = "";
  $("#example-button").hidden = !examples[model];
  $("#example-button").disabled = true;
  fields.setAttribute("aria-busy", "true");
  fields.replaceChildren(element("p", "loading-copy", "Loading input options..."));
  try {
    state.details = await request(`/models/${model}`, { signal: controller.signal });
    if (model === "symptom") {
      if (!state.symptoms.length) {
        const catalogue = await request("/symptoms", { signal: controller.signal });
        state.symptoms = catalogue.symptoms;
      }
      renderSymptoms();
    } else if (model === "malaria") renderUpload();
    else renderClinical();
    state.ready = true;
    $("#example-button").disabled = false;
  } catch (failure) {
    if (controller.signal.aborted) return;
    showError(explainFailure(failure));
    const retry = element("button", "secondary-button", "Try loading again");
    retry.type = "button";
    retry.addEventListener("click", () => chooseModel(model));
    fields.replaceChildren(element("p", "loading-copy", "The input options did not load."), retry);
  } finally {
    if (!controller.signal.aborted) {
      state.loading = false;
      fields.setAttribute("aria-busy", "false");
      busy(false);
    }
  }
}

// Results

function renderSymptomsResult(data) {
  results.append(element("h2", "", "Prediction results"));
  results.append(element("p", "result-context", "Closest matches in the training data, not a diagnosis."));
  const top = data.predictions[0]?.probability || 1;
  const list = element("ol", "condition-list");
  for (const item of data.predictions) {
    const row = element("li", "condition");
    const score = element("div", "score-row");
    score.append(element("span", "", "Model score"), element("strong", "", pct(item.probability)));
    const bar = element("div", "score-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.append(element("span"));
    requestAnimationFrame(() => requestAnimationFrame(() => bar.style.setProperty("--size", String(item.probability / top))));
    row.append(element("h3", "", item.disease), score, bar);
    if (item.matched_symptoms.length) {
      const shared = item.matched_symptoms.map((key) => key.replaceAll("_", " ")).join(", ");
      row.append(element("p", "", `Shared symptoms: ${shared}.`));
    }
    list.append(row);
  }
  results.append(list);
  const accuracy = data.model_metrics?.cross_validation?.accuracy;
  if (accuracy !== undefined) {
    results.append(element("p", "model-footnote", `Cross-validated accuracy on clean training patterns: ${pct(accuracy, 0)}. `
      + "Real symptoms are messier, so expect the model to do worse on them."));
  }
}

function describeValue(spec, value) {
  if (value === null || value === undefined) return "not given";
  if (spec?.choices) return spec.choices[String(value)] ?? String(value);
  return `${Number(value.toFixed(2))}${spec?.unit ? ` ${spec.unit}` : ""}`;
}

function factorsSection(data) {
  const section = element("section", "result-section");
  section.append(element("h3", "", "Inputs that changed the score"));
  const specs = new Map(state.details.inputs.map((item) => [item.name, item]));
  const largest = Math.max(...data.factors.map((factor) => Math.abs(factor.effect)), 0.001);
  for (const factor of data.factors) {
    const spec = specs.get(factor.feature);
    const row = element("div", "factor");
    const head = element("div", "factor-head");
    const name = element("span", "", factor.label);
    name.append(element("span", "factor-value",
      `${describeValue(spec, factor.value)}, typical ${describeValue(spec, factor.typical_value)}`));
    const change = `${factor.effect >= 0 ? "+" : "-"}${Math.abs(factor.effect * 100).toFixed(1)} points`;
    head.append(name, element("strong", "", change));
    const bar = element("div", factor.effect < 0 ? "factor-bar is-negative" : "factor-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--size", String(Math.abs(factor.effect) / largest));
    bar.append(element("span"));
    row.append(head, bar);
    section.append(row);
  }
  section.append(element("p", "", data.factors.length
    ? "Each figure is the change in percentage points when that input is swapped for a typical training value. Pink raised the score, blue lowered it."
    : "None of the values you gave moved the rounded score away from what typical training values give."));
  return section;
}

function operatingPoint(data, isImage) {
  const point = data.operating_point;
  if (!point) return null;
  const [cases, others, flagged] = isImage
    ? ["infected cells", "uninfected cells", "cells it flagged"]
    : ["people who had the condition", "people who did not", "people it flagged"];
  let text = `On its test data, this cutoff caught ${Math.round(point.sensitivity * 100)} in 100 ${cases} `
    + `and cleared ${Math.round(point.specificity * 100)} in 100 ${others}.`;
  if (point.ppv !== undefined) {
    text += ` Of the ${flagged}, ${Math.round(point.ppv * 100)} in 100 were true cases.`;
  }
  return element("p", "operating-point", text);
}

function renderRiskResult(data) {
  const isImage = state.model === "malaria";
  results.append(element("h2", "", "Prediction results"));
  results.append(element("p", "result-context", isImage ? "Single-cell classification" : capitalize(data.condition || models[state.model].title)));
  if (isImage && state.preview) {
    const image = element("div", "result-image");
    const thumbnail = element("img");
    thumbnail.src = state.preview;
    thumbnail.alt = "";
    image.append(thumbnail, element("span", "", data.condition ? capitalize(data.condition) : "Your uploaded cell"));
    results.append(image);
  }
  const flaggedClass = data.flagged ? " is-flagged" : "";
  results.append(element("p", "estimate-label", "Model estimate"));
  results.append(element("p", `estimate${flaggedClass}`, pct(data.probability)));
  results.append(element("span", `outcome${flaggedClass}`, data.flagged ? "Above the model threshold" : "Below the model threshold"));
  const scale = ruler(data.threshold, data.probability);
  if (data.flagged) scale.classList.add("is-flagged");
  results.append(scale);
  const point = operatingPoint(data, isImage);
  if (point) results.append(point);
  if (isImage) {
    results.append(element("p", "model-footnote",
      "The score is for this dataset's parasitized-cell label. It says nothing about whether a person has malaria."));
  } else {
    results.append(factorsSection(data));
    if (data.imputed_fields.length) {
      const labels = new Map(state.details.inputs.map((item) => [item.name, item.label]));
      const imputed = element("section", "result-section");
      imputed.append(element("h3", "", `Measurements filled in (${data.imputed_fields.length})`));
      imputed.append(element("p", "", `${data.imputed_fields.map((name) => labels.get(name) || name).join(", ")}. `
        + "You left these blank, so the model used the training median."));
      results.append(imputed);
    }
  }
  const metrics = data.model_metrics;
  if (metrics?.roc_auc !== undefined) {
    let text = `Test ROC AUC ${metrics.roc_auc.toFixed(2)}`;
    if (metrics.roc_auc_ci95) text += ` (95% interval ${metrics.roc_auc_ci95[0].toFixed(2)} to ${metrics.roc_auc_ci95[1].toFixed(2)})`;
    if (metrics.test_rows) text += ` on ${metrics.test_rows.toLocaleString("en")} held-out ${isImage ? "cells" : "records"}`;
    results.append(element("p", "model-footnote", `${text}.`));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.ready) return;
  showError();
  let payload;
  const headers = {};
  if (state.model === "symptom") {
    if (!state.selected.size) return showError("Choose at least one symptom before running a prediction.");
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify({ symptoms: [...state.selected], top_k: 3 });
  } else if (state.model === "malaria") {
    if (!state.file) return showError("Choose a blood cell image before running a prediction.");
    headers["Content-Type"] = state.file.type;
    payload = state.file;
  } else {
    if (!form.checkValidity()) {
      const optional = $(".optional-inputs");
      if (optional && optional.querySelector(":invalid")) optional.open = true;
      form.reportValidity();
      return showError("Fill in the required fields and keep each value inside its range.");
    }
    headers["Content-Type"] = "application/json";
    const values = {};
    for (const spec of state.details.inputs) {
      const value = form.elements.namedItem(spec.name).value;
      if (value !== "") values[spec.name] = Number(value);
    }
    payload = JSON.stringify(values);
  }
  state.run?.abort();
  const controller = new AbortController();
  state.run = controller;
  const revision = state.revision;
  busy(true);
  try {
    const endpoint = state.model === "symptom" ? "symptoms" : state.model;
    const data = await request(`/predict/${endpoint}`, { method: "POST", headers, body: payload, signal: controller.signal });
    if (revision !== state.revision) return;
    results.replaceChildren();
    if (state.model === "symptom") renderSymptomsResult(data);
    else renderRiskResult(data);
    results.append(element("p", "disclaimer", data.disclaimer));
    // On narrow screens the result sits below the form, out of view.
    const top = results.getBoundingClientRect().top;
    if (top > window.innerHeight * 0.6) {
      results.scrollIntoView({ block: "start", behavior: reducedMotion.matches ? "auto" : "smooth" });
    }
  } catch (failure) {
    if (!controller.signal.aborted && revision === state.revision) showError(explainFailure(failure));
  } finally {
    if (revision === state.revision) busy(false);
  }
});

form.addEventListener("input", (event) => {
  if (event.target.matches("input[type=number], select")) invalidate();
});

$("#example-button").addEventListener("click", () => {
  if (!state.ready || !examples[state.model]) return;
  invalidate();
  if (state.model === "symptom") {
    state.selected = new Set(examples.symptom);
    $("#symptom-search").value = "";
    selectedSymptoms();
    symptomOptions();
    return;
  }
  form.reset();
  for (const [key, value] of Object.entries(examples[state.model])) form.elements.namedItem(key).value = value;
  const optional = $(".optional-inputs");
  if (optional) optional.open = true;
});

$("#clear-button").addEventListener("click", () => {
  invalidate();
  state.selected.clear();
  releaseImage();
  if (!state.ready) return;
  form.reset();
  if (state.model === "symptom") { selectedSymptoms(); symptomOptions(); }
  if (state.model === "malaria") renderUpload();
});

document.querySelectorAll("[data-model]").forEach((button) => {
  button.addEventListener("click", () => {
    const model = button.dataset.model;
    if (location.hash !== `#/checks/${model}`) history.pushState(null, "", `#/checks/${model}`);
    openModel(model);
  });
});

// Skips reloading when the model is already showing or still loading.
function openModel(model) {
  if (model === state.model && (state.ready || state.loading)) return;
  chooseModel(model);
}

// Service status and privacy wording

async function loadStatus() {
  const status = $("#service-status");
  try {
    const health = await request("/health");
    const entries = Object.values(health.models);
    const loaded = entries.filter((model) => model.loaded).length;
    status.textContent = `${loaded} of ${entries.length} models ready`;
    status.classList.add(loaded === entries.length ? "is-ready" : "is-degraded");
    const logging = health.database.logging_enabled;
    $("#submit-note").textContent = logging
      ? "This deployment records prediction inputs for review. Do not enter anything that identifies a person."
      : "Inputs go to the server to run the model. Prediction logging is off.";
    $("#privacy-note").textContent = logging
      ? "This deployment stores submitted inputs and results so an administrator can review them. Do not submit identifying information. Uploaded images are stored only as their dimensions and a hash."
      : "Prediction logging is off on this deployment. Your inputs go to the server to run the model and are not kept. This site does not save them in your browser either.";
  } catch {
    status.textContent = "Service status unavailable";
    status.classList.add("is-degraded");
    $("#submit-note").textContent = "Inputs go to the server to run the model. Logging status is unknown.";
    $("#privacy-note").textContent = "Could not check whether this deployment logs predictions. Do not submit identifying information.";
  }
}

// Models page, built from what each artifact reports about itself

const modelPage = {
  symptom: { name: "Symptom checker", unit: "rows" },
  heart: { name: "Heart disease", unit: "patients" },
  diabetes: { name: "Diabetes", unit: "women" },
  liver: { name: "Liver disease", unit: "patients" },
  dengue: { name: "Dengue severity", unit: "cases" },
  malaria: { name: "Malaria cell", unit: "cells" },
};
let modelsLoaded = null;

function fact(list, term, ...content) {
  const dd = element("dd");
  dd.append(...content);
  list.append(element("dt", "", term), dd);
}

function count(value) {
  return value.toLocaleString("en");
}

function sourceLinks(urls) {
  const wrap = element("span", "sources");
  const hosts = new Map();
  for (const url of urls) {
    try {
      const host = new URL(url).hostname;
      if (!hosts.has(host)) hosts.set(host, url);
    } catch { /* skip a malformed source */ }
  }
  for (const [host, url] of hosts) {
    const link = element("a", "", host);
    link.href = url;
    link.rel = "noreferrer";
    wrap.append(link);
  }
  return wrap;
}

function modelSection(name, detail) {
  const { unit } = modelPage[name];
  const section = element("section", "model-section");
  section.id = `model-${name}`;
  section.setAttribute("aria-labelledby", `model-${name}-title`);
  const intro = element("div", "model-intro");
  const heading = element("h2", "", modelPage[name].name);
  heading.id = `model-${name}-title`;
  heading.tabIndex = -1;
  const run = element("a", "", "Run this check");
  run.href = `#/checks/${name}`;
  intro.append(heading, run);

  const facts = element("dl", "facts");
  const data = detail.dataset;
  const metrics = detail.metrics;
  if (name === "symptom") {
    fact(facts, "Predicts", `A ranking of ${detail.summary.classes} conditions.`);
    fact(facts, "Data", `${count(data.rows_raw)} rows that reduce to ${count(data.rows_unique)} distinct symptom patterns.`);
    const noisy = metrics.noise_2_dropped_2_added?.accuracy;
    fact(facts, "Test results", `${pct(metrics.cross_validation.accuracy, 0)} cross-validated accuracy on clean patterns`
      + (noisy !== undefined ? `, ${pct(noisy)} when two true symptoms are dropped and two random ones added.` : "."));
    fact(facts, "Source", sourceLinks([data.source]));
  } else {
    const test = metrics.test;
    const ci = metrics.test_ci95?.roc_auc;
    fact(facts, "Predicts", capitalize(detail.summary.condition));
    const patients = data.patients ? ` from ${count(data.patients)} patients` : "";
    fact(facts, "Data", `${count(data.rows)} ${unit}${patients}. ${count(data.rows_train)} trained the model and `
      + `${count(test.rows)} were held out to test it. ${pct(data.prevalence, 0)} had the condition.`);
    const auc = element("span", "", `ROC AUC ${test.roc_auc.toFixed(2)}`);
    if (ci) auc.append(element("span", "interval", ` (95% interval ${ci[0].toFixed(2)} to ${ci[1].toFixed(2)})`));
    fact(facts, "Ranking quality", auc);
    fact(facts, `At its ${pct(detail.summary.threshold)} cutoff`,
      `Catches ${Math.round(test.sensitivity * 100)} in 100 true cases and clears ${Math.round(test.specificity * 100)} in 100 others. `
      + `${Math.round(test.ppv * 100)} in 100 of its flags are true cases.`);
    fact(facts, "Sources", sourceLinks(data.sources || []));
  }
  fact(facts, "Known limit", models[name].scope);
  section.append(intro, facts);
  return section;
}

function loadModels() {
  modelsLoaded ??= (async () => {
    const box = $("#model-sections");
    try {
      const details = await Promise.all(Object.keys(modelPage).map((name) => request(`/models/${name}`)));
      box.replaceChildren(...details.map((detail) => modelSection(detail.name, detail)));
    } catch (failure) {
      modelsLoaded = null;
      const retry = element("button", "secondary-button", "Try loading again");
      retry.type = "button";
      retry.addEventListener("click", () => route());
      box.replaceChildren(element("p", "loading-copy", `Model details did not load. ${explainFailure(failure)}`), retry);
    } finally {
      box.setAttribute("aria-busy", "false");
    }
  })();
  return modelsLoaded;
}

// Pages live in this one document and are addressed by the hash, so the
// server keeps serving a single file.

const titles = {
  home: "Pattern | Disease prediction lab",
  checks: "Checks | Pattern",
  models: "Models | Pattern",
  about: "About | Pattern",
  missing: "Page not found | Pattern",
};
const menuButton = $(".menu-button");
const nav = $("#site-nav");
let firstRoute = true;

function setMenu(open) {
  menuButton.setAttribute("aria-expanded", String(open));
  nav.classList.toggle("is-open", open);
}

async function route() {
  const hash = location.hash;
  const initial = firstRoute;
  firstRoute = false;
  // In-page anchors such as the skip link are not pages.
  if (hash && !hash.startsWith("#/") && !initial) return;
  const [view, detail] = (hash.startsWith("#/") ? hash.slice(2) : "").split("/");
  let page = view || "home";
  if (!(page in titles) || page === "missing") page = "missing";
  if (page === "checks" && detail && !(detail in models)) page = "missing";

  const previous = document.querySelector(".view:not([hidden])")?.dataset.view;
  document.querySelectorAll(".view").forEach((section) => { section.hidden = section.dataset.view !== page; });
  document.querySelectorAll("[data-route]").forEach((link) => {
    if (link.dataset.route === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.title = titles[page];
  setMenu(false);

  if (page === "checks") openModel(detail || state.model);
  let target = document.querySelector(`.view[data-view="${page}"] h1`);
  if (page === "models") {
    await loadModels();
    const section = detail && document.getElementById(`model-${detail}`);
    if (section) target = section.querySelector("h2");
  }
  // A fresh load keeps the browser's own focus; a page change moves it to the new heading.
  if (initial && target.tagName === "H1") return;
  if (previous !== page || target.tagName !== "H1") {
    target.scrollIntoView({ block: "start" });
    target.focus({ preventScroll: true });
  }
}

menuButton.addEventListener("click", () => setMenu(menuButton.getAttribute("aria-expanded") !== "true"));
$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  const main = $("#main");
  main.focus();
  main.scrollIntoView();
});
window.addEventListener("hashchange", route);

route();
loadStatus();
