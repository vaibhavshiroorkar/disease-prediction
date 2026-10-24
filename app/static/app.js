const $ = (selector) => document.querySelector(selector);
const fields = $("#input-fields");
const form = $("#prediction-form");
const results = $("#results");
const emptyResult = results.firstElementChild.cloneNode(true);
const submit = $("#submit-button");
const error = $("#form-error");
const state = {
  model: "symptom", ready: false, symptoms: [], selected: new Set(),
  details: null, file: null, preview: null, load: null, run: null, revision: 0,
};

const descriptions = {
  symptom: ["Symptom explorer", "Start with the symptoms.",
    "Choose the symptoms to include. The model ranks conditions from its training dataset.",
    "Trained on 304 distinct patterns. Scores are not calibrated probabilities of having a condition."],
  heart: ["Heart model", "Add clinical measurements.",
    "Explore the model's estimate for coronary artery disease using measured inputs.",
    "Trained on 918 patients from four hospitals. Performance and calibration vary between hospitals."],
  diabetes: ["Diabetes model", "Add clinical measurements.",
    "Explore the dataset's diabetes outcome using glucose and other measurements.",
    "Trained only on Pima women aged 21 and over. It has not been validated for other populations."],
  liver: ["Liver model", "Add clinical measurements.",
    "Explore how age, blood measurements, and other inputs affect the liver model's score.",
    "Trained on 570 records from Andhra Pradesh. Its chosen threshold produces many false positives."],
  dengue: ["Dengue severity", "Start with a confirmed case.",
    "Estimate the dataset's outcome of dengue with warning signs or severe dengue.",
    "For lab-confirmed dengue only. Trained on Brazil's 2023 records; performance fell on 2025 data."],
  malaria: ["Malaria cell model", "Look at a single blood cell.",
    "Upload one cropped red blood cell from a Giemsa-stained thin smear.",
    "Trained on single cells from one hospital. Whole slides, test strips, and other images are outside its scope."],
};

const examples = {
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

function showError(message = "") {
  error.textContent = message;
  error.hidden = !message;
}

function busy(value) {
  submit.disabled = value || !state.ready;
  submit.textContent = value ? "Running..." : "Run prediction";
  results.setAttribute("aria-busy", String(value));
}

function invalidate() {
  state.revision += 1;
  state.run?.abort();
  state.run = null;
  results.replaceChildren(emptyResult.cloneNode(true));
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
  const body = await response.json();
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

function selectedSymptoms() {
  const selected = $("#selected-symptoms");
  selected.replaceChildren();
  for (const symptom of state.symptoms.filter((item) => state.selected.has(item.key))) {
    const chip = element("button", "symptom-chip", `${symptom.label} x`);
    chip.type = "button";
    chip.setAttribute("aria-label", `Remove ${symptom.label}`);
    chip.addEventListener("click", () => {
      state.selected.delete(symptom.key);
      invalidate();
      selectedSymptoms();
      symptomOptions();
    });
    selected.append(chip);
  }
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
  if (!matches.length) options.append(element("p", "no-matches", "No matching symptoms. Try a different word."));
}

function renderSymptoms() {
  const label = element("label", "", "Find a symptom");
  label.htmlFor = "symptom-search";
  const search = element("input");
  search.type = "search";
  search.id = "symptom-search";
  search.placeholder = "Search headache, fatigue, itching...";
  search.autocomplete = "off";
  search.addEventListener("input", symptomOptions);
  const selected = element("div", "selected-symptoms");
  selected.id = "selected-symptoms";
  selected.setAttribute("aria-label", "Selected symptoms");
  const options = element("div", "symptom-options");
  options.id = "symptom-options";
  options.setAttribute("role", "group");
  options.setAttribute("aria-label", "Available symptoms");
  fields.replaceChildren(label, search, selected, options);
  selectedSymptoms();
  symptomOptions();
}

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
      const option = element("option", "", description.charAt(0).toUpperCase() + description.slice(1));
      option.value = value;
      control.append(option);
    }
  } else {
    control = element("input");
    control.type = "number";
    control.step = "any";
    if (spec.minimum !== undefined) control.min = spec.minimum;
    if (spec.maximum !== undefined) control.max = spec.maximum;
    control.placeholder = spec.required ? "Required" : "Optional";
    control.inputMode = "decimal";
  }
  control.id = `field-${spec.name}`;
  control.name = spec.name;
  control.required = Boolean(spec.required);
  const help = element("small", "", spec.description);
  help.id = `help-${spec.name}`;
  control.setAttribute("aria-describedby", help.id);
  wrapper.append(label, control, help);
  return wrapper;
}

function renderClinical() {
  const required = element("div", "field-grid");
  const optional = element("div", "field-grid");
  for (const spec of state.details.inputs) {
    (spec.required ? required : optional).append(clinicalField(spec));
  }
  const details = element("details", "optional-inputs");
  const optionalCount = state.details.inputs.filter((item) => !item.required).length;
  details.append(element("summary", "", `Additional measurements (${optionalCount} optional)`), optional);
  fields.replaceChildren(required, details);
  $("#input-count").textContent = "Required fields first";
}

function releaseImage() {
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.preview = null;
  state.file = null;
}

function renderUpload() {
  const zone = element("div", "upload-zone");
  const label = element("label", "", "Blood cell image");
  label.htmlFor = "cell-image";
  const input = element("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg";
  input.id = "cell-image";
  const preview = element("img", "image-preview");
  preview.alt = "Preview of the uploaded blood cell";
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
  zone.append(label, input, element("p", "", "PNG or JPEG, up to 2 MB. One cropped cell only."), preview);
  fields.replaceChildren(zone);
  $("#input-count").textContent = "One cell, one image";
}

async function chooseModel(model) {
  state.load?.abort();
  const controller = new AbortController();
  state.load = controller;
  state.model = model;
  state.ready = false;
  releaseImage();
  invalidate();
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.model === model));
  });
  const [kind, title, description, note] = descriptions[model];
  $("#model-kind").textContent = kind;
  $("#input-title").textContent = title;
  $("#model-description").textContent = description;
  $("#population-note").textContent = note;
  $("#example-button").hidden = model === "malaria";
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
    const retry = element("button", "text-button", "Retry loading");
    retry.type = "button";
    retry.addEventListener("click", () => chooseModel(model));
    fields.replaceChildren(element("p", "loading-copy", "The input options could not be loaded."), retry);
  } finally {
    if (!controller.signal.aborted) {
      fields.setAttribute("aria-busy", "false");
      busy(false);
    }
  }
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderSymptomsResult(data) {
  results.append(element("h2", "", "Prediction results"));
  results.append(element("p", "result-context", "Ranked matches in the training dataset, not a diagnosis."));
  const list = element("ol", "condition-list");
  for (const item of data.predictions) {
    const row = element("li", "condition");
    const label = element("div", "score-label");
    label.append(element("span", "", "Model score"), element("span", "", pct(item.probability)));
    const bar = element("progress");
    bar.max = 1;
    bar.value = item.probability;
    bar.setAttribute("aria-label", `${item.disease} model score`);
    row.append(element("h3", "", item.disease), label, bar);
    if (item.matched_symptoms.length) {
      row.append(element("p", "", `Shared symptoms: ${item.matched_symptoms.map((s) => s.replaceAll("_", " ")).join(", ")}`));
    }
    list.append(row);
  }
  results.append(list);
}

function renderRiskResult(data) {
  const isImage = state.model === "malaria";
  results.append(element("h2", "", "Prediction results"));
  results.append(element("p", "result-context", isImage ? "Single-cell classification" : descriptions[state.model][0]));
  results.append(element("p", "panel-label", "Model estimate"));
  results.append(element("p", "estimate", pct(data.probability)));
  results.append(element("span", "outcome-label", data.flagged ? "Above the model threshold" : "Below the model threshold"));
  results.append(element("p", "result-context", `Threshold: ${pct(data.threshold)}. This cutoff was chosen during training.`));
  if (isImage) {
    results.append(element("p", "result-context", "The score estimates the parasitized-cell label in this dataset. It does not diagnose malaria in a person."));
  } else {
    const factors = element("section", "result-detail");
    factors.append(element("h3", "", "Inputs that changed the score"));
    for (const factor of data.factors) {
      const row = element("div", "factor");
      const change = `${factor.effect >= 0 ? "+" : ""}${(factor.effect * 100).toFixed(1)} pp`;
      row.append(element("span", "", factor.label), element("strong", "", change));
      factors.append(row);
    }
    factors.append(element("p", "", data.factors.length
      ? "Percentage-point changes compared with replacing each input with a typical training value."
      : "None of the supplied values changed the rounded score compared with typical training values."));
    results.append(factors);
    if (data.imputed_fields.length) {
      const imputed = element("section", "result-detail");
      const labels = new Map(state.details.inputs.map((item) => [item.name, item.label]));
      imputed.append(element("h3", "", "Measurements filled in"));
      imputed.append(element("p", "", `${data.imputed_fields.map((name) => labels.get(name) || name).join(", ")}. The model used training medians for these inputs.`));
      results.append(imputed);
    }
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
    if (!form.reportValidity()) return showError("Complete the required fields and check the measurement ranges.");
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
  } catch (failure) {
    if (!controller.signal.aborted && revision === state.revision) showError(explainFailure(failure));
  } finally {
    if (revision === state.revision) busy(false);
  }
});

form.addEventListener("input", (event) => {
  if (event.target.matches("input[type=number], select")) invalidate();
});
form.addEventListener("change", (event) => {
  if (event.target.matches("select")) invalidate();
});

$("#example-button").addEventListener("click", () => {
  if (!state.ready) return;
  invalidate();
  if (state.model === "symptom") {
    state.selected = new Set(["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"]);
    $("#symptom-search").value = "";
    selectedSymptoms();
    symptomOptions();
  } else if (examples[state.model]) {
    form.reset();
    for (const [key, value] of Object.entries(examples[state.model])) form.elements.namedItem(key).value = value;
    $(".optional-inputs").open = true;
  }
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
  button.addEventListener("click", () => chooseModel(button.dataset.model));
});

async function loadStatus() {
  try {
    const health = await request("/health");
    const loaded = Object.values(health.models).filter((model) => model.loaded).length;
    $("#service-status").textContent = `${loaded} of 6 models ready`;
    const logging = health.database.logging_enabled;
    $("#submit-note").textContent = logging
      ? "This deployment records prediction inputs for review."
      : "Inputs are sent to the server. Prediction logging is off.";
    $("#privacy-note").textContent = logging
      ? "This deployment stores submitted inputs and model results for administrator review. Do not submit identifying information. Uploaded images are represented by their dimensions and a hash."
      : "Prediction logging is currently off on this deployment. Inputs are sent to the server to run the model. This page does not store them in browser storage.";
  } catch {
    $("#service-status").textContent = "Service status unavailable";
    $("#submit-note").textContent = "Inputs are sent to the server. Logging status is unknown.";
    $("#privacy-note").textContent = "Could not check this deployment's logging settings. Do not submit identifying information.";
  }
}

chooseModel("symptom");
loadStatus();
