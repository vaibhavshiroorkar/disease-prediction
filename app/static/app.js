const $ = (selector) => document.querySelector(selector);
const fields = $("#input-fields");
const form = $("#prediction-form");
const results = $("#results");
const submit = $("#submit-button");
const error = $("#form-error");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const state = {
  model: "symptom", ready: false, symptoms: [], selected: new Set(), active: -1,
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
    description: "Estimates how likely an angiogram would show a major artery more than half blocked.",
    scope: "Accuracy and calibration differ between the four hospitals it learned from.",
  },
  diabetes: {
    title: "Diabetes within five years",
    description: "Estimates the chance of a diabetes diagnosis within five years of the examination.",
    scope: "It only saw Pima women aged 21 and over. Nobody has checked it on any other population.",
  },
  liver: {
    title: "Recorded liver patient",
    description: "Estimates whether a record of age and blood tests belongs to a liver patient.",
    scope: "It learned from records in Andhra Pradesh. Its cutoff flags many people who are not liver patients.",
  },
  dengue: {
    title: "Severity of confirmed dengue",
    description: "Estimates the chance a lab-confirmed case develops warning signs or turns severe.",
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

// Symptoms that need a clinician today, whatever the ranking says.
const urgentSymptoms = new Set([
  "chest_pain", "breathlessness", "coma", "altered_sensorium", "slurred_speech",
  "weakness_of_one_body_side", "stomach_bleeding", "blood_in_sputum", "bloody_stool",
  "acute_liver_failure",
]);

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

// The 0 to 100% scale with the cutoff line. The marker is placed after the
// first paint so that it slides from zero to the score.
function ruler(cut, score) {
  const node = element("div", "ruler");
  node.style.setProperty("--cut", String(cut));
  const track = element("div", "ruler-track");
  track.append(element("span", "ruler-cut"));
  track.append(element("span", "ruler-marker"));
  node.setAttribute("role", "img");
  node.setAttribute("aria-label", `Score ${pct(score)} on a scale from 0 to 100%, cutoff at ${pct(cut)}`);
  requestAnimationFrame(() => requestAnimationFrame(() => node.style.setProperty("--pos", String(score))));
  const ends = element("div", "ruler-ends");
  ends.append(element("span", "", "0%"), element("span", "", `cutoff ${pct(cut)}`), element("span", "", "100%"));
  node.append(track, ends);
  return node;
}

function emptyResult() {
  const box = element("div", "empty-result");
  const text = state.model === "symptom"
    ? "The three conditions that best match your symptoms will show here, with the symptoms they share."
    : "The score will show here on a 0 to 100% scale, with the model's cutoff and the inputs that moved it.";
  box.append(element("h2", "", "Result"), element("p", "", `No result yet. ${text}`));
  return box;
}

// Clearing or switching models empties the result. Editing an input only marks
// it out of date, so the live region is not rebuilt on every keystroke.
function invalidate(mode = "reset") {
  state.revision += 1;
  state.run?.abort();
  state.run = null;
  showError();
  busy(false);
  const hasResult = results.dataset.filled === "true";
  if (mode === "stale" && !hasResult && results.firstChild) return;
  if (mode === "stale" && hasResult) {
    if (!results.classList.contains("is-stale")) {
      results.classList.add("is-stale");
      results.prepend(element("p", "stale-note", "Inputs changed. Run the prediction again to update this result."));
    }
    return;
  }
  results.classList.remove("is-stale");
  results.dataset.filled = "false";
  results.replaceChildren(emptyResult());
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options, cache: "no-store",
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(60000)])
      : AbortSignal.timeout(60000),
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
  // The free host sleeps when idle and takes about a minute to wake.
  if (failure.name === "TimeoutError") return "The server took too long to answer. It may have been asleep; try again now.";
  if (failure instanceof TypeError) return "Could not reach the service. Check your connection and try again.";
  return failure.message;
}

// Symptom picker: a search box that drives a multi-select list. Only the box
// and the chosen-symptom chips take Tab stops; arrow keys move through the list.

function symptomLabel(key) {
  return state.symptoms.find((item) => item.key === key)?.label ?? key.replaceAll("_", " ");
}

function toggleSymptom(key) {
  invalidate("stale");
  if (state.selected.has(key)) state.selected.delete(key);
  else if (state.selected.size >= 50) return showError("Choose up to 50 symptoms per prediction.");
  else state.selected.add(key);
  selectedSymptoms();
  symptomOptions();
}

function selectedSymptoms() {
  const selected = $("#selected-symptoms");
  selected.replaceChildren();
  for (const key of state.selected) {
    const chip = element("button", "symptom-chip", symptomLabel(key));
    chip.type = "button";
    chip.setAttribute("aria-label", `Remove ${symptomLabel(key)}`);
    chip.addEventListener("click", () => {
      toggleSymptom(key);
      $("#symptom-search").focus();
    });
    selected.append(chip);
  }
  if (!state.selected.size) selected.append(element("span", "selected-empty", "No symptoms chosen yet."));
  $("#input-count").textContent = `${state.selected.size} selected`;
}

function visibleSymptoms() {
  const query = $("#symptom-search").value.trim().toLowerCase();
  return state.symptoms.filter((item) => item.label.toLowerCase().includes(query));
}

function symptomOptions() {
  const list = $("#symptom-options");
  const search = $("#symptom-search");
  const matches = visibleSymptoms();
  state.active = Math.min(state.active, matches.length - 1);
  list.replaceChildren();
  matches.forEach((symptom, index) => {
    const option = element("li", "symptom-option", symptom.label);
    option.id = `symptom-${symptom.key}`;
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", String(state.selected.has(symptom.key)));
    if (index === state.active) option.classList.add("is-active");
    option.addEventListener("mousedown", (event) => event.preventDefault());
    option.addEventListener("click", () => {
      state.active = index;
      toggleSymptom(symptom.key);
    });
    list.append(option);
  });
  if (!matches.length) list.append(element("li", "no-matches", "No symptom matches that search. Try a shorter word."));
  const active = matches[state.active];
  if (active) {
    search.setAttribute("aria-activedescendant", `symptom-${active.key}`);
    document.getElementById(`symptom-${active.key}`)?.scrollIntoView({ block: "nearest" });
  } else {
    search.removeAttribute("aria-activedescendant");
  }
}

function renderSymptoms() {
  const wrap = element("div", "symptom-picker");
  const label = element("label", "", `Search ${state.symptoms.length} symptoms`);
  label.htmlFor = "symptom-search";
  const search = element("input");
  search.type = "search";
  search.id = "symptom-search";
  search.autocomplete = "off";
  search.setAttribute("role", "combobox");
  search.setAttribute("aria-controls", "symptom-options");
  search.setAttribute("aria-expanded", "true");
  search.setAttribute("aria-autocomplete", "list");
  search.setAttribute("aria-describedby", "picker-hint");
  search.addEventListener("input", () => {
    state.active = search.value.trim() ? 0 : -1;
    symptomOptions();
  });
  search.addEventListener("keydown", (event) => {
    const count = visibleSymptoms().length;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!count) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      state.active = (state.active + step + count) % count;
      symptomOptions();
    } else if (event.key === "Enter" && state.active >= 0 && count) {
      // Enter adds the highlighted symptom instead of submitting the form.
      event.preventDefault();
      toggleSymptom(visibleSymptoms()[state.active].key);
    } else if (event.key === "Escape" && search.value) {
      search.value = "";
      state.active = -1;
      symptomOptions();
    }
  });
  const hint = element("p", "picker-hint", "Type to filter. Up and down arrows move through the list, Enter adds or removes a symptom.");
  hint.id = "picker-hint";
  const selected = element("div", "selected-symptoms");
  selected.id = "selected-symptoms";
  const list = element("ul", "symptom-options");
  list.id = "symptom-options";
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-multiselectable", "true");
  list.setAttribute("aria-label", "Symptoms");
  wrap.append(label, search, hint, selected, list);
  fields.replaceChildren(wrap);
  state.active = -1;
  selectedSymptoms();
  symptomOptions();
}

// Clinical forms, built from the input specs each model publishes

// Some published descriptions only repeat the label ("Headache.") or describe
// the API encoding ("1 for male, 0 for female"); neither helps a person.
function usefulDescription(spec) {
  const text = (spec.description || "").trim();
  const plain = (value) => value.toLowerCase().replace(/[.\s]+$/, "");
  if (!text || /^\d/.test(text) || /^Age in years/.test(text)) return "";
  if (plain(text) === plain(spec.label) || plain(text) === plain(`${spec.label} in ${spec.unit}`)) return "";
  return text;
}

const isYesNo = (spec) => spec.choices && Object.keys(spec.choices).join() === "0,1"
  && spec.choices["0"] === "no" && spec.choices["1"] === "yes";

// Short two-way choices read faster as a segmented control than as a menu.
function segmented(spec) {
  const fieldset = element("fieldset", "segmented");
  fieldset.append(element("legend", "", spec.label));
  const segments = element("div", "segments");
  const options = Object.entries(spec.choices);
  if (!spec.required) options.unshift(["", "Not given"]);
  for (const [value, description] of options) {
    const label = element("label");
    const input = element("input");
    input.type = "radio";
    input.name = spec.name;
    input.value = value;
    input.required = Boolean(spec.required);
    if (value === "") input.defaultChecked = input.checked = true;
    label.append(input, element("span", "", value === "" ? description : capitalize(description)));
    segments.append(label);
  }
  fieldset.append(segments);
  const description = usefulDescription(spec);
  if (description) fieldset.append(element("small", "", description));
  return fieldset;
}

function clinicalField(spec) {
  if (spec.choices && Object.keys(spec.choices).length <= 2) return segmented(spec);
  const wrapper = element("div", "field");
  const label = element("label", "", spec.label);
  label.htmlFor = `field-${spec.name}`;
  let control;
  const helpParts = [];
  if (spec.choices) {
    control = element("select");
    const empty = element("option", "", spec.required ? "Choose an option" : "Not given");
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
    if (spec.minimum !== undefined && spec.maximum !== undefined) {
      helpParts.push(`${spec.minimum} to ${spec.maximum}${spec.unit ? ` ${spec.unit}` : ""}.`);
    }
  }
  control.id = `field-${spec.name}`;
  control.name = spec.name;
  control.required = Boolean(spec.required);
  const description = usefulDescription(spec);
  if (description) helpParts.unshift(description);
  if (spec.unit && !spec.choices) {
    const wrap = element("div", "input-wrap");
    wrap.append(control, element("span", "unit", spec.unit));
    wrapper.append(label, wrap);
  } else {
    wrapper.append(label, control);
  }
  if (helpParts.length) {
    const help = element("small", "", helpParts.join(" "));
    help.id = `help-${spec.name}`;
    control.setAttribute("aria-describedby", help.id);
    wrapper.append(help);
  }
  return wrapper;
}

function renderClinical() {
  const inputs = state.details.inputs;
  const required = element("div", "field-grid");
  const optional = element("div", "field-grid");
  const yesNo = element("div", "yes-no-list");
  for (const spec of inputs) {
    if (spec.required) required.append(clinicalField(spec));
    else if (isYesNo(spec) && inputs.filter(isYesNo).length > 4) yesNo.append(clinicalField(spec));
    else optional.append(clinicalField(spec));
  }
  const optionalCount = inputs.filter((item) => !item.required).length;
  const children = [required];
  if (optionalCount) {
    const details = element("details", "optional-inputs");
    details.append(element("summary", "", `More measurements (${optionalCount} optional)`), optional);
    if (yesNo.childElementCount) {
      details.append(element("h3", "", "Signs and conditions. Leave as Not given if unknown."), yesNo);
    }
    children.push(details);
  }
  fields.replaceChildren(...children);
  const requiredCount = inputs.length - optionalCount;
  $("#input-count").textContent = `${requiredCount} required, ${optionalCount} optional`;
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
  const help = element("p", "", "PNG or JPEG, up to 2 MB, cropped to one cell. You can also drop the file here.");
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

const tabs = [...document.querySelectorAll("[role=tab][data-model]")];

function selectTab(model) {
  for (const tab of tabs) {
    const selected = tab.dataset.model === model;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
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
  selectTab(model);
  const { title, description, scope } = models[model];
  $("#model-title").textContent = title;
  $("#model-description").textContent = description;
  $("#scope-note").textContent = scope;
  $("#input-count").textContent = "";
  $("#example-button").hidden = !examples[model];
  $("#example-button").disabled = true;
  fields.setAttribute("aria-busy", "true");
  fields.replaceChildren(element("p", "loading-copy", "Loading inputs..."));
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
    const retry = element("button", "button", "Try loading again");
    retry.type = "button";
    retry.addEventListener("click", () => chooseModel(model));
    fields.replaceChildren(element("p", "loading-copy", "The inputs did not load."), retry);
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
  results.append(element("h2", "", "Result"));
  results.append(element("p", "result-context", "Closest matches in the training data, not a diagnosis."));
  const urgent = [...state.selected].filter((key) => urgentSymptoms.has(key));
  if (urgent.length) {
    const box = element("div", "urgent");
    box.append(element("strong", "", "Get medical help today."),
      document.createTextNode(`${urgent.map(symptomLabel).map(capitalize).join(", ")} can be a sign of something serious. `
        + "Do not wait on this ranking."));
    results.append(box);
  }
  if (state.selected.size < 3) {
    results.append(element("p", "caution", "Too few symptoms to rank well. With one or two, many conditions match "
      + "about equally and the order below means little."));
  }
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
      row.append(element("p", "", `Shares: ${item.matched_symptoms.map(symptomLabel).join(", ")}.`));
    }
    list.append(row);
  }
  results.append(list);
  const accuracy = data.model_metrics?.cross_validation?.accuracy;
  if (accuracy !== undefined) {
    results.append(element("p", "model-footnote", `Cross-validated accuracy on clean textbook patterns: ${pct(accuracy, 0)}. `
      + "Real symptoms are messier, so expect it to do worse on them."));
  }
}

function describeValue(spec, value) {
  if (value === null || value === undefined) return "not given";
  if (spec?.choices) return spec.choices[String(Math.round(value))] ?? String(value);
  return `${Number(value.toFixed(2))}${spec?.unit ? ` ${spec.unit}` : ""}`;
}

function factorsSection(data) {
  const section = element("section", "result-section");
  section.append(element("h3", "", "Inputs that changed the score"));
  const specs = new Map(state.details.inputs.map((item) => [item.name, item]));
  const largest = Math.max(...data.factors.map((factor) => Math.abs(factor.effect)), 0.001);
  for (const factor of data.factors) {
    const spec = specs.get(factor.feature);
    const row = element("div", factor.effect < 0 ? "factor is-negative" : "factor");
    const name = element("span", "", factor.label);
    const change = `${factor.effect >= 0 ? "+" : "-"}${Math.abs(factor.effect * 100).toFixed(1)} pts`;
    const bar = element("div", "factor-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--size", String(Math.abs(factor.effect) / largest));
    bar.append(element("span"));
    row.append(name, element("strong", "", change),
      element("span", "factor-value", `${describeValue(spec, factor.value)}, typical ${describeValue(spec, factor.typical_value)}`),
      bar);
    section.append(row);
  }
  section.append(element("p", "", data.factors.length
    ? "Each figure is the change in percentage points when that input is swapped for a typical training value."
    : "None of the values you gave moved the rounded score away from what typical values give."));
  return section;
}

function filledSection(data) {
  const specs = new Map(state.details.inputs.map((item) => [item.name, item]));
  const section = element("section", "result-section");
  section.append(element("h3", "", `Left blank, filled in (${data.imputed_fields.length})`));
  const list = element("ul", "filled-list");
  for (const name of data.imputed_fields) {
    const spec = specs.get(name);
    const value = data.imputed_values?.[name];
    const how = spec?.choices ? "most common" : "median";
    const row = element("li");
    row.append(element("span", "", spec?.label || name),
      element("span", "", value === undefined ? how : `${describeValue(spec, value)} (${how})`));
    list.append(row);
  }
  section.append(list);
  return section;
}

function operatingPoint(data, isImage) {
  const point = data.operating_point;
  if (!point) return null;
  const [cases, others] = isImage ? ["infected cells", "uninfected cells"] : ["who had it", "who did not"];
  const list = element("dl", "metric-list");
  const add = (term, value, note) => {
    const dd = element("dd", "", `${Math.round(value * 100)}% `);
    dd.append(element("small", "", note));
    list.append(element("dt", "", term), dd);
  };
  add("Sensitivity", point.sensitivity, `of ${cases} were flagged`);
  add("Specificity", point.specificity, `of ${others} were cleared`);
  if (point.ppv !== undefined) add("Precision", point.ppv, "of flags were real cases");
  const section = element("section", "result-section");
  section.append(element("h3", "", "This cutoff on held-out data"), list);
  return section;
}

function renderRiskResult(data) {
  const isImage = state.model === "malaria";
  results.append(element("h2", "", "Result"));
  results.append(element("p", "result-context", isImage ? "Single-cell classification" : capitalize(data.condition || models[state.model].title)));
  if (isImage && state.preview) {
    const image = element("div", "result-image");
    const thumbnail = element("img");
    thumbnail.src = state.preview;
    thumbnail.alt = "";
    image.append(thumbnail, element("span", "", "Your image"));
    results.append(image);
  }
  const flaggedClass = data.flagged ? " is-flagged" : "";
  results.append(element("p", "estimate-label", "Model estimate"));
  results.append(element("p", "estimate", pct(data.probability)));
  const verdict = isImage
    ? data.flagged ? "Flagged: reads as parasitized" : "Not flagged: reads as uninfected"
    : data.flagged ? "Flagged: at or above the cutoff" : "Not flagged: below the cutoff";
  results.append(element("p", `outcome${flaggedClass}`, verdict));
  const scale = ruler(data.threshold, data.probability);
  if (data.flagged) scale.classList.add("is-flagged");
  results.append(scale);
  const point = operatingPoint(data, isImage);
  if (point) results.append(point);
  if (isImage) {
    results.append(element("p", "model-footnote",
      "This reads one cell. It says nothing about whether a person has malaria; that takes hundreds of cells and a microscopist."));
  } else {
    results.append(factorsSection(data));
    if (data.imputed_fields.length) results.append(filledSection(data));
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
    results.classList.remove("is-stale");
    results.replaceChildren();
    if (state.model === "symptom") renderSymptomsResult(data);
    else renderRiskResult(data);
    results.append(element("p", "disclaimer", data.disclaimer));
    results.dataset.filled = "true";
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
  if (event.target.matches("input[type=number], input[type=radio], select")) invalidate("stale");
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

// Tabs follow the manual-activation pattern: arrows move focus, Enter or a
// click opens the model. Opening on focus would fetch on every arrow press.
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => {
    const model = tab.dataset.model;
    if (location.hash !== `#/checks/${model}`) history.pushState(null, "", `#/checks/${model}`);
    openModel(model);
  });
  tab.addEventListener("keydown", (event) => {
    const keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
    let next = null;
    if (event.key in keys) next = (index + keys[event.key] + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    if (next === null) return;
    event.preventDefault();
    tabs[next].focus();
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
    status.textContent = `${loaded} of ${entries.length} models loaded`;
    status.classList.add(loaded === entries.length ? "is-ready" : "is-degraded");
    const logging = health.database.logging_enabled;
    $("#submit-note").textContent = logging
      ? "This deployment records inputs for review. Do not enter anything that identifies a person."
      : "Inputs go to the server to run the model and are not stored.";
    $("#privacy-note").textContent = logging
      ? "This deployment stores submitted inputs and results so an administrator can review them. Do not submit identifying information. Uploaded images are stored only as their dimensions and a hash."
      : "Prediction logging is off on this deployment. Your inputs go to the server to run the model and are not kept. This site does not save them in your browser either.";
  } catch {
    status.textContent = "Service status unavailable";
    status.classList.add("is-degraded");
    $("#submit-note").textContent = "Inputs go to the server to run the model. Whether they are stored is unknown.";
    $("#privacy-note").textContent = "Could not check whether this deployment logs predictions. Do not submit identifying information.";
  }
}

// Model details feed the start page table and the Models page.

const modelPage = {
  symptom: { name: "Symptom checker", unit: "rows" },
  heart: { name: "Heart disease", unit: "patients" },
  diabetes: { name: "Diabetes", unit: "women" },
  liver: { name: "Liver disease", unit: "patients" },
  dengue: { name: "Dengue severity", unit: "cases" },
  malaria: { name: "Malaria cell", unit: "cells" },
};
let detailsLoaded = null;

function loadDetails() {
  detailsLoaded ??= Promise.all(Object.keys(modelPage).map((name) => request(`/models/${name}`)))
    .catch((failure) => {
      detailsLoaded = null;
      throw failure;
    });
  return detailsLoaded;
}

async function fillCheckTable() {
  try {
    const details = await loadDetails();
    for (const detail of details) {
      const row = document.querySelector(`[data-check="${detail.name}"]`);
      if (!row) continue;
      const score = row.querySelector(".score");
      if (detail.name === "symptom") {
        score.textContent = `${pct(detail.metrics.cross_validation.accuracy, 0)} accuracy`;
      } else {
        score.textContent = `${detail.metrics.test.roc_auc.toFixed(2)} AUC`;
      }
      const needs = row.querySelector(".needs");
      const required = (detail.inputs || []).filter((item) => item.required).map((item) => item.label);
      if (required.length) needs.textContent = required.join(", ");
    }
  } catch {
    // The table still lists every check; only the scores stay blank.
  }
}

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
  const run = element("a", "", "Open this check");
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
      `Sensitivity ${pct(test.sensitivity, 0)}, specificity ${pct(test.specificity, 0)}, `
      + `precision ${pct(test.ppv, 0)}.`);
    fact(facts, "Sources", sourceLinks(data.sources || []));
  }
  fact(facts, "Known limit", models[name].scope);
  section.append(intro, facts);
  return section;
}

async function loadModels() {
  const box = $("#model-sections");
  if (box.querySelector(".model-section")) return;
  try {
    const details = await loadDetails();
    box.replaceChildren(...details.map((detail) => modelSection(detail.name, detail)));
  } catch (failure) {
    const retry = element("button", "button", "Try loading again");
    retry.type = "button";
    retry.addEventListener("click", () => route());
    box.replaceChildren(element("p", "loading-copy", `Model details did not load. ${explainFailure(failure)}`), retry);
  } finally {
    box.setAttribute("aria-busy", "false");
  }
}

// Pages live in this one document and are addressed by the hash, so the
// server keeps serving a single file.

const titles = {
  home: "Disease Prediction",
  checks: "Checks | Disease Prediction",
  models: "Models | Disease Prediction",
  about: "About | Disease Prediction",
  missing: "Page not found | Disease Prediction",
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
  if (page === "home") fillCheckTable();
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
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
    setMenu(false);
    menuButton.focus();
  }
});
document.addEventListener("click", (event) => {
  if (!nav.contains(event.target) && !menuButton.contains(event.target)) setMenu(false);
});
$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  const main = $("#main");
  main.focus();
  main.scrollIntoView();
});
window.addEventListener("hashchange", route);

route();
loadStatus();
