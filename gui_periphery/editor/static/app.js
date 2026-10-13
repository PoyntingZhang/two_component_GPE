"use strict";

const $ = selector => document.querySelector(selector);
const clone = value => JSON.parse(JSON.stringify(value));
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const draftKey = "two-component-gpe-config-editor-v3";
const savePathKey = draftKey + "-save-path";
const formulaTimers = new WeakMap();
let config, defaults, stageTemplate, defaultSavePath, selected = 0, tab = "evolution", yamlText = "", revision = 0, timer, dirty = false, replaceAction;
const stage = () => config.stages[selected];
const stageName = (item, index) => item.name.trim() || `Stage ${index + 1}`;
const get = (object, path) => path.split(".").reduce((value, key) => value[key], object);
function set(object, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((item, key) => item[key], object)[last] = value;
}
const fmt = value => Number.isFinite(value) ? new Intl.NumberFormat("en", {maximumFractionDigits: 5}).format(value) : "—";

async function api(path, data) {
  const options = {headers: {"X-GPE-Session": window.GPE_SESSION}};
  if (data !== undefined) Object.assign(options, {
    method: "POST", headers: {...options.headers, "Content-Type": "application/json"}, body: JSON.stringify(data)
  });
  const response = await fetch(path.replace(/^\//, ""), options);
  const result = await response.json();
  if (!response.ok && !result.errors) throw new Error(result.error || "The request could not be completed.");
  return result;
}

function notice(message, error = false) {
  $("#notice").textContent = message;
  $("#notice").className = "notice" + (error ? " error" : "");
  $("#notice").hidden = false;
}

function rememberSavePath(path) {
  $("#save-path").value = path;
  try { localStorage.setItem(savePathKey,path); } catch (_) { /* Saving still works without browser storage. */ }
}

function field(label, path, {type = "number", unit = "", hint = "", options, min, step = "any", wide = false} = {}) {
  const value = get(stage(), path), id = "f-" + path.replaceAll(".", "-");
  const attributes = `id="${id}" data-path="${path}"`;
  let input;
  if (options) input = `<select ${attributes}>${options.map(([key, name]) => `<option value="${key}" ${key === value ? "selected" : ""}>${name}</option>`).join("")}</select>`;
  else input = `<input ${attributes} type="${type}" value="${escapeHTML(value)}" ${type === "number" ? `step="${step}" ${min !== undefined ? `min="${min}"` : ""}` : 'spellcheck="false"'}>`;
  return `<label class="field ${wide ? "span-all" : ""}" for="${id}">${label}${unit ? `<span class="unit">${unit}</span>` : ""}${input}${hint ? `<small>${hint}</small>` : ""}</label>`;
}

function toggle(label, path, disabled = false) {
  return `<label class="toggle"><input type="checkbox" data-path="${path}" ${get(stage(), path) ? "checked" : ""} ${disabled ? "disabled" : ""} aria-label="${label}"><span>${get(stage(), path) ? "On" : "Off"}</span></label>`;
}

function check(label, path) {
  return `<label class="check-line"><input type="checkbox" data-path="${path}" ${get(stage(), path) ? "checked" : ""}>${label}</label>`;
}

function section(title, content, {description = "", control = ""} = {}) {
  return `<section class="form-section"><div class="section-heading"><h3>${title}</h3>${control}</div>${description ? `<p>${description}</p>` : ""}${content}</section>`;
}

function initialEditor(component) {
  const group = `initial_state.component${component}`, initial = stage().initial_state[`component${component}`];
  return section(`Component ${component} · initial state`, `<div class="form-grid">
    ${field("Start from", `${group}.type`, {options:[["gaussian","Generate a Gaussian"],["load","Load a saved wavefunction"],["continue","Continue previous stage"]]})}
    ${initial.type === "gaussian" ? field("Gaussian width", `${group}.sigma`, {unit:`σ${component === 1 ? "₁" : "₂"}`,min:0}) : ""}
    ${initial.type === "load" ? field("Wavefunction file", `${group}.path`, {type:"text",wide:true,hint:"Full target path to psi1_STEP.npy or psi2_STEP.npy. Keep its JSON metadata and grid.npz beside it. The other field is not loaded."}) : ""}
    </div>${initial.type === "continue" ? `<div class="hint">${selected === 0 ? "Stage 1 has no previous state. Choose a Gaussian or load a file." : `Carry component ${component} forward with its remaining norm.`}</div>` : ""}`);
}

function renderEvolution() {
  const s = stage();
  return section("Stage settings", `<div class="form-grid">
    ${field("Name", "name", {type:"text", hint:`Optional · defaults to Stage ${selected + 1}`})}
    ${field("Simulation type", "evolution.type", {options:[["imaginary","Imaginary time"],["real","Real time"]]})}
    ${field("Total steps", "evolution.steps", {min:1,step:1})}
    ${field("Step size", "evolution.dt", {min:0,unit:s.evolution.type === "imaginary" ? "Δτ" : "Δt"})}
    </div><div class="derived" id="duration">${s.evolution.type === "imaginary" ? "Imaginary" : "Real"} duration = ${fmt(s.evolution.steps * s.evolution.dt)}</div>`)
    + initialEditor(1) + initialEditor(2)
    + section("Cloud", `<div class="form-grid">
    ${field("Scattering length a₁₁", "cloud.a11", {unit:"a₀"})}
    ${field("Scattering length a₂₂", "cloud.a22", {unit:"a₀"})}
    ${field("Scattering length a₁₂", "cloud.a12", {unit:"a₀"})}
    ${field("Reference atom number N₁", "cloud.N1", {min:1,step:1})}
    ${field("Reference atom number N₂", "cloud.N2", {min:1,step:1})}
    </div><div class="hint"><strong>Stage time starts at zero.</strong> ${s.evolution.type === "imaginary" ? "Each component is normalized separately after each imaginary step." : "Real-time evolution preserves loss from the absorber; it does not renormalize ψ."}</div>`);
}

function potentialEditor(component, kind) {
  const group = "potential" + component, id = group + "-" + kind;
  const dynamic = kind === "dynamic", locked = dynamic && stage().evolution.type === "imaginary";
  const title = dynamic ? "Time dependent (t=0 at stage start)" : "Time independent";
  const variables = dynamic ? "x, y, z, t" : "x, y, z";
  return section(`Component ${component} · ${title}`, `<label class="field" for="formula-${id}">V(${variables})
    <textarea id="formula-${id}" class="code" rows="2" spellcheck="false" data-path="${group}.${kind}.expression" data-formula="${id}" data-component="${group}" data-kind="${kind}" aria-describedby="formula-help-${id} formula-preview-${id}" ${locked || !stage()[group][kind].enabled ? "disabled" : ""}>${escapeHTML(stage()[group][kind].expression)}</textarea></label>
    <div id="formula-preview-${id}" class="formula-preview pending" role="status" aria-live="polite">Checking formula…</div>
    <p class="formula-help" id="formula-help-${id}">Variables: ${variables}. Constant: pi. Use + − * / ** and parentheses.<br>Functions: sin, cos, tan, exp, sqrt, tanh, abs, log. Angles in radians.<br>Put coefficients directly in the formula; entering 0 switches this contribution off.</p>
    ${locked ? '<div class="hint">Imaginary time evolution does not accept a time dependent potential.</div>' : `<p class="potential-state" id="potential-state-${id}">${stage()[group][kind].enabled ? "This contribution is on." : "Off. Enable this contribution to enter a formula."}</p>`}`,
    {control:toggle(`Enable ${dynamic ? "time dependent" : "time independent"} potential`, `${group}.${kind}.enabled`, locked)});
}

async function checkFormula(input, commit = false) {
  clearTimeout(formulaTimers.get(input));
  const source = input.value, id = input.dataset.formula, kind = input.dataset.kind, group = input.dataset.component, owner = stage();
  const preview = $(`#formula-preview-${id}`);
  try {
    const result = await api("/api/formula", {expression:source, time_dependent:kind === "dynamic"});
    if (commit && result.valid && result.is_zero && config.stages.includes(owner) &&
        owner[group][kind].expression === source && owner[group][kind].enabled) {
      owner[group][kind].enabled = false;
      owner[group][kind].expression = "0";
      if (stage() === owner) renderEditor();
      changed();
      return;
    }
    if (!input.isConnected || input.value !== source) return;
    preview.className = "formula-preview " + (result.valid ? "valid" : "invalid");
    if (result.valid) {
      preview.innerHTML = result.mathml;
      input.removeAttribute("aria-invalid");
    } else {
      preview.innerHTML = `<code>${escapeHTML(source || "(empty formula)")}</code><span>${escapeHTML(result.message)}</span>`;
      input.setAttribute("aria-invalid", "true");
    }
  } catch (error) {
    if (!input.isConnected || input.value !== source) return;
    preview.className = "formula-preview invalid";
    preview.textContent = "Could not check this formula: " + error.message;
  }
}

function renderPotentials() {
  const s = stage(), imaginary = s.evolution.type === "imaginary";
  return '<p class="potential-intro">Each component has its own static and dynamic potentials, added to the shared box and disorder.</p>'
    + potentialEditor(1,"static") + potentialEditor(1,"dynamic")
    + potentialEditor(2,"static") + potentialEditor(2,"dynamic")
    + section("Momentum cutoff", s.k_cut.enabled ? `<div class="form-grid">${field("Cutoff magnitude", "k_cut.value", {min:0,unit:"k"})}</div>` : '<p class="off-message">No momentum cutoff.</p>',
      {description:"Real time only. Remove modes with |k| above this value during each kinetic step.", control:toggle("Enable momentum cutoff","k_cut.enabled",imaginary)})
    + section("Box trap", s.box.enabled ? `<div class="form-grid">
      ${field("Radius", "box.R", {unit:"R",min:0})}${field("Length", "box.L", {unit:"L",min:0})}
      ${field("Trap depth", "box.trap_depth", {unit:"U_D",min:0})}${field("Wall thickness", "box.wall_thickness", {min:0})}
      </div>` : '<p class="off-message">No box potential in this stage.</p>', {control:toggle("Enable box trap","box.enabled")})
    + section("Absorber", s.absorber.enabled ? `<div class="form-grid">
      ${field("Activation distance", "absorber.activation_distance", {min:0})}
      ${field("Strength", "absorber.strength", {min:0})}</div>` : `<p class="off-message">${imaginary ? "Imaginary time evolution does not accept an absorber." : "Absorption is off for this stage."}</p>`, {control:toggle("Enable absorber","absorber.enabled",imaginary)})
    + section("Gaussian disorder", s.noise.enabled ? `<div class="form-grid">
      ${field("Fraction of trap depth", "noise.fraction_of_trap_depth", {min:0})}
      ${field("Random seed", "noise.seed", {min:0,step:1})}
      ${field("Correlation length", "noise.correlation_length", {min:0,hint:"Use 0 for uncorrelated noise. Correlated disorder is not implemented yet."})}
      </div><div class="derived" id="noise-amplitude">Noise σ = ${fmt(s.noise.fraction_of_trap_depth*s.box.trap_depth)}</div>
      <div class="hint">Noise covers the whole simulation region. Its strength follows this stage’s trap depth. The same seed reproduces the underlying pattern on the same grid.</div>` : '<p class="off-message">No disorder is added in this stage.</p>', {control:toggle("Enable Gaussian disorder","noise.enabled")});
}

const intervals = [
  ["Scalars · N, Eₖᵢₙ, Lz","print_scalar_interval"], ["Momentum profiles · n(k), Nk","show_nk_interval"],
  ["Horizontal TOF · integrate kz","show_TOF_image_interval"], ["Vertical TOF · integrate kx","show_verticle_TOF_interval"],
  ["TOF slice · kz = 0","show_TOF_slice_interval"], ["Vertical TOF slice · kx = 0","show_verticle_TOF_slice_interval"],
  ["Magnifier · integrate z","show_magnifier_interval"], ["Magnifier slice · z = 0","show_magnifier_slice_interval"],
  ["Phase slice · z = 0","show_phase_interval"]
];

function outputFolder(s, index, useOverride = true) {
  const slug = s.name.trim().replace(/[^A-Za-z0-9_-]+/g,"_").replace(/^_+|_+$/g,"");
  return (useOverride && s.save.directory) || `stage_${String(index+1).padStart(2,"0")}${slug ? "_"+slug : ""}`;
}

function renderOutputs() {
  return section("Monitoring intervals", `<div class="interval-grid">${intervals.map(([name,key]) => field(name, "monitor."+key,{min:0,step:1})).join("")}</div>`, {description:"Completed steps within this stage. Set 0 to disable. Both components write diagnostics directly to the run folder on Drive."})
    + section("Save intervals", `<div class="form-grid">
      ${field("Scalars", "save.scalar_interval",{min:0,step:1})}${field("Momentum profiles", "save.nk_interval",{min:0,step:1})}
      ${field("Wavefunction snapshots", "save.psi_interval",{min:0,step:1})}
      </div><div class="save-options">${check("Save initial state","save.save_initial")}${check("Save final state","save.save_final")}</div>`, {description:"Separate from monitoring. Set 0 to disable periodic saves."})
    + section("Stage output", `<div class="form-grid">${field("Directory override", "save.directory", {type:"text",wide:true,hint:"Optional subfolder under both output roots. Prepared Colab runs require a relative subfolder. Each run gets a new timestamped folder."})}</div><div class="hint" id="output-location"></div>`);
}

function renderStageList() {
  $("#stage-count").textContent = config.stages.length;
  $("#stage-list").innerHTML = config.stages.map((s,i) => `<button class="stage-item ${i === selected ? "active" : ""}" data-stage="${i}" ${i === selected ? 'aria-current="step"' : ""}>
    <span class="stage-index">${String(i+1).padStart(2,"0")}</span><span><strong>${escapeHTML(stageName(s,i))}</strong><small>${s.evolution.type === "imaginary" ? "Imaginary time" : "Real time"}</small><span class="stage-line">${fmt(s.evolution.steps)} steps · ${[1,2].map(c=>`ψ${c}: ${{continue:"continue",load:"load",gaussian:"Gaussian"}[s.initial_state[`component${c}`].type]}`).join(" · ")}</span></span></button>`).join("");
  $("#stage-number").textContent = "STAGE " + String(selected+1).padStart(2,"0");
  $("#stage-heading").textContent = stageName(stage(),selected);
  $("#total-steps").textContent = fmt(config.stages.reduce((n,s)=>n+(Number(s.evolution.steps)||0),0));
  $("#real-time").textContent = fmt(config.stages.filter(s=>s.evolution.type === "real").reduce((n,s)=>n+s.evolution.steps*s.evolution.dt,0));
  $("#grid-spacing").textContent = fmt(config.simulation.simulation_range/config.simulation.N_grids);
  $("#state-memory").textContent = fmt(32*config.simulation.N_grids**3/1024**3)+" GiB";
}

function renderEditor() {
  renderStageList();
  $("#stage-tools").innerHTML = `<button class="icon-button" data-action="up" title="Move stage up" aria-label="Move stage up" ${selected === 0 ? "disabled" : ""}>↑</button>
    <button class="icon-button" data-action="down" title="Move stage down" aria-label="Move stage down" ${selected === config.stages.length-1 ? "disabled" : ""}>↓</button>
    <button class="icon-button" data-action="duplicate" title="Duplicate stage" aria-label="Duplicate stage">⧉</button>
    <button class="icon-button danger" data-action="delete" title="Delete stage" aria-label="Delete stage" ${config.stages.length === 1 ? "disabled" : ""}>×</button>`;
  document.querySelectorAll("[data-tab]").forEach(button => {
    button.classList.toggle("active",button.dataset.tab === tab);
    button.setAttribute("aria-selected",String(button.dataset.tab === tab));
  });
  $("#editor").innerHTML = tab === "evolution" ? renderEvolution() : tab === "potentials" ? renderPotentials() : renderOutputs();
  document.querySelectorAll("[data-formula]").forEach(input => checkFormula(input));
  derived();
}

function derived() {
  const s = stage();
  if ($("#duration")) $("#duration").textContent = `${s.evolution.type === "imaginary" ? "Imaginary" : "Real"} duration = ${fmt(s.evolution.steps*s.evolution.dt)}`;
  if ($("#noise-amplitude")) $("#noise-amplitude").textContent = `Noise σ = ${fmt(s.noise.fraction_of_trap_depth*s.box.trap_depth)}`;
  if ($("#output-location")) {
    const folder = outputFolder(s,selected);
    const absolute = folder.startsWith("/") || folder.startsWith("~");
    const data = absolute ? folder + "/<run>" : config.simulation.data_root.replace(/\/$/,"") + "/<run>/" + folder;
    const monitor = config.simulation.monitor_root.replace(/\/$/,"") + "/<run>/" + (absolute ? outputFolder(s,selected,false) : folder);
    $("#output-location").textContent = "Wavefunctions: " + data + " · Monitoring: " + monitor;
  }
}

function renderAll() {
  document.querySelectorAll("[data-global]").forEach(input => input.value = config.simulation[input.dataset.global]);
  renderEditor();
}

function changed() {
  dirty = true;
  revision++;
  yamlText = "";
  $("#save-config").disabled = true;
  $("#copy-yaml").disabled = true;
  $("#validation-status").textContent = "Checking configuration…";
  $("#status-dot").className = "";
  try { localStorage.setItem(draftKey,JSON.stringify(config)); } catch (_) { /* Export remains available without browser storage. */ }
  renderStageList();
  derived();
  clearTimeout(timer);
  timer = setTimeout(validate,220);
}

async function validate() {
  const current = revision;
  try {
    const result = await api("/api/render",{config});
    if (current !== revision) return;
    const errors = result.errors || [];
    yamlText = result.yaml || "";
    $("#yaml-preview").value = yamlText || "Resolve the configuration issues below to generate YAML.";
    $("#save-config").disabled = errors.length > 0;
    $("#copy-yaml").disabled = errors.length > 0;
    $("#validation-status").textContent = errors.length ? `${errors.length} setting${errors.length === 1 ? " needs" : "s need"} attention` : "Ready to export · draft saved in this browser";
    $("#status-dot").className = errors.length ? "invalid" : "valid";
    $("#validation").hidden = !errors.length;
    $("#validation").innerHTML = errors.length ? `<strong>Check these settings</strong><ul>${errors.map(e=>`<li><button data-error="${escapeHTML(e.path)}">${escapeHTML(e.message)}</button></li>`).join("")}</ul>` : "";
    document.querySelectorAll("[aria-invalid]").forEach(input => input.removeAttribute("aria-invalid"));
    errors.forEach(e => {
      const prefix = `stages.${selected}.`;
      const target = e.path.startsWith(prefix) ? document.querySelector(`[data-path="${CSS.escape(e.path.slice(prefix.length))}"]`) : e.path.startsWith("simulation.") ? document.querySelector(`[data-global="${CSS.escape(e.path.slice(11))}"]`) : null;
      if (target) target.setAttribute("aria-invalid","true");
    });
  } catch (error) {
    if (current !== revision) return;
    $("#validation-status").textContent = "Editor connection lost · your browser draft is retained";
    $("#status-dot").className = "invalid";
    notice(error.message,true);
  }
}

function replaceConfig(value, message) {
  config = clone(value); selected = 0; tab = "evolution";
  renderAll(); changed(); dirty = false;
  notice(message);
}

function requestReplace(action) {
  if (!dirty) return action();
  replaceAction = action;
  $("#replace-dialog").showModal();
}

$("#cancel-replace").onclick = () => $("#replace-dialog").close();
$("#confirm-replace").onclick = () => { $("#replace-dialog").close(); replaceAction(); };
$("#new-config").onclick = () => requestReplace(() => {
  rememberSavePath(defaultSavePath);
  replaceConfig(defaults,"New experiment. Defaults are read from config/defaults.py without importing the simulator.");
});
$("#import-config").onclick = () => $("#import-file").click();
$("#import-file").onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const result = await api("/api/import",{yaml:await file.text()});
    if (result.errors) throw new Error(result.errors.map(e=>e.message).join(" "));
    requestReplace(() => {
      // The browser supplies a filename, not the original file's full path.
      const path = $("#save-path").value;
      const separator = Math.max(path.lastIndexOf("/"),path.lastIndexOf("\\"));
      rememberSavePath(path.slice(0,separator+1) + file.name);
      replaceConfig(result.config,`Opened ${file.name}. Check the destination folder in Save YAML before replacing a file.`);
    });
  } catch (e) { notice(e.message,true); }
  event.target.value = "";
};

document.querySelectorAll("[data-global]").forEach(input => input.addEventListener("input",() => {
  config.simulation[input.dataset.global] = input.type === "number" ? (input.value === "" ? null : Number(input.value)) : input.value;
  changed();
}));

$("#editor").addEventListener("input",event => {
  const input = event.target;
  if (!input.dataset.path || input.type === "checkbox" || input.tagName === "SELECT") return;
  set(stage(),input.dataset.path,input.type === "number" ? (input.value === "" ? null : Number(input.value)) : input.value);
  if (input.dataset.formula) {
    const preview = $(`#formula-preview-${input.dataset.formula}`);
    preview.className = "formula-preview pending";
    preview.textContent = "Checking formula…";
    clearTimeout(formulaTimers.get(input));
    formulaTimers.set(input,setTimeout(() => checkFormula(input),180));
  }
  changed();
});

$("#editor").addEventListener("change",event => {
  const input = event.target;
  if (input.dataset.formula) return checkFormula(input,true);
  if (!input.dataset.path || (input.type !== "checkbox" && input.tagName !== "SELECT")) return;
  const value = input.type === "checkbox" ? input.checked : input.value;
  set(stage(),input.dataset.path,value);
  if (input.dataset.path === "evolution.type" && value === "imaginary") {
    stage().absorber.enabled = false;
    for (const group of ["potential1", "potential2"]) {
      stage()[group].dynamic.enabled = false;
      stage()[group].dynamic.expression = "0";
    }
    stage().k_cut.enabled = false;
  }
  if (/^potential[12]\./.test(input.dataset.path) && input.dataset.path.endsWith(".enabled") && !value)
    set(stage(), input.dataset.path.replace(/enabled$/, "expression"), "0");
  renderEditor(); changed();
});

document.addEventListener("click",event => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.stage !== undefined) { selected = Number(button.dataset.stage); renderEditor(); validate(); }
  if (button.dataset.tab) { tab = button.dataset.tab; renderEditor(); validate(); }
  if (button.dataset.action) {
    const action = button.dataset.action;
    if (action === "up" || action === "down") {
      const next = selected + (action === "up" ? -1 : 1);
      [config.stages[selected],config.stages[next]] = [config.stages[next],config.stages[selected]];
      selected = next;
    } else if (action === "duplicate") {
      const copy = clone(stage()); copy.name = stageName(stage(),selected)+" copy"; copy.save.directory = "";
      config.stages.splice(selected+1,0,copy); selected++;
    } else if (action === "delete") {
      config.stages.splice(selected,1); selected = Math.min(selected,config.stages.length-1);
    }
    renderEditor(); changed();
  }
  if (button.id === "add-stage") {
    config.stages.push(clone(stageTemplate)); selected = config.stages.length-1; tab = "evolution";
    renderEditor(); changed();
  }
  if (button.dataset.error !== undefined) {
    const parts = button.dataset.error.split(".");
    if (parts[0] === "stages" && parts.length > 2) {
      selected = Number(parts[1]);
      tab = ["potential1","potential2","box","absorber","noise","k_cut"].includes(parts[2]) ? "potentials" : ["monitor","save"].includes(parts[2]) ? "outputs" : "evolution";
      renderEditor();
      const input = document.querySelector(`[data-path="${CSS.escape(parts.slice(2).join("."))}"]`);
      input?.focus();
    } else if (parts[0] === "simulation") document.querySelector(`[data-global="${CSS.escape(parts[1])}"]`)?.focus();
  }
});

$("#toggle-preview").onclick = () => {
  const visible = $("#yaml-panel").hidden;
  $("#yaml-panel").hidden = !visible;
  $("#workspace").classList.toggle("with-preview",visible);
  $("#toggle-preview").setAttribute("aria-expanded",String(visible));
  $("#toggle-preview").innerHTML = `${visible ? "Hide" : "Show"} YAML preview <span aria-hidden="true">{ }</span>`;
};
$("#copy-yaml").onclick = async () => {
  try { await navigator.clipboard.writeText(yamlText); notice("YAML copied to the clipboard."); }
  catch (_) { $("#yaml-preview").focus(); $("#yaml-preview").select(); notice("Select Copy from your browser to copy the highlighted YAML."); }
};
$("#save-config").onclick = () => {
  $("#save-error").hidden = true;
  $("#overwrite").checked = false;
  $("#save-dialog").showModal();
};
$("#save-path").addEventListener("input",event => rememberSavePath(event.target.value));
$("#write-yaml").onclick = async () => {
  const button = $("#write-yaml"); button.disabled = true;
  try {
    const result = await api("/api/save",{config,path:$("#save-path").value,overwrite:$("#overwrite").checked});
    if (result.errors) throw new Error(result.errors.map(e=>e.message).join(" "));
    rememberSavePath(result.path);
    $("#save-dialog").close(); dirty = false; notice(`Saved ${result.path}`);
  } catch (e) { $("#save-error").textContent = e.message; $("#save-error").hidden = false; }
  finally { button.disabled = false; }
};
$("#download-yaml").onclick = async () => {
  try {
    const result = await api("/api/render",{config});
    if (result.errors?.length) throw new Error(result.errors.map(e=>e.message).join(" "));
    const url = URL.createObjectURL(new Blob([result.yaml],{type:"application/yaml;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url;
    const name = $("#save-path").value.split(/[\\/]/).pop() || "experiment.yaml";
    link.download = /\.ya?ml$/i.test(name) ? name : name+".yaml";
    link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
    $("#save-dialog").close(); dirty = false; notice("YAML sent to your browser’s downloads.");
  } catch (e) { $("#save-error").textContent = e.message; $("#save-error").hidden = false; }
};

async function start() {
  try {
    const result = await api("/api/defaults"); defaults = result.config; stageTemplate = result.stage;
    defaultSavePath = result.save_path;
    $("#save-path").value = defaultSavePath;
    config = clone(defaults);
    try {
      const saved = JSON.parse(localStorage.getItem(draftKey));
      if (saved?.schema_version === 3 && saved.stages?.length) {
        const checked = await api("/api/render",{config:saved});
        if (checked.yaml) {
          const migrated = await api("/api/import",{yaml:checked.yaml});
          config = migrated.config;
          $("#save-path").value = localStorage.getItem(savePathKey) || defaultSavePath;
          dirty = true; notice("Restored your last browser draft.");
        }
      } else if (!saved && localStorage.getItem("gpe-config-editor-v1")) {
        notice("The potential format has changed. Starting a new draft; your earlier browser draft remains stored separately.");
      }
    } catch (_) { /* Start fresh if browser storage is unavailable. */ }
    renderAll(); await validate();
  } catch (e) { notice("Could not load the editor: "+e.message,true); }
}
start();
