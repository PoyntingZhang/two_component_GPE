"use strict";

let workflowState = null, monitorData = null, selectedView = "experiment", activePrompt = null;
let statePending = false, monitorPending = false, lastRefresh = 0, wasBusy = false, lastFolder = null;
let lastOperationError = "";
const profileFields = ["run-target", "run-output", "run-cupy", "run-python", "run-modules", "run-account", "run-partition", "run-walltime", "ssh-host", "ssh-user", "ssh-port", "remote-parent"];
const profileKey = "two-component-gpe-ssh-profile-v1";

function workflowNotice(message, error=false) {
  $("#workflow-notice").textContent = message;
  $("#workflow-notice").className = "notice" + (error ? " error" : "");
  $("#workflow-notice").hidden = false;
}

function showView(name) {
  selectedView = name;
  document.querySelectorAll(".workflow-view").forEach(view => view.hidden = view.id !== "view-" + name);
  document.querySelectorAll("[data-view]").forEach(button => {
    button.classList.toggle("active", button.dataset.view === name);
    if (button.dataset.view === name) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (name === "launch" && config) $("#prepare-summary").textContent = `${config.stages.length} stages · ${config.simulation.N_grids}³ grid · ${config.stages.reduce((n,s)=>n+s.evolution.steps,0).toLocaleString()} steps. Prepare copies this draft; later edits need a new run.`;
  if (name === "monitor") readMonitor();
}
document.querySelectorAll("[data-view]").forEach(button => button.onclick = () => showView(button.dataset.view));
$("#go-monitor").onclick = () => showView("monitor");

function targetChanged() {
  const local = $("#run-target").value === "local";
  $("#connection-card").hidden = local;
  document.querySelectorAll(".hpc-setting").forEach(node => node.hidden = local);
  $("#target-hint").textContent = local
    ? workflowState?.local_execution_error || "Setup and simulation run on the machine hosting this GUI. To use Colab's GPU, open the GUI inside the Colab GPU runtime."
    : "Prepare the run on this machine, then connect and upload it to the HPC.";
  const preparedLocal = (workflowState?.run?.target || $("#run-target").value) === "local";
  $("#environment-hint").textContent = preparedLocal
    ? workflowState?.local_execution_error || "Set up environment installs dependencies on the machine hosting this GUI. On Colab, you can skip setup if its existing environment is ready."
    : "Set up environment installs this prepared run's dependencies on the connected HPC after upload. Skip it when an appropriate environment already exists.";
}
try {
  const profile = JSON.parse(localStorage.getItem(profileKey) || "{}");
  profileFields.forEach(id => { if (profile[id] !== undefined) $("#"+id).value = profile[id]; });
} catch (_) { /* Profiles are optional; credentials never go here. */ }
profileFields.forEach(id => $("#"+id).addEventListener("change", () => {
  try { localStorage.setItem(profileKey, JSON.stringify(Object.fromEntries(profileFields.map(key => [key,$("#"+key).value])))); } catch (_) {}
  targetChanged();
}));
targetChanged();

async function action(path, body={}) {
  try {
    $("#workflow-notice").hidden = true;
    const result = await api(path, body);
    if (result.errors) throw new Error(result.errors.map(item=>item.message).join(" "));
    await pollState();
  } catch (error) { workflowNotice(error.message, true); }
}
$("#prepare-run").onclick = async () => {
  if (!config) return;
  const checked = await api("/api/render", {config});
  if (checked.errors?.length) { workflowNotice("Resolve the Experiment validation issues before preparing a run.", true); showView("experiment"); return; }
  await action("/api/run/prepare", {config:clone(config), name:$("#run-name").value.trim(), notes:$("#run-notes").value,
    output:$("#run-output").value.trim(), cupy_package:$("#run-cupy").value,
    launch:{target:$("#run-target").value, account:$("#run-account").value, partition:$("#run-partition").value,
      walltime:$("#run-walltime").value, modules:$("#run-modules").value, python:$("#run-python").value}});
};
$("#open-run").onclick = () => action("/api/run/open", {folder:$("#open-run-path").value.trim()});
$("#ssh-connect").onclick = () => action("/api/ssh/connect", {host:$("#ssh-host").value.trim(), username:$("#ssh-user").value.trim(), port:Number($("#ssh-port").value)});
$("#ssh-disconnect").onclick = () => action("/api/ssh/disconnect");
$("#ssh-prompt").onsubmit = async event => {
  event.preventDefault();
  if (!activePrompt) return;
  const answer = $("#ssh-answer").value, id = activePrompt;
  $("#ssh-answer").value = "";
  $("#ssh-prompt").hidden = true;
  await action("/api/ssh/answer", {id, answer});
};
$("#upload-run").onclick = () => action("/api/run/upload", {parent:$("#remote-parent").value.trim()});
$("#setup-run").onclick = () => action("/api/run/setup");
$("#submit-run").onclick = () => action("/api/run/submit");
$("#cancel-run").onclick = () => {
  if (confirm(`Cancel ${workflowState?.run?.name || "this run"}? This stops its active job.`)) action("/api/run/cancel");
};
$("#download-run").onclick = async () => {
  try {
    const response = await fetch("api/run/download", {headers:{"X-GPE-Session":window.GPE_SESSION}});
    if (!response.ok) throw new Error((await response.json()).error);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href=url; link.download=workflowState.run.name+".tar.gz"; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  } catch (error) { workflowNotice(error.message,true); }
};

async function pollState() {
  if (statePending) return;
  statePending = true;
  try {
    workflowState = await api("/api/workflow");
    const {operation:op, connection:connection, run, folder} = workflowState;
    $("#job-badge").textContent = run.name ? `${run.name} · ${run.status}` : "No run prepared";
    $("#launch-status").textContent = run.name ? `${run.status}${run.job_id ? " · Job " + run.job_id : ""}` : "Prepare a run to begin.";
    $("#prepared-details").textContent = folder ? `Prepared folder: ${folder}\n${run.remote_path ? "Remote folder: " + run.remote_path : ""}` : "";
    $("#operation-name").textContent = (op.busy ? "Working: " : "") + (op.name || "Activity");
    $("#operation-output").textContent = op.output || (op.busy ? "Working…" : "No operation yet.");
    $("#operation-error").textContent = op.error || "";
    if (op.error && op.error !== lastOperationError) workflowNotice(op.error, true);
    lastOperationError = op.error || "";
    $("#connection-status").textContent = connection.connected ? `Connected · ${connection.target}` : connection.connecting ? "Waiting for SSH…" : "Disconnected";
    $("#ssh-output").textContent = connection.output;
    $("#ssh-output").hidden = !connection.output;
    if (connection.prompt && connection.prompt.id !== activePrompt) {
      activePrompt = connection.prompt.id;
      $("#ssh-prompt-text").textContent = connection.prompt.text;
      $("#ssh-prompt").hidden = false;
      $("#ssh-answer").value = "";
      $("#ssh-answer").focus();
    } else if (!connection.prompt) { activePrompt = null; $("#ssh-prompt").hidden = true; }
    ["prepare-run", "open-run", "ssh-connect", "upload-run", "setup-run", "submit-run", "refresh-monitor", "cancel-run"].forEach(id => $("#"+id).disabled = op.busy);
    $("#ssh-connect").disabled ||= connection.connecting || connection.connected;
    $("#download-run").disabled = !folder || op.busy;
    $("#upload-run").disabled ||= !folder || !connection.connected || run.target !== "slurm" || !!run.remote_path;
    const localBlocked = run.target === "local" && !!workflowState.local_execution_error;
    $("#setup-run").disabled ||= !folder || !!run.job_id || !!run.pid || localBlocked || (run.target === "slurm" && (!connection.connected || !run.remote_path));
    $("#submit-run").disabled ||= !folder || !!run.job_id || !!run.pid || localBlocked || (run.target === "slurm" && (!connection.connected || !run.remote_path));
    $("#submit-run").textContent = run.target === "slurm" ? "Submit GPU job" : "Launch simulation";
    $("#cancel-run").disabled ||= !(run.job_id || run.pid) || /completed|failed|cancelled|stopped/i.test(run.status);
    if (folder && folder !== lastFolder) {
      lastFolder = folder;
      $("#run-target").value = run.target;
      $("#monitor-folder").value = "";
      $("#monitor-stage").value = "";
      resetMonitorTimes();
      workflowNotice(`Opened ${run.name}. Launch uses this prepared configuration.`);
      if (selectedView === "monitor") readMonitor();
    }
    targetChanged();
    if (wasBusy && !op.busy && selectedView === "monitor") readMonitor();
    wasBusy = op.busy;
  } catch (error) { $("#job-badge").textContent = "Workspace connection unavailable"; }
  finally { statePending=false; }
}


pollState();
setInterval(async()=>{
  await pollState();
  if(selectedView==="monitor" && $("#auto-refresh").checked && !document.hidden && !workflowState?.operation.busy && Date.now()-lastRefresh>=Math.max(5,Number($("#refresh-seconds").value)||15)*1000) refreshMonitor();
},1800);
