// Independent saved-time selections. null means follow the newest readable sample.
const monitorTimes = {spectrum:null, radial:null, image:null, scalar:null};
const plotStates = Object.fromEntries(Object.keys(monitorTimes).map(key=>[key, {wanted:null, shown:null, busy:false}]));
const plotCache = new Map();
let monitorAgain = false, monitorVersion = 0, monitorTimer;

function resetMonitorTimes() {
  Object.keys(monitorTimes).forEach(key=>monitorTimes[key]=null);
  monitorVersion++;
}
function options(select, values, first, selectedValue) {
  select.replaceChildren();
  if (first !== null) select.add(new Option(first,""));
  values.forEach(value=>select.add(new Option(value,value)));
  select.value = values.includes(selectedValue) ? selectedValue : first !== null ? "" : values[0] || "";
}
function monitorSamples(key) {
  if (!monitorData) return [];
  return key === "scalar" ? monitorData.scalars.map(row=>row.step) :
    key === "image" ? monitorData.image_samples : monitorData.spectra;
}
function timeControl(key, sample) {
  const samples = monitorSamples(key), slider = $("#"+key+"-time");
  const selected = monitorTimes[key] ?? (key === "scalar" ? sample?.step : sample?.file);
  const index = samples.indexOf(selected);
  const shownIndex = samples.indexOf(key === "scalar" ? sample?.step : sample?.file);
  slider.max = Math.max(0,samples.length-1);
  slider.value = Math.max(0,index);
  slider.disabled = !samples.length;
  $("#"+key+"-latest").disabled = monitorTimes[key] === null;
  $("#"+key+"-time-label").textContent = sample ?
    `t = ${fmt(sample.time)} · ${shownIndex+1}/${samples.length}${monitorTimes[key] === null ? " · latest" : ""}` : "No readable sample";
}
function optionalNumber(id) {
  const input = $("#"+id);
  if (!input.value.trim()) return null;
  return Number(input.value);
}

async function readMonitor() {
  if (monitorPending) {monitorAgain=true;return;}
  if (!workflowState?.folder && !$("#monitor-folder").value.trim()) return;
  monitorPending = true;
  const version = monitorVersion;
  try {
    const stageValue = $("#monitor-stage").value;
    const data = await api("/api/monitor", {folder:$("#monitor-folder").value.trim(), stage:stageValue,
      spectrum:monitorTimes.spectrum || "", radial:monitorTimes.radial || "",
      image:$("#monitor-image").value, image_sample:monitorTimes.image || "",
      image_limit:Number($("#image-limit").value)});
    if (version !== monitorVersion) {monitorAgain=true;return;}
    if (!stageValue && monitorData?.stage && data.stage!==monitorData.stage) {
      resetMonitorTimes();monitorData=null;monitorAgain=true;return;
    }
    monitorData = data;
    options($("#monitor-stage"),data.stages,"Latest active stage",stageValue);
    options($("#monitor-image"),data.images,null,data.image?.kind || $("#monitor-image").value);
    $("#monitor-message").textContent = `${data.stage ? "Showing " + data.stage + ". " : ""}${data.warnings.join(" ")}`;
    $("#simulation-log").textContent = data.log || "No log available yet.";
    drawMonitor();
  } catch (error) { $("#monitor-message").textContent = error.message; }
  finally {
    monitorPending=false;
    if (monitorAgain) {monitorAgain=false;readMonitor();}
  }
}

// Throttle rather than wait for dragging to stop. One read runs at a time.
function queueMonitorRead() {
  if (!monitorTimer) monitorTimer=setTimeout(()=>{monitorTimer=null;readMonitor();},60);
}
async function refreshMonitor() {
  lastRefresh=Date.now();
  if (workflowState?.folder && !$("#monitor-folder").value.trim()) {
    if (!workflowState.operation.busy) await action("/api/run/refresh");
  }
  await readMonitor();
}

async function showPlot(key, data, options={}) {
  const version=monitorVersion;
  const state=plotStates[key], request=JSON.stringify({kind:key,data,options,version});
  state.wanted=request;
  if (state.busy) return;
  if (state.shown===request) {plotDetails(key,data,options);return;}
  state.busy=true;
  const element=$("#"+key+"-plot");
  element.setAttribute("aria-busy","true");
  try {
    let svg=plotCache.get(request);
    if (!svg) {
      svg=(await api("/api/monitor/plot",{kind:key,data,options})).svg;
      plotCache.set(request,svg);
      if (plotCache.size>24) plotCache.delete(plotCache.keys().next().value);
    }
    // Show completed intermediate frames during a drag, but never from an old stage.
    if (version===monitorVersion) {
      element.innerHTML=svg;
      state.shown=request;
      element.querySelector("svg").setAttribute("aria-hidden","true");
      plotDetails(key,data,options);
    }
  } catch(error) {
    if(state.wanted===request){element.textContent=error.message;state.shown=null;}
  } finally {
    state.busy=false;
    element.setAttribute("aria-busy","false");
    if(state.wanted!==request){const next=JSON.parse(state.wanted);showPlot(key,next.data,next.options);}
  }
}
function plotDetails(key,data,options) {
  const sample=key==="scalar" ? data[options.index] : data;
  timeControl(key,sample);
  let caption="No saved data yet.";
  if(sample) {
    if(key==="scalar") caption=`Selected step ${sample.step} · summary above follows this slider. Full history stays visible.`;
    else if(key==="image") caption=`${sample.kind} · t = ${fmt(sample.time)} · step ${sample.step} · ${sample.kind.startsWith("phase") ? "original phase colours" : "BEC colours"} · display sampled to ≤256 pixels per axis`;
    else caption=`t = ${fmt(sample.time)} · step ${sample.step} · grey: incomplete shells`;
  }
  $("#"+key+"-caption").textContent=caption;
  if(key==="scalar") scalarSummary(sample);
}
function scalarSummary(row) {
  if (!row) {$("#scalar-cards").replaceChildren();return;}
  const energy=Number($("#energy-unit").value);
  const meanL=row.norm>0 && row.Lz!==undefined ? row.Lz/row.norm : null;
  const number=value=>Number.isFinite(value) && value!==0 && (Math.abs(value)<.0001 || Math.abs(value)>=1e6) ? value.toExponential(4) : fmt(value);
  const cards=[
    ["Stage time",fmt(row.time),"dimensionless"],
    ["Norm",number(row.norm),"saved ∫|ψ|² d³x"],
    ["Kinetic energy",number(row.E_kin),Number.isFinite(row.E_kin) && energy>0 ? `[${number(row.E_kin*energy)} nK] · dimensionless above` : "dimensionless"],
    ["Mean angular momentum / particle",meanL===null ? "—" : number(meanL)+" ℏ","saved Lz / norm"]
  ];
  $("#scalar-cards").innerHTML=cards.map(([label,value,note])=>`<div class="scalar-card"><small>${escapeHTML(label)}</small><strong>${escapeHTML(value)}</strong><small>${escapeHTML(note)}</small></div>`).join("");
}
function drawMonitor() {
  if(!monitorData)return;
  const max=optionalNumber("k-max");
  for(const key of ["spectrum","radial"]) {
    const sample=monitorData[key];
    showPlot(key,sample,{k_max:max});
  }
  const image=monitorData.image,phase=image?.kind.startsWith("phase");
  $("#image-log").disabled=!!phase;
  showPlot("image",image,{log:$("#image-log").checked,vmin:optionalNumber("image-vmin"),vmax:optionalNumber("image-vmax"),image_limit:Number($("#image-limit").value)});
  const rows=monitorData.scalars;
  const index=monitorTimes.scalar===null ? rows.length-1 : rows.findIndex(row=>row.step===monitorTimes.scalar);
  // All rows remain on the slider. Only the drawn history line is thinned.
  const stride=Math.max(1,Math.ceil(rows.length/4000));
  const indices=[...new Set([...rows.keys()].filter(i=>i%stride===0).concat(index,rows.length-1))].filter(i=>i>=0).sort((a,b)=>a-b);
  showPlot("scalar",indices.map(i=>rows[i]),{quantity:$("#scalar-quantity").value,index:Math.max(0,indices.indexOf(index))});
}

$("#refresh-monitor").onclick=refreshMonitor;
$("#open-monitor").onclick=()=>{$("#monitor-stage").value="";resetMonitorTimes();readMonitor();};
$("#monitor-stage").onchange=()=>{resetMonitorTimes();readMonitor();};
$("#monitor-image").onchange=()=>{monitorTimes.image=null;monitorVersion++;queueMonitorRead();};
$("#image-limit").onchange=queueMonitorRead;
["image-log","image-vmin","image-vmax","k-max","scalar-quantity","energy-unit"].forEach(id=>$("#"+id).onchange=drawMonitor);
for(const key of Object.keys(monitorTimes)) {
  $("#"+key+"-time").oninput=event=>{
    const samples=monitorSamples(key), index=Math.round(Number(event.target.value));
    monitorTimes[key]=samples[index] ?? null;
    if(key==="scalar") drawMonitor(); else queueMonitorRead();
  };
  $("#"+key+"-latest").onclick=()=>{
    monitorTimes[key]=null;
    if(key==="scalar") drawMonitor(); else queueMonitorRead();
  };
}
