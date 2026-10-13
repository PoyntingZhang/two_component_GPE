"use strict";
let workflowState=null,monitorData=null,selectedView="experiment",monitorPending=false,lastRefresh=0;
let statePending=false,lastFolder=null,wasBusy=false,lastOperationError="";
function workflowNotice(message,error=false){$("#workflow-notice").textContent=message;$("#workflow-notice").className="notice"+(error?" error":"");$("#workflow-notice").hidden=false;}
function showView(name){
 selectedView=name;
 document.querySelectorAll(".workflow-view").forEach(v=>v.hidden=v.id!=="view-"+name);
 document.querySelectorAll("[data-view]").forEach(b=>{b.classList.toggle("active",b.dataset.view===name);if(b.dataset.view===name)b.setAttribute("aria-current","page");else b.removeAttribute("aria-current");});
 if(name==="launch"&&config)$("#prepare-summary").textContent=`${config.stages.length} stages · ${config.simulation.N_grids}³ grid. Prepare captures this draft in a new Drive folder.`;
 if(name==="monitor")readMonitor();
}
document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>showView(b.dataset.view));
$("#go-monitor").onclick=()=>showView("monitor");
async function action(path,body={}){try{$("#workflow-notice").hidden=true;const result=await api(path,body);if(result.errors)throw Error(result.errors.map(e=>e.message).join(" "));await pollState();}catch(e){workflowNotice(e.message,true);}}
$("#prepare-run").onclick=async()=>{
 try{
  if(!config)return;
  const checked=await api("/api/render",{config});
  if(checked.errors?.length){workflowNotice("Resolve the Experiment validation issues first.",true);showView("experiment");return;}
  await action("/api/run/prepare",{config:clone(config),name:$("#run-name").value.trim(),notes:$("#run-notes").value,output:$("#run-output").value.trim(),cupy_package:$("#run-cupy").value});
 }catch(e){workflowNotice(e.message,true);}
};
$("#open-run").onclick=()=>action("/api/run/open",{folder:$("#open-run-path").value.trim()});
$("#setup-run").onclick=()=>action("/api/run/setup");
$("#submit-run").onclick=()=>action("/api/run/submit");
$("#cancel-run").onclick=()=>{if(confirm(`Stop ${workflowState?.run?.name||"this simulation"}?`))action("/api/run/cancel");};
$("#download-run").onclick=async()=>{try{const response=await fetch("api/run/download",{headers:{"X-GPE-Session":window.GPE_SESSION}});if(!response.ok)throw Error((await response.json()).error);const url=URL.createObjectURL(await response.blob());const link=document.createElement("a");link.href=url;link.download=workflowState.run.name+".tar.gz";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){workflowNotice(e.message,true);}};
async function pollState(){
 if(statePending)return;statePending=true;
 try{
  workflowState=await api("/api/workflow");const {operation:op,run,folder}=workflowState;
  if(!$("#run-output").value)$("#run-output").value=workflowState.output_parent;
  $("#storage-status").textContent=workflowState.storage_error||"Output: <launch name>/config/run.yaml, notes.txt, data/, monitor/, logs/.";
  $("#job-badge").textContent=run.name?`${run.name} · ${run.status}`:"No run prepared";
  $("#launch-status").textContent=run.status||"Prepare a run to begin.";
  $("#prepared-details").textContent=folder?`Drive launch folder: ${folder}`:"";
  $("#operation-name").textContent=(op.busy?"Working: ":"")+(op.name||"Activity");
  $("#operation-output").textContent=op.output||(op.busy?"Working…":"No operation yet.");
  $("#operation-error").textContent=op.error||"";
  if(op.error&&op.error!==lastOperationError)workflowNotice(op.error,true);lastOperationError=op.error||"";
  ["prepare-run","open-run","setup-run","submit-run","refresh-monitor","cancel-run"].forEach(id=>$("#"+id).disabled=op.busy);
  $("#setup-run").disabled||=!folder||!!run.pid;
  $("#submit-run").disabled||=!folder||!!run.pid||!!workflowState.storage_error||/unknown|Launch failed/.test(run.status||"");
  $("#cancel-run").disabled||=!run.pid||run.status!=="Running";
  $("#download-run").disabled=!folder||op.busy;
  if(folder&&folder!==lastFolder){lastFolder=folder;$("#monitor-folder").value="";$("#monitor-stage").value="";resetMonitorTimes();workflowNotice(`Opened ${run.name}. Saved data is read directly from Drive.`);if(selectedView==="monitor")readMonitor();}
  if(wasBusy&&!op.busy&&selectedView==="monitor")readMonitor();wasBusy=op.busy;
 }catch(e){$("#job-badge").textContent="Notebook connection unavailable";}finally{statePending=false;}
}
pollState();
setInterval(async()=>{await pollState();if(selectedView==="monitor"&&$("#auto-refresh").checked&&!document.hidden&&!workflowState?.operation.busy&&Date.now()-lastRefresh>=Math.max(5,Number($("#refresh-seconds").value)||15)*1000)refreshMonitor();},1800);
