import './digit-action.css';
const API='http://127.0.0.1:8000';
export function installDigitAction(host) {
  const $=id=>document.getElementById(id), pad=$('digit-pad'), ctx=pad.getContext('2d');
  let active=false, busy=false, sample=false, drawing=false, version=0, playback=0, pollTimer, lastTrace;
  const message=text=>$('digit-message').textContent=text;
  async function api(path,body) {
    let res;
    try { res=await fetch(API+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
    catch { throw Error('识别服务未连接，请运行项目 start.cmd'); }
    const data=await res.json(); if(!res.ok)throw Error(typeof data.detail==='string'?data.detail:'输入无效'); return data;
  }
  function setBusy(value) {busy=value; for(const id of ['digit-predict','digit-learn','digit-sample','digit-clear','digit-train'])$(id).disabled=value;}
  function reset() {
    if(active)host.clear();
    version++; clearTimeout(playback); lastTrace=null; $('digit-replay').disabled=true;
    ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);sample=false;
    $('digit-results').textContent='';$('digit-step').textContent='尚未输入';$('digit-thought').textContent='等待数字输入';
    message('写一个数字，然后让果蝇识别。');
  }
  function position(e){const r=pad.getBoundingClientRect();return [(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height]}
  pad.onpointerdown=e=>{if(busy)return;drawing=true;version++;sample=false;clearTimeout(playback);lastTrace=null;$('digit-replay').disabled=true;$('digit-results').textContent='';$('digit-thought').textContent='让我看看…';$('digit-step').textContent='黑板已更新，等待识别';host.clear();pad.setPointerCapture(e.pointerId);ctx.strokeStyle='#fff';ctx.fillStyle='#fff';ctx.lineWidth=19;ctx.lineCap=ctx.lineJoin='round';let [x,y]=position(e);ctx.beginPath();ctx.arc(x,y,9.5,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(x,y)};
  pad.onpointermove=e=>{if(drawing){ctx.lineTo(...position(e));ctx.stroke()}};
  pad.onpointerup=pad.onpointercancel=()=>drawing=false;
  function payload(){let c=document.createElement('canvas');c.width=c.height=28;let cc=c.getContext('2d');cc.drawImage(pad,0,0,28,28);let d=cc.getImageData(0,0,28,28).data;return {pixels:Array.from({length:784},(_,i)=>d[4*i]/255),normalize:!sample,recurrent:true}}
  function replay(trace) {
    clearTimeout(playback);let i=0;
    function next(){if(!active)return;try{host.frame(trace,i);$('digit-step').textContent=`第 ${i}/${trace.steps} 步 · 计算状态慢放`;}catch(e){message(e.message);return}if(++i<trace.frames.length)playback=setTimeout(next,450)}next();
  }
  function show(result) {
    lastTrace=result.trace; replay(lastTrace);$('digit-replay').disabled=false;
    $('digit-thought').replaceChildren();
    const answer=document.createElement('strong'),detail=document.createElement('small');
    answer.textContent=`这是 ${result.digit}！`;
    detail.textContent=`分类分数 ${(Math.max(...result.probabilities)*100).toFixed(1)}%`;
    $('digit-thought').append(answer,detail);
    $('digit-results').innerHTML=result.probabilities.map((p,i)=>`<div><b>${i}</b><span><i style="width:${p*100}%"></i></span><small>${(p*100).toFixed(1)}%</small></div>`).join('');
    message(`果蝇识别完成 · ${result.elapsed_ms} ms · ${result.model_version==='personal-readout.pt'?'已学习你的笔迹':'MNIST 初始模型'}`);
  }
  async function infer() {if(busy)return;let token=++version;setBusy(true);message('图像输入 → 果蝇连接组传播 → 分类读出…');$('digit-thought').textContent='正在识别…';try{let result=await api('/api/predict',payload());if(active&&version===token)show(result)}catch(e){if(active){message(e.message);$('digit-thought').textContent='等待有效输入'}}finally{setBusy(false)}}
  async function poll(){clearTimeout(pollTimer);if(!active)return;try{const s=await api('/api/status');const progress=s.training_active?`训练中 · ${({loading:'加载数据',extracting:'连接组特征提取',training:'读出层训练',complete:'训练完成'})[s.stage]||s.stage}${s.epoch?' / 第 '+s.epoch+' 轮':''}${s.total?' / '+s.done+' / '+s.total:''}`:s.stage==='failed'?s.error:`已就绪 · 学习记录 ${s.lessons} 条 · ${s.model_version==='personal-readout.pt'?'个性化模型（未重测）':'初始模型测试准确率 '+(s.base_test_accuracy==null?'见训练报告':(s.base_test_accuracy*100).toFixed(2)+'%')}`;$('digit-training').textContent=progress;$('digit-train').disabled=busy||s.training_active;$('digit-learn').disabled=busy||s.training_active;}catch(e){$('digit-training').textContent=e.message}pollTimer=setTimeout(poll,2500)}
  function close(){if(!active)return;active=false;version++;clearTimeout(playback);clearTimeout(pollTimer);$('digit-panel').hidden=$('digit-thought').hidden=true;$('digit-action').setAttribute('aria-pressed','false');document.body.classList.remove('digit-mode');host.leave()}
  $('digit-action').onclick=()=>{if(active)return;try{host.enter();active=true;$('digit-panel').hidden=$('digit-thought').hidden=false;$('digit-action').setAttribute('aria-pressed','true');document.body.classList.add('digit-mode');poll()}catch(e){alert(e.message)}};
  $('digit-close').onclick=close;$('digit-clear').onclick=reset;$('digit-predict').onclick=infer;
  $('digit-replay').onclick=()=>lastTrace&&replay(lastTrace);
  $('digit-sample').onclick=async()=>{if(busy)return;setBusy(true);try{let s=await api('/api/sample/'+Math.floor(Math.random()*10000));if(!active)return;reset();let c=document.createElement('canvas');c.width=c.height=28;let cc=c.getContext('2d'),im=cc.createImageData(28,28);s.pixels.forEach((v,i)=>{im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v*255;im.data[4*i+3]=255});cc.putImageData(im,0,0);ctx.drawImage(c,0,0,280,280);sample=true;$('digit-label').value=s.label;message(`独立测试集示例 #${s.index} · 标签 ${s.label}；示例用于识别，不参与个人教学。`)}catch(e){message(e.message)}finally{setBusy(false)}};
  $('digit-learn').onclick=async()=>{if(busy)return;if(sample){message('请手写一个数字进行教学，避免把独立测试集用于训练。');return}setBusy(true);let token=version;try{message('正在用你的正确标签训练读出层，并回放训练样本以减少遗忘…');let input=payload();let lesson=await api('/api/learn',{...input,label:Number($('digit-label').value)});if(!active||token!==version)return;let result=await api('/api/predict',input);show(result);message(`已学习数字 ${lesson.label} · 当前样本分数 ${(lesson.before*100).toFixed(1)}% → ${(lesson.after*100).toFixed(1)}%（不是测试准确率）`);poll()}catch(e){message(e.message)}finally{setBusy(false)}};
  $('digit-train').onclick=async()=>{if(busy)return;setBusy(true);try{await api('/api/train',{});message('MNIST 训练已启动；旧权重已备份。完成后自动加载新模型。');poll()}catch(e){message(e.message)}finally{setBusy(false)}};
  reset();return {close};
}
