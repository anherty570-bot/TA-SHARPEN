import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.webgpu.min.mjs";
import { Muxer, ArrayBufferTarget } from "https://cdn.jsdelivr.net/npm/mp4-muxer@5/build/mp4-muxer.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const $ = s => document.querySelector(s), vid = $("#v"), sleep = ms => new Promise(r => setTimeout(r, ms));
const MODELS = { fast: "realesr-general-x4v3", anime: "realesr-animevideov3" };
let file = null, V = null, cancelFlag = false, useWasm = false, busy = false;
const sessions = {}, cache = new Map();
const fmt = t => { const m = Math.floor(t / 60); return String(m).padStart(2, "0") + ":" + (t - m * 60).toFixed(3).padStart(6, "0"); };
const parse = s => { s = s.trim(); if (s.includes(":")) { const [m, x] = s.split(":"); return +m * 60 + +x; } return +s; };
const err = m => { $("#err").hidden = !m; $("#err").textContent = m || ""; };
const warn = m => { $("#warn").hidden = !m; $("#warn").textContent = m || ""; };
const CANC = "CANCELLED", chk = () => { if (cancelFlag) throw new Error(CANC); };

// ---------- AI (ONNX Runtime Web, Real-ESRGAN compact nets) ----------
async function getSession(model) {
  const key = model + useWasm;
  if (!sessions[key]) sessions[key] = await ort.InferenceSession.create(`models/${MODELS[model]}.onnx`,
    { executionProviders: navigator.gpu && !useWasm ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "all" });
  return sessions[key];
}
async function runNet(model, tensor) {
  try { return (await (await getSession(model)).run({ input: tensor })).output; }
  catch (e) {
    if (navigator.gpu && !useWasm) { useWasm = true; warn("WebGPU lỗi → chuyển sang WASM (chậm hơn nhiều)."); return runNet(model, tensor); }
    throw new Error("Model AI lỗi: " + (e.message || e));
  }
}
// Tiled x4 inference; scale 2 = network output (x4) downsampled with high-quality filter.
async function sr(src, scale, model, onTile) {
  const w = src.width, h = src.height, px = src.getContext("2d").getImageData(0, 0, w, h).data;
  const W4 = w * 4, out = new ImageData(W4, h * 4), T = navigator.gpu && !useWasm ? 192 : 96, P = 8;
  const nT = Math.ceil(w / T) * Math.ceil(h / T); let k = 0;
  for (let y0 = 0; y0 < h; y0 += T) for (let x0 = 0; x0 < w; x0 += T) {
    chk(); const x1 = Math.min(x0 + T, w), y1 = Math.min(y0 + T, h);
    const sx = Math.max(x0 - P, 0), sy = Math.max(y0 - P, 0), ex = Math.min(x1 + P, w), ey = Math.min(y1 + P, h), tw = ex - sx, th = ey - sy;
    const inp = new Float32Array(3 * tw * th), pl = tw * th;
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      const i = ((sy + y) * w + sx + x) * 4, j = y * tw + x;
      inp[j] = px[i] / 255; inp[pl + j] = px[i + 1] / 255; inp[2 * pl + j] = px[i + 2] / 255;
    }
    const o = (await runNet(model, new ort.Tensor("float32", inp, [1, 3, th, tw]))).data, ow = tw * 4, op = ow * th * 4;
    for (let oy = (y0 - sy) * 4; oy < (y1 - sy) * 4; oy++) for (let ox = (x0 - sx) * 4; ox < (x1 - sx) * 4; ox++) {
      const d = ((sy * 4 + oy) * W4 + sx * 4 + ox) * 4, s = oy * ow + ox;
      out.data[d] = o[s] * 255; out.data[d + 1] = o[op + s] * 255; out.data[d + 2] = o[2 * op + s] * 255; out.data[d + 3] = 255;
    }
    onTile && onTile(++k / nT);
  }
  const big = Object.assign(document.createElement("canvas"), { width: W4, height: h * 4 }); big.getContext("2d").putImageData(out, 0, 0);
  if (scale === 4) return big;
  const c = Object.assign(document.createElement("canvas"), { width: w * 2, height: h * 2 }), x = c.getContext("2d");
  x.imageSmoothingQuality = "high"; x.drawImage(big, 0, 0, c.width, c.height); return c;
}

// ---------- video loading / frame access ----------
const seek = (t) => new Promise((res, rej) => { vid.onseeked = () => res(); vid.onerror = () => rej(new Error("Không đọc được video")); vid.currentTime = t; });
const snap = v => [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120].reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a);
function estFps() { // measured from real decoded frame timestamps
  return new Promise(res => {
    if (!vid.requestVideoFrameCallback) return res(30);
    const ts = [], done = () => { vid.pause(); const d = ts.slice(1).map((t, i) => t - ts[i]).filter(x => x > 0); res(d.length ? snap(1 / Math.min(...d)) : 30); };
    const cb = (_, m) => { ts.push(m.mediaTime); ts.length >= 10 ? done() : vid.requestVideoFrameCallback(cb); };
    vid.requestVideoFrameCallback(cb); vid.play().catch(() => res(30)); setTimeout(() => ts.length < 10 && done(), 2500);
  });
}
$("#file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return; err(); warn(); file = f; cache.clear();
  vid.src = URL.createObjectURL(f);
  try { await new Promise((r, j) => { vid.onloadedmetadata = r; vid.onerror = () => j(new Error("Trình duyệt không đọc được video này (codec không hỗ trợ)")); }); }
  catch (x) { return err(x.message); }
  const fps = await estFps(); await seek(0);
  V = { w: vid.videoWidth, h: vid.videoHeight, dur: vid.duration, fps };
  $("#meta").textContent = `${f.name}\n${(f.size / 1048576).toFixed(1)} MB · ${V.w} × ${V.h} · ~${fps} FPS (ước lượng) · ${fmt(V.dur)}`;
  $("#tl").max = V.dur; $("#s-vid").hidden = false;
  if (!navigator.gpu) warn("Thiết bị này không có WebGPU: AI sẽ chạy bằng CPU (WASM), rất chậm. Nên dùng Chrome/Edge mới trên máy tính hoặc Android đời mới.");
};
const go = t => { t = Math.min(Math.max(t, 0), V.dur); $("#tl").value = t; $("#ts").value = fmt(t); return seek(t); };
$("#tl").oninput = e => go(+e.target.value);
vid.ontimeupdate = () => { if (!vid.seeking) { $("#tl").value = vid.currentTime; $("#ts").value = fmt(vid.currentTime); } };
$("#ts").onchange = e => go(parse(e.target.value) || 0);
$("#prev").onclick = () => { vid.pause(); go(vid.currentTime - 1 / V.fps); };
$("#next").onclick = () => { vid.pause(); go(vid.currentTime + 1 / V.fps); };
const mk = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
const url = c => new Promise(r => c.toBlob(b => r(URL.createObjectURL(b)), "image/png"));
const lock = on => { busy = on; $("#go").disabled = $("#full").disabled = on; };

// ---------- frame preview ----------
$("#go").onclick = async () => {
  if (busy) return; err(); cancelFlag = false; lock(true); $("#pp").hidden = false; $("#cancel").hidden = false;
  try {
    vid.pause(); const t = parse($("#ts").value), scale = +$("#scale").value, model = $("#model").value, key = [t.toFixed(3), scale, model].join("|");
    if (!cache.has(key)) {
      await seek(t); const c = mk(V.w, V.h); c.getContext("2d").drawImage(vid, 0, 0);
      const t0 = performance.now(), a = await sr(c, scale, model, p => { $("#pp").value = 100 * p; $("#pt").textContent = `Đang xử lý ${(100 * p).toFixed(0)}%`; });
      cache.set(key, [await url(c), await url(a), (performance.now() - t0) / 1000]);
    }
    const [b, a, sec] = cache.get(key); $("#ib").src = b; $("#ia").src = a; $("#s-cmp").hidden = false;
    $("#pt").textContent = `Xong trong ${sec.toFixed(1)} s (kết quả được cache theo timestamp/scale/model)`; $("#cmp").scrollIntoView();
  } catch (x) { x.message === CANC ? $("#pt").textContent = "Đã hủy" : (err(x.message), $("#pt").textContent = ""); }
  lock(false); $("#pp").hidden = true; $("#cancel").hidden = true;
};
// compare / zoom / pan
let z = 1, px = 0, py = 0; const cmp = $("#cmp");
const apply = () => { const r = cmp.getBoundingClientRect(); px = Math.min(0, Math.max(px, r.width * (1 - z))); py = Math.min(0, Math.max(py, r.height * (1 - z))); $("#zw").style.transform = `translate(${px}px,${py}px) scale(${z})`; };
$("#sl").oninput = e => $("#ia").style.clipPath = `inset(0 0 0 ${e.target.value}%)`;
$("#zm").oninput = e => { z = +e.target.value; $("#zv").textContent = z.toFixed(1) + "×"; apply(); };
let drag = null;
cmp.onpointerdown = e => { drag = [e.clientX - px, e.clientY - py]; cmp.setPointerCapture(e.pointerId); };
cmp.onpointermove = e => { if (drag && z > 1) { px = e.clientX - drag[0]; py = e.clientY - drag[1]; apply(); } };
cmp.onpointerup = () => drag = null;
$("#fs").onclick = () => document.fullscreenElement ? document.exitFullscreen() : (cmp.requestFullscreen ? cmp.requestFullscreen().catch(() => cmp.classList.toggle("fs")) : cmp.classList.toggle("fs"));

// ---------- full video: seek frame -> AI -> WebCodecs H.264 -> mp4-muxer (+AAC audio) ----------
async function prepAudio() {
  if (!window.AudioEncoder) return null;
  try {
    const ac = new AudioContext(), buf = await ac.decodeAudioData(await file.arrayBuffer()); ac.close();
    const ch = Math.min(buf.numberOfChannels, 2), cfg = { codec: "mp4a.40.2", sampleRate: buf.sampleRate, numberOfChannels: ch, bitrate: 160000 };
    return (await AudioEncoder.isConfigSupported(cfg)).supported ? { buf, ch, cfg } : null;
  } catch { return null; }
}
async function encodeAudio(a, muxer) {
  const ae = new AudioEncoder({ output: (c, m) => muxer.addAudioChunk(c, m), error: e => { throw e; } }); ae.configure(a.cfg);
  const { buf, ch } = a, sr = buf.sampleRate;
  for (let i = 0; i < buf.length; i += 4096) {
    const n = Math.min(4096, buf.length - i), d = new Float32Array(n * ch);
    for (let c = 0; c < ch; c++) d.set(buf.getChannelData(c).subarray(i, i + n), c * n);
    ae.encode(new AudioData({ format: "f32-planar", sampleRate: sr, numberOfFrames: n, numberOfChannels: ch, timestamp: Math.round(i / sr * 1e6), data: d }));
    while (ae.encodeQueueSize > 16) await sleep(5);
  }
  await ae.flush(); ae.close();
}
$("#full").onclick = async () => {
  if (busy) return; err(); cancelFlag = false; $("#dl").hidden = true;
  if (!window.VideoEncoder) return err("Trình duyệt không hỗ trợ WebCodecs (cần Chrome/Edge/Safari mới).");
  if (V.w * V.h > 1280 * 720 * 1.05) return err("Xử lý cả video trong trình duyệt giới hạn ở 720p. Hãy giảm độ phân giải video trước.");
  const scale = +$("#scale").value, model = $("#model").value, fps = +$("#fps").value || V.fps, total = Math.floor(V.dur * fps);
  const ow = V.w * scale, oh = V.h * scale; let enc;
  lock(true); $("#fp").hidden = false; $("#cancel").hidden = false; vid.pause();
  try {
    const audio = await prepAudio(); if (!audio) warn("Không giữ được audio (video không có tiếng, hoặc trình duyệt không hỗ trợ mã hóa AAC). Video xuất ra sẽ không có tiếng.");
    const cfgs = ["avc1.640034", "avc1.64002A", "avc1.4D4028"].map(codec => ({ codec, width: ow, height: oh, framerate: fps, bitrate: Math.min(60e6, Math.round(ow * oh * fps * 0.12)) }));
    let cfg = null; for (const c of cfgs) if ((await VideoEncoder.isConfigSupported(c)).supported) { cfg = c; break; }
    if (!cfg) throw new Error(`Trình duyệt không mã hóa được H.264 ở ${ow}×${oh}. Thử 2× hoặc video nhỏ hơn.`);
    const target = new ArrayBufferTarget(), muxer = new Muxer({ target, video: { codec: "avc", width: ow, height: oh }, audio: audio ? { codec: "aac", numberOfChannels: audio.ch, sampleRate: audio.buf.sampleRate } : undefined, fastStart: "in-memory" });
    let encErr = null; enc = new VideoEncoder({ output: (c, m) => muxer.addVideoChunk(c, m), error: e => encErr = e }); enc.configure(cfg);
    const t0 = performance.now(), src = mk(V.w, V.h), sx = src.getContext("2d", { willReadFrequently: true });
    for (let i = 0; i < total; i++) {
      chk(); if (encErr) throw encErr;
      await seek(Math.min(V.dur - 0.001, i / fps + 0.25 / V.fps)); sx.drawImage(vid, 0, 0);
      const out = await sr(src, scale, model), f = new VideoFrame(out, { timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps) });
      enc.encode(f, { keyFrame: i % Math.round(fps * 2) === 0 }); f.close();
      while (enc.encodeQueueSize > 6) await sleep(5);
      const n = i + 1, rate = n / ((performance.now() - t0) / 1000);
      $("#fp").value = 100 * n / total; $("#ft").textContent = `Frame ${n}/${total} · ${rate.toFixed(2)} FPS` + (n >= 5 ? ` · còn ${fmt((total - n) / rate).slice(0, 5)}` : "");
    }
    await enc.flush(); enc.close(); enc = null;
    if (audio) { $("#ft").textContent = "Đang mã hóa audio…"; await encodeAudio(audio, muxer); }
    muxer.finalize(); const blob = new Blob([target.buffer], { type: "video/mp4" });
    $("#dl").href = URL.createObjectURL(blob); $("#dl").download = `upscaled_${scale}x.mp4`; $("#dl").hidden = false;
    $("#ft").textContent = `Hoàn tất · ${(blob.size / 1048576).toFixed(1)} MB`;
  } catch (x) { try { enc && enc.close(); } catch {} x.message === CANC ? $("#ft").textContent = "Đã hủy" : (err(x.message), $("#ft").textContent = ""); }
  await seek(0).catch(() => {}); lock(false); $("#fp").hidden = true; $("#cancel").hidden = true;
};
$("#cancel").onclick = () => cancelFlag = true;
