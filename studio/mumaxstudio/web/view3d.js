"use strict";
/* 3D field viewer (WebGL 2, no libraries). Shows a vector field as arrows, cones or cells, coloured with the
   mumax3 colour wheel (hue = in-plane angle, white = +z, black = -z) or by one component.
   Sources: the running simulation (live), OVF files from the workspace, or OVF files dropped on the panel.
   Several files form a sequence that can be played back. Orbit with the left button, pan with the right
   button or Shift+drag, zoom with the wheel. */
const VIEW3D = (() => {
  let gl, canvas, prog, vao, glyphBuf = {}, instBuf, nInst = 0, ready = false;
  let frames = [], cur = -1, playing = false, playTimer = 0;
  let field = null;
  const cam = { yaw: -0.6, pitch: 0.75, dist: 2.6, tx: 0, ty: 0, tz: 0, fov: 0.62 };
  const opt = { glyph: "arrow", color: "wheel", stride: 1, size: 1, slice: [[0, 1], [0, 1], [0, 1]], outline: true, light: true };
  let listeners = {};
  let liveSource = false;

  // ---------------------------------------------------------------- glyph meshes (unit length along +z, centred)
  function cylinder(r0, r1, z0, z1, n, out) {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI;
      const c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
      const slope = (r0 - r1) / (z1 - z0), k = 1 / Math.hypot(1, slope);
      const n0 = [c0 * k, s0 * k, slope * k], n1 = [c1 * k, s1 * k, slope * k];
      const p00 = [r0 * c0, r0 * s0, z0], p01 = [r0 * c1, r0 * s1, z0], p10 = [r1 * c0, r1 * s0, z1], p11 = [r1 * c1, r1 * s1, z1];
      out.push(...p00, ...n0, ...p01, ...n1, ...p11, ...n1, ...p00, ...n0, ...p11, ...n1, ...p10, ...n0);
    }
  }
  function disk(r, z, n, up, out) {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI;
      const nn = [0, 0, up ? 1 : -1];
      const p0 = [0, 0, z], p1 = [r * Math.cos(a0), r * Math.sin(a0), z], p2 = [r * Math.cos(a1), r * Math.sin(a1), z];
      if (up) out.push(...p0, ...nn, ...p1, ...nn, ...p2, ...nn); else out.push(...p0, ...nn, ...p2, ...nn, ...p1, ...nn);
    }
  }
  function makeArrow() {
    const o = [];
    cylinder(0.07, 0.07, -0.5, 0.12, 12, o); disk(0.07, -0.5, 12, false, o);
    cylinder(0.2, 0.0, 0.12, 0.5, 16, o); disk(0.2, 0.12, 16, false, o);
    return new Float32Array(o);
  }
  function makeCone() {
    const o = [];
    cylinder(0.3, 0.0, -0.45, 0.45, 18, o); disk(0.3, -0.45, 18, false, o);
    return new Float32Array(o);
  }
  function makeCube() {
    const o = [], h = 0.5;
    const faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
      [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
    for (const [n, u, v] of faces) {
      const c = n.map((x) => x * h);
      const P = (a, b) => [c[0] + (u[0] * a + v[0] * b) * h, c[1] + (u[1] * a + v[1] * b) * h, c[2] + (u[2] * a + v[2] * b) * h];
      const q = [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)];
      for (const i of [0, 1, 2, 0, 2, 3]) o.push(...q[i], ...n);
    }
    return new Float32Array(o);
  }

  // ---------------------------------------------------------------- shaders
  const VS = `#version 300 es
  precision highp float;
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec3 aNor;
  layout(location=2) in vec3 iPos;
  layout(location=3) in vec4 iVec;   // xyz: direction (unit), w: colour scalar in [-1, 1] or |v| / max
  uniform mat4 uPV;
  uniform vec3 uScale;      // glyph size (world units) for x, y, z of the glyph frame
  uniform int uMode;        // 0 wheel, 1 scalar, 2 magnitude
  uniform int uAlign;       // 1: rotate the glyph to the vector, 0: axis-aligned (cells)
  out vec3 vNor; out vec3 vCol;
  vec3 hsl(float h, float s, float l) {
    vec3 k = mod(vec3(0.0, 8.0, 4.0) + h * 12.0, 12.0);
    float a = s * min(l, 1.0 - l);
    return l - a * clamp(min(k - 3.0, 9.0 - k), -1.0, 1.0);
  }
  vec3 diverging(float t) {   // blue - white - red
    t = clamp(t, -1.0, 1.0);
    vec3 b = vec3(0.19, 0.33, 0.78), w = vec3(0.96, 0.95, 0.93), r = vec3(0.75, 0.21, 0.17);
    return t < 0.0 ? mix(w, b, -t) : mix(w, r, t);
  }
  vec3 seq(float t) {         // dark - amber
    t = clamp(t, 0.0, 1.0);
    return mix(vec3(0.10, 0.10, 0.12), vec3(0.98, 0.72, 0.25), t) + vec3(0.0, 0.0, 0.25) * (1.0 - t) * t;
  }
  void main() {
    vec3 d = iVec.xyz;
    mat3 R = mat3(1.0);
    if (uAlign == 1) {
      vec3 z = normalize(d + vec3(0.0, 0.0, 1e-9));
      vec3 a = abs(z.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
      vec3 x = normalize(cross(a, z));
      vec3 y = cross(z, x);
      R = mat3(x, y, z);
    }
    vec3 p = R * (aPos * uScale) + iPos;
    vNor = R * aNor;
    if (uMode == 0) {
      float h = atan(d.y, d.x) / 6.2831853;
      vCol = hsl(fract(h), 1.0, clamp(0.5 + 0.5 * d.z, 0.0, 1.0));
    } else if (uMode == 1) vCol = diverging(iVec.w);
    else vCol = seq(iVec.w);
    gl_Position = uPV * vec4(p, 1.0);
  }`;
  const FS = `#version 300 es
  precision highp float;
  in vec3 vNor; in vec3 vCol;
  uniform int uLight;
  uniform vec3 uLightDir;
  out vec4 outColor;
  void main() {
    vec3 n = normalize(vNor);
    float diff = uLight == 1 ? 0.55 + 0.45 * max(dot(n, uLightDir), 0.0) + 0.12 * max(dot(n, -uLightDir), 0.0) : 1.0;
    outColor = vec4(vCol * diff, 1.0);
  }`;
  const LVS = `#version 300 es
  layout(location=0) in vec3 aPos;
  uniform mat4 uPV;
  void main() { gl_Position = uPV * vec4(aPos, 1.0); }`;
  const LFS = `#version 300 es
  precision mediump float;
  uniform vec3 uColor;
  out vec4 o;
  void main() { o = vec4(uColor, 1.0); }`;
  let lineProg, lineVao, lineBuf;

  function compile(vs, fs) {
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function initGL() {
    canvas = document.getElementById("view3d");
    gl = canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) { status("WebGL 2 is not available in this browser, so the 3D view cannot be shown."); return false; }
    prog = compile(VS, FS);
    lineProg = compile(LVS, LFS);
    for (const [k, f] of [["arrow", makeArrow], ["cone", makeCone], ["cube", makeCube]]) {
      const d = f();
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW);
      glyphBuf[k] = { buf: b, n: d.length / 6 };
    }
    instBuf = gl.createBuffer();
    vao = gl.createVertexArray();
    lineVao = gl.createVertexArray();
    lineBuf = gl.createBuffer();
    ready = true;
    return true;
  }

  // ---------------------------------------------------------------- matrices
  function persp(fov, asp, n, f) {
    const t = 1 / Math.tan(fov / 2), r = 1 / (n - f);
    return [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (n + f) * r, -1, 0, 0, 2 * n * f * r, 0];
  }
  function lookAt(e, c, u) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const nrm = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
    const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dt = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = nrm(sub(e, c)), x = nrm(cr(u, z)), y = cr(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dt(x, e), -dt(y, e), -dt(z, e), 1];
  }
  function mul(a, b) {
    const o = new Array(16).fill(0);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) o[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
    return o;
  }
  function eye() {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    return [cam.tx + cam.dist * cp * Math.cos(cam.yaw), cam.ty + cam.dist * cp * Math.sin(cam.yaw), cam.tz + cam.dist * sp];
  }

  // ---------------------------------------------------------------- building the instance buffer
  let extent = [1, 1, 1], world = 1;
  function build() {
    if (!ready || !field) { nInst = 0; draw(); return; }
    const f = field, N = f.nx * f.ny * f.nz;
    const L = [f.nx * f.cell[0], f.ny * f.cell[1], f.nz * f.cell[2]];
    world = Math.max(...L);
    extent = L.map((v) => v / world);   // normalised box, longest side 1
    const st = Math.max(1, opt.stride | 0);
    const lo = opt.slice.map(([a], i) => Math.floor(a * [f.nx, f.ny, f.nz][i]));
    const hi = opt.slice.map(([, b], i) => Math.ceil(b * [f.nx, f.ny, f.nz][i]));
    const nc = f.ncomp;
    let vmax = 0;
    for (let i = 0; i < N; i++) if (f.norm[i] > vmax) vmax = f.norm[i];
    vmax = vmax || 1;
    const comp = opt.color === "x" ? 0 : opt.color === "y" ? 1 : opt.color === "z" ? 2 : -1;
    const est = Math.ceil((hi[0] - lo[0]) / st) * Math.ceil((hi[1] - lo[1]) / st) * Math.max(1, hi[2] - lo[2]);
    const out = new Float32Array(Math.max(1, est) * 7);
    let k = 0, sum = [0, 0, 0], cnt = 0;
    const zst = f.nz > 64 ? st : 1;
    for (let iz = lo[2]; iz < hi[2]; iz += zst) {
      for (let iy = lo[1]; iy < hi[1]; iy += st) {
        for (let ix = lo[0]; ix < hi[0]; ix += st) {
          const i = ix + f.nx * (iy + f.ny * iz);
          const n = f.norm[i];
          if (!(n > 0) || (f.geom && !f.geom[i])) continue;
          if (k + 7 > out.length) break;
          const vx = f.data[i], vy = nc > 1 ? f.data[N + i] : 0, vz = nc > 2 ? f.data[2 * N + i] : 0;
          out[k] = ((ix + 0.5) * f.cell[0]) / world - extent[0] / 2;
          out[k + 1] = ((iy + 0.5) * f.cell[1]) / world - extent[1] / 2;
          out[k + 2] = ((iz + 0.5) * f.cell[2]) / world - extent[2] / 2;
          if (nc >= 3) { out[k + 3] = vx / n; out[k + 4] = vy / n; out[k + 5] = vz / n; }
          else { out[k + 3] = 0; out[k + 4] = 0; out[k + 5] = 1; }
          out[k + 6] = nc === 1 ? f.data[i] / vmax : comp >= 0 ? [vx, vy, vz][comp] / vmax : n / vmax;
          sum[0] += vx; sum[1] += vy; sum[2] += vz; cnt++;
          k += 7;
        }
      }
    }
    nInst = k / 7;
    field.avg = cnt ? sum.map((s) => s / cnt) : [0, 0, 0];
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, out.subarray(0, k), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 28, 0); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 28, 12); gl.vertexAttribDivisor(3, 1);
    bindGlyph();
    // outline of the box
    const [a, b, c] = extent.map((v) => v / 2);
    const P = [[-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c], [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c]];
    const E = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    const lines = new Float32Array(E.flatMap((i) => P[i]));
    // axis triad at the corner
    const s = 0.12;
    const ax = [-a, -b, -c];
    const tri = new Float32Array([...ax, ax[0] + s, ax[1], ax[2], ...ax, ax[0], ax[1] + s, ax[2], ...ax, ax[0], ax[1], ax[2] + s]);
    gl.bindVertexArray(lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    const all = new Float32Array(lines.length + tri.length);
    all.set(lines); all.set(tri, lines.length);
    gl.bufferData(gl.ARRAY_BUFFER, all, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
    gl.bindVertexArray(null);
    draw();
    emit("info");
  }
  function bindGlyph() {
    gl.bindVertexArray(vao);
    const g = glyphBuf[opt.glyph];
    gl.bindBuffer(gl.ARRAY_BUFFER, g.buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0); gl.vertexAttribDivisor(0, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12); gl.vertexAttribDivisor(1, 0);
    gl.bindVertexArray(null);
  }

  // ---------------------------------------------------------------- drawing
  function themeColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const m = /^#([0-9a-f]{6})$/i.exec(v);
    if (!m) return fallback;
    const n = parseInt(m[1], 16);
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  }
  function draw() {
    if (!ready) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, Hh = canvas.clientHeight;
    if (!W || !Hh) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(Hh * dpr)) {
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hh * dpr);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    const bg = themeColor("--view-bg", [0.96, 0.95, 0.93]);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!field) return;
    gl.enable(gl.DEPTH_TEST);
    const e = eye();
    const up = Math.cos(cam.pitch) >= 0 ? [0, 0, 1] : [0, 0, -1];
    const PV = mul(persp(cam.fov, W / Hh, 0.01, 50), lookAt(e, [cam.tx, cam.ty, cam.tz], up));
    if (nInst) {
      gl.useProgram(prog);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uPV"), false, PV);
      const f = field, st = Math.max(1, opt.stride | 0);
      const cs = [f.cell[0] * st / world, f.cell[1] * st / world, f.cell[2] * (f.nz > 64 ? st : 1) / world];
      let sc;
      if (opt.glyph === "cube") sc = cs.map((v) => v * 0.98 * Math.min(1, opt.size));
      else { const s = Math.min(cs[0], cs[1]) * 0.95 * opt.size; sc = [s, s, s]; }
      gl.uniform3fv(gl.getUniformLocation(prog, "uScale"), sc);
      gl.uniform1i(gl.getUniformLocation(prog, "uMode"), field.ncomp === 1 ? 1 : opt.color === "wheel" ? 0 : opt.color === "mag" ? 2 : 1);
      gl.uniform1i(gl.getUniformLocation(prog, "uAlign"), opt.glyph === "cube" ? 0 : 1);
      gl.uniform1i(gl.getUniformLocation(prog, "uLight"), opt.light ? 1 : 0);
      const ld = [e[0] - cam.tx + 0.4, e[1] - cam.ty - 0.3, e[2] - cam.tz + 1.2];
      const ll = Math.hypot(...ld);
      gl.uniform3fv(gl.getUniformLocation(prog, "uLightDir"), ld.map((v) => v / ll));
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, glyphBuf[opt.glyph].n, nInst);
      gl.bindVertexArray(null);
    }
    if (opt.outline) {
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uPV"), false, PV);
      gl.bindVertexArray(lineVao);
      gl.uniform3fv(gl.getUniformLocation(lineProg, "uColor"), themeColor("--view-line", [0.45, 0.45, 0.45]));
      gl.drawArrays(gl.LINES, 0, 24);
      const cols = [[0.75, 0.21, 0.17], [0.15, 0.55, 0.3], [0.19, 0.33, 0.78]];
      for (let i = 0; i < 3; i++) { gl.uniform3fv(gl.getUniformLocation(lineProg, "uColor"), cols[i]); gl.drawArrays(gl.LINES, 24 + 2 * i, 2); }
      gl.bindVertexArray(null);
    }
  }

  // ---------------------------------------------------------------- camera interaction
  function bindMouse() {
    let drag = null;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.button === 1 || e.shiftKey };
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const s = cam.dist * 0.0016;
        const rx = -Math.sin(cam.yaw), ry = Math.cos(cam.yaw);
        cam.tx -= (dx * rx) * s; cam.ty -= (dx * ry) * s;
        const upx = -Math.cos(cam.yaw) * Math.sin(cam.pitch), upy = -Math.sin(cam.yaw) * Math.sin(cam.pitch), upz = Math.cos(cam.pitch);
        cam.tx += dy * upx * s; cam.ty += dy * upy * s; cam.tz += dy * upz * s;
      } else {
        cam.yaw -= dx * 0.008;
        cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch + dy * 0.008));
      }
      draw();
    });
    const end = () => { drag = null; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      cam.dist = Math.max(0.05, Math.min(20, cam.dist * Math.exp(e.deltaY * 0.0012)));
      draw();
    }, { passive: false });
  }
  function viewFrom(which) {
    Object.assign(cam, { tx: 0, ty: 0, tz: 0 });
    if (which === "top") { cam.yaw = -Math.PI / 2; cam.pitch = 1.5699; }
    else if (which === "front") { cam.yaw = -Math.PI / 2; cam.pitch = 0; }
    else if (which === "side") { cam.yaw = 0; cam.pitch = 0; }
    else { cam.yaw = -0.6; cam.pitch = 0.75; }
    cam.dist = 2.6 * Math.max(0.55, Math.max(...extent));
    draw();
  }

  // ---------------------------------------------------------------- frames
  function setFrames(list, keepIndex = false) {
    frames = list;
    cur = keepIndex ? Math.min(cur, frames.length - 1) : 0;
    liveSource = false;
    showFrame(cur);
    emit("frames");
  }
  async function showFrame(i) {
    if (!frames.length) return;
    cur = (i + frames.length) % frames.length;
    const fr = frames[cur];
    try {
      if (!fr.field) fr.field = await fr.load();
      const first = !field || field.nx !== fr.field.nx || field.ny !== fr.field.ny || field.nz !== fr.field.nz;
      field = fr.field;
      build();
      if (first) viewFrom("iso");
      status("");
    } catch (e) { status(e.message); }
    emit("frames");
  }
  function play(on) {
    playing = on === undefined ? !playing : on;
    clearTimeout(playTimer);
    const step = async () => {
      if (!playing) return;
      await showFrame(cur + 1);
      playTimer = setTimeout(step, 1000 / (Number(document.getElementById("v3fps").value) || 8));
    };
    if (playing) step();
    emit("frames");
  }

  // ---------------------------------------------------------------- live data and files
  async function loadLive(api) {
    const buf = await api("/api/field3d?q=" + encodeURIComponent(document.getElementById("v3quantity").value || "m"), { binary: true });
    const f = OVF.fromStudio(buf);
    const first = !field || !liveSource || field.nx !== f.nx || field.ny !== f.ny || field.nz !== f.nz;
    field = f; liveSource = true;
    frames = []; cur = -1;
    build();
    if (first) viewFrom("iso");
    emit("frames");
  }
  function addLocalFiles(fileList) {
    const files = Array.from(fileList).filter((f) => /\.ovf$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (!files.length) { status("Drop .ovf files here (one file, or a whole series to play back)."); return; }
    setFrames(files.map((f) => ({ name: f.name, load: async () => OVF.parse(await f.arrayBuffer(), f.name) })));
  }
  function addWorkspaceFiles(paths, api) {
    setFrames(paths.map((p) => ({
      name: p.split("/").pop(), path: p,
      load: async () => OVF.parse(await (await api("/api/file?path=" + encodeURIComponent(p), { blob: true })).arrayBuffer(), p.split("/").pop()),
    })));
  }

  // ---------------------------------------------------------------- misc
  function status(t) {
    const el = document.getElementById("v3status");
    el.textContent = t; el.hidden = !t;
  }
  function emit(ev) { (listeners[ev] || []).forEach((f) => f()); }
  function on(ev, f) { (listeners[ev] = listeners[ev] || []).push(f); }
  function info() {
    if (!field) return null;
    return {
      name: field.name, t: field.t, n: [field.nx, field.ny, field.nz], fullN: field.fullN, stride: field.stride,
      cell: field.cell, avg: field.avg, glyphs: nInst, frame: cur, frames: frames.length, live: liveSource,
      frameName: frames[cur] ? frames[cur].name : "", ncomp: field.ncomp,
    };
  }
  function init() {
    if (!initGL()) return;
    bindMouse();
    const wrap = document.getElementById("v3wrap");
    ["dragenter", "dragover"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.add("drop"); }));
    ["dragleave", "drop"].forEach((t) => wrap.addEventListener(t, (e) => { e.preventDefault(); wrap.classList.remove("drop"); }));
    wrap.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) addLocalFiles(e.dataTransfer.files); });
    draw();
  }
  return {
    init, draw, build, on, info, play, showFrame, viewFrom, loadLive, addLocalFiles, addWorkspaceFiles, status,
    opt, get playing() { return playing; }, get hasField() { return !!field; }, get live() { return liveSource; },
    png(cb) { draw(); canvas.toBlob(cb); },
  };
})();
