"use strict";
/* Geometry builder: an ordered stack of shapes combined with boolean operations, each with its own transform.
   The authoritative preview is rasterised by the engine (/api/preview_geom), so imported STL models, images and
   custom code look exactly as they will in the simulation. Outlines and hit-testing of the primitive shapes are done
   here, which is what makes drag / rotate / scale in the viewer immediate. All lengths in the UI are nanometres. */

const GEOM = (() => {
  const OPS = {
    add: ["∪", "add to magnet"], sub: ["−", "cut out"], and: ["∩", "intersect"], xor: ["⊕", "toggle (xor)"],
    region: ["R", "region only (no change to the magnet)"],
  };
  // type: label, params [key, label, default (nm unless noted), kind], 2D?
  const TYPES = {
    rect: { label: "Rectangle", p: [["w", "width", 200], ["h", "height", 100]] },
    circle: { label: "Disk", p: [["d", "diameter", 150]] },
    ellipse: { label: "Ellipse", p: [["dx", "diameter x", 200], ["dy", "diameter y", 100]] },
    ring: { label: "Ring", p: [["dout", "outer diameter", 200], ["din", "inner diameter", 100]] },
    regpoly: { label: "Regular polygon", p: [["n", "sides", 6, "int"], ["d", "diameter", 160]], py: false },
    polygon: { label: "Polygon", p: [["verts", "vertices x,y; x,y; … (nm)", "-100,-60; 100,-60; 0,90", "text"]] },
    cuboid: { label: "Cuboid", p: [["w", "size x", 200], ["h", "size y", 100], ["t", "size z", 20]] },
    cylinder: { label: "Cylinder", p: [["d", "diameter", 150], ["t", "height", 20]] },
    ellipsoid: { label: "Ellipsoid", p: [["dx", "diameter x", 200], ["dy", "diameter y", 100], ["dz", "diameter z", 40]] },
    sphere: { label: "Sphere", p: [["d", "diameter", 100]] },
    cone: { label: "Cone", p: [["d", "base diameter", 150], ["t", "height", 40]] },
    torus: { label: "Torus", p: [["D", "ring diameter", 160], ["d", "tube diameter", 40]] },
    layers: { label: "Layers", p: [["z0", "first layer index", 0, "int"], ["z1", "last layer + 1", 1, "int"]], fixed: true },
    stl: { label: "3D model (STL)", p: [["file", "file in workspace", "model.stl", "file"], ["unit", "nm per file unit", 1], ["res", "voxels along longest side", 192, "int"]] },
    image: { label: "Image mask", p: [["file", "file in workspace", "mask.png", "file"]], fixed: true },
    custom: { label: "Custom inside(x, y, z)", p: [["code", "Python expression in x, y, z (metres), NumPy arrays", "(x/120e-9)**2 + (y/60e-9)**2 < 1 + 0.3*np.cos(6*np.arctan2(y, x))", "code"]] },
  };

  let shapes = [];
  let sel = -1;
  let uid = 1;
  let onChange = () => {};
  const $ = (s) => document.querySelector(s);
  const nm = (v) => WAVES.num(v * 1e-9);

  function mesh() {
    const g = (id) => Number(document.getElementById(id).value) || 0;
    const pbc = (document.getElementById("pbc").value || "0,0,0").split(/[,\s]+/).map((v) => Math.max(0, parseInt(v, 10) || 0));
    while (pbc.length < 3) pbc.push(0);
    return { n: [Math.max(1, g("nx") | 0), Math.max(1, g("ny") | 0), Math.max(1, g("nz") | 0)],
      cell: [g("cx"), g("cy"), g("cz")], pbc: pbc.slice(0, 3), smooth: Math.max(0, g("edgesmooth") | 0) };
  }

  function add(type) {
    const p = {};
    for (const [k, , v] of TYPES[type].p) p[k] = v;
    const s = { id: uid++, type, op: shapes.length ? "add" : "add", name: TYPES[type].label, on: true, region: 0, p,
      x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, s: 1, rep: [0, 0] };
    shapes.push(s);
    sel = shapes.length - 1;
    changed();
  }
  function remove(i) { shapes.splice(i, 1); sel = Math.min(sel, shapes.length - 1); changed(); }
  function duplicate(i) {
    const c = JSON.parse(JSON.stringify(shapes[i]));
    c.id = uid++; c.x += 20; c.y -= 20; c.name = shapes[i].name + " copy";
    shapes.splice(i + 1, 0, c); sel = i + 1; changed();
  }
  function move(i, d) {
    const j = i + d;
    if (j < 0 || j >= shapes.length) return;
    [shapes[i], shapes[j]] = [shapes[j], shapes[i]];
    sel = j; changed();
  }
  function changed(keepEditor) {
    renderList();
    if (!keepEditor) renderEditor();
    onChange();
  }

  // ---------------------------------------------------------------- code generation
  function needsPython() { return shapes.some((s) => s.on && s.type === "custom"); }

  function baseExpr(s, lang) {
    const p = s.p;
    switch (s.type) {
      case "rect": return `Rect(${nm(p.w)}, ${nm(p.h)})`;
      case "circle": return `Circle(${nm(p.d)})`;
      case "ellipse": return `Ellipse(${nm(p.dx)}, ${nm(p.dy)})`;
      case "ring": return `Circle(${nm(p.dout)}).Sub(Circle(${nm(p.din)}))`;
      case "regpoly": return `RegularPolygon(${Math.max(3, p.n | 0)}, ${nm(p.d)})`;
      case "polygon": {
        const v = parseVerts(p.verts);
        return `Polygon([${v.map((q) => nm(q[0])).join(", ")}], [${v.map((q) => nm(q[1])).join(", ")}])`;
      }
      case "cuboid": return `Cuboid(${nm(p.w)}, ${nm(p.h)}, ${nm(p.t)})`;
      case "cylinder": return `Cylinder(${nm(p.d)}, ${nm(p.t)})`;
      case "ellipsoid": return `Ellipsoid(${nm(p.dx)}, ${nm(p.dy)}, ${nm(p.dz)})`;
      case "sphere": return `Ellipsoid(${nm(p.d)}, ${nm(p.d)}, ${nm(p.d)})`;
      case "cone": return `Cone(${nm(p.d)}, ${nm(p.t)})`;
      case "torus": return `Torus(${nm(p.D)}, ${nm(p.d)})`;
      case "layers": return `Layers(${p.z0 | 0}, ${p.z1 | 0})`;
      case "stl": return `STLShape(${JSON.stringify(String(p.file))}, ${WAVES.num(p.unit * 1e-9)}, ${p.res | 0})`;
      case "image": return `ImageShape(${JSON.stringify(String(p.file))})`;
      case "custom": return lang === "py" ? `Shape(lambda x, y, z: ${p.code})` : "Universe()";
      default: return "Universe()";
    }
  }
  function shapeExpr(s, lang) {
    let e = baseExpr(s, lang);
    if (TYPES[s.type].fixed) return e;
    const deg = (a) => WAVES.num(a * Math.PI / 180);
    if (s.s !== 1) e += `.Scale(${WAVES.num(s.s)}, ${WAVES.num(s.s)}, 1)`;
    if (s.rx) e += `.RotX(${deg(s.rx)})`;
    if (s.ry) e += `.RotY(${deg(s.ry)})`;
    if (s.rz) e += `.RotZ(${deg(s.rz)})`;
    if (s.rep[0] || s.rep[1]) e += `.Repeat(${nm(s.rep[0])}, ${nm(s.rep[1])}, 0)`;
    if (s.x || s.y || s.z) e += `.Transl(${nm(s.x)}, ${nm(s.y)}, ${nm(s.z)})`;
    return e;
  }
  // lines of script code (without mesh) that set the geometry and the regions
  function code(lang) {
    const L = [];
    const decl = (n, e) => L.push(lang === "py" ? `${n} = ${e}` : `${n} := ${e}`);
    const set = (n, e) => L.push(`${n} = ${e}`);
    const geo = shapes.filter((s) => s.on && s.op !== "region");
    if (geo.length) {
      const first = geo[0];
      decl("magnet", first.op === "sub" ? `Universe().Sub(${shapeExpr(first, lang)})` : shapeExpr(first, lang));
      for (const s of geo.slice(1)) {
        const m = { add: "Add", sub: "Sub", and: "Intersect", xor: "Xor" }[s.op];
        set("magnet", `magnet.${m}(${shapeExpr(s, lang)})`);
      }
      L.push("SetGeom(magnet)");
    }
    for (const s of shapes) {
      if (s.on && s.region > 0) L.push(`DefRegion(${s.region | 0}, ${shapeExpr(s, lang)})`);
    }
    return L;
  }
  function regionsUsed() {
    return [...new Set(shapes.filter((s) => s.on && s.region > 0).map((s) => s.region | 0))].sort((a, b) => a - b);
  }
  function parseVerts(t) {
    return String(t).split(";").map((p) => p.split(",").map(Number)).filter((q) => q.length === 2 && q.every(isFinite));
  }

  // ---------------------------------------------------------------- client-side inside tests (nm, at height z)
  function insideFn(s) {
    const p = s.p;
    const c = Math.cos(-s.rz * Math.PI / 180), sn = Math.sin(-s.rz * Math.PI / 180);
    const rep = s.rep;
    const wrap = (v, P) => (P ? ((((v + P / 2) % P) + P) % P) - P / 2 : v);
    let f;
    switch (s.type) {
      case "rect": f = (x, y) => Math.abs(x) <= p.w / 2 && Math.abs(y) <= p.h / 2; break;
      case "cuboid": f = (x, y, z) => Math.abs(x) <= p.w / 2 && Math.abs(y) <= p.h / 2 && Math.abs(z) <= p.t / 2; break;
      case "circle": f = (x, y) => x * x + y * y <= p.d * p.d / 4; break;
      case "cylinder": f = (x, y, z) => x * x + y * y <= p.d * p.d / 4 && Math.abs(z) <= p.t / 2; break;
      case "ellipse": f = (x, y) => (x / (p.dx / 2)) ** 2 + (y / (p.dy / 2)) ** 2 <= 1; break;
      case "ellipsoid": f = (x, y, z) => (x / (p.dx / 2)) ** 2 + (y / (p.dy / 2)) ** 2 + (z / (p.dz / 2)) ** 2 <= 1; break;
      case "sphere": f = (x, y, z) => x * x + y * y + z * z <= p.d * p.d / 4; break;
      case "ring": f = (x, y) => { const r = x * x + y * y; return r <= p.dout * p.dout / 4 && r >= p.din * p.din / 4; }; break;
      case "torus": f = (x, y, z) => (Math.hypot(x, y) - p.D / 2) ** 2 + z * z <= p.d * p.d / 4; break;
      case "cone": f = (x, y, z) => { const u = (z + p.t / 2) / p.t; return u >= 0 && u <= 1 && Math.hypot(x, y) <= (1 - u) * p.d / 2; }; break;
      case "regpoly": case "polygon": {
        const v = s.type === "polygon" ? parseVerts(p.verts) : Array.from({ length: Math.max(3, p.n | 0) },
          (_, k) => [p.d / 2 * Math.cos(2 * Math.PI * k / Math.max(3, p.n | 0)), p.d / 2 * Math.sin(2 * Math.PI * k / Math.max(3, p.n | 0))]);
        f = (x, y) => {
          let ins = false;
          for (let i = 0, j = v.length - 1; i < v.length; j = i++) {
            const [xi, yi] = v[i], [xj, yj] = v[j];
            if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) ins = !ins;
          }
          return ins;
        };
        break;
      }
      default: return null; // STL, image, custom, layers: only the engine preview knows
    }
    return (x, y, z) => {
      let X = x - s.x, Y = y - s.y;
      const Z = z - s.z;
      X = wrap(X, rep[0]); Y = wrap(Y, rep[1]);
      const xr = X * c - Y * sn, yr = X * sn + Y * c;
      return f(xr / s.s, yr / s.s, Z);
    };
  }

  // ---------------------------------------------------------------- list UI
  function renderList() {
    const ol = $("#shapeList");
    ol.innerHTML = "";
    $("#shapeEmpty").hidden = shapes.length > 0;
    shapes.forEach((s, i) => {
      const li = document.createElement("li");
      li.className = "shapeitem" + (i === sel ? " sel" : "") + (s.on ? "" : " off");
      const op = document.createElement("span");
      op.className = "opbadge " + s.op;
      op.textContent = OPS[s.op][0];
      op.title = OPS[s.op][1];
      const name = document.createElement("span");
      name.className = "shapename";
      name.textContent = s.name;
      if (s.region > 0) { const r = document.createElement("small"); r.textContent = "region " + s.region; name.appendChild(r); }
      const acts = document.createElement("span");
      acts.className = "shapeacts";
      const btn = (icon, title, fn) => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "iconbtn"; b.title = title; b.setAttribute("aria-label", title);
        b.innerHTML = `<svg><use href="#i-${icon}"/></svg>`;
        b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
        acts.appendChild(b);
      };
      btn("eye", s.on ? "Hide" : "Show", () => { s.on = !s.on; changed(true); });
      btn("up", "Move up", () => move(i, -1));
      btn("down", "Move down", () => move(i, 1));
      btn("copy", "Duplicate (Shift+D)", () => duplicate(i));
      btn("x", "Remove (Del)", () => remove(i));
      li.append(op, name, acts);
      li.addEventListener("click", () => { sel = i; changed(); });
      ol.appendChild(li);
    });
  }

  function field(label, value, onInput, opts = {}) {
    const lab = document.createElement("label");
    if (opts.span) lab.className = "span" + opts.span;
    lab.append(label);
    let inp;
    if (opts.kind === "code" || opts.kind === "text") {
      inp = document.createElement(opts.kind === "code" ? "textarea" : "input");
      if (opts.kind === "code") { inp.className = "code"; inp.rows = 3; inp.spellcheck = false; }
      inp.value = value;
    } else if (opts.options) {
      inp = document.createElement("select");
      for (const [v, t] of opts.options) { const o = document.createElement("option"); o.value = v; o.textContent = t; inp.appendChild(o); }
      inp.value = value;
    } else {
      inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.value = value;
    }
    inp.addEventListener("input", () => onInput(opts.kind === "code" || opts.kind === "text" || opts.options ? inp.value : Number(inp.value)));
    if (opts.unit) {
      const u = document.createElement("span"); u.className = "u";
      const i = document.createElement("i"); i.textContent = opts.unit;
      u.append(inp, i); lab.appendChild(u);
    } else lab.appendChild(inp);
    return lab;
  }

  function renderEditor() {
    const box = $("#shapeEdit");
    const s = shapes[sel];
    box.hidden = !s;
    if (!s) return;
    box.innerHTML = "";
    const T = TYPES[s.type];
    const g1 = document.createElement("div"); g1.className = "grid3";
    g1.append(
      field("name", s.name, (v) => { s.name = v; renderList(); }, { kind: "text", span: 2 }),
      field("operation", s.op, (v) => { s.op = v; changed(true); }, { options: Object.entries(OPS).map(([k, v]) => [k, `${v[0]}  ${v[1]}`]) }),
    );
    box.appendChild(g1);
    const h = document.createElement("h4"); h.textContent = T.label; h.style.marginTop = "10px";
    box.appendChild(h);
    const g2 = document.createElement("div"); g2.className = "grid3";
    for (const [k, label, , kind] of T.p) {
      if (kind === "file") {
        const lab = field(label, s.p[k], (v) => { s.p[k] = v; changed(true); }, { kind: "text", span: 2 });
        const up = document.createElement("label");
        up.className = "btn small filebtn"; up.textContent = "Upload…";
        const fi = document.createElement("input"); fi.type = "file";
        fi.accept = s.type === "stl" ? ".stl" : ".png,.jpg,.jpeg";
        fi.addEventListener("change", async () => {
          const f = fi.files[0];
          if (!f) return;
          const name = await STUDIO.upload(f);
          if (name) { s.p[k] = name; renderEditor(); changed(true); }
        });
        up.appendChild(fi);
        const wrap = document.createElement("label"); wrap.append(" ", up);
        g2.append(lab, wrap);
      } else if (kind === "code" || kind === "text") {
        g2.appendChild(field(label, s.p[k], (v) => { s.p[k] = v; changed(true); }, { kind, span: 3 }));
      } else {
        g2.appendChild(field(label, s.p[k], (v) => { s.p[k] = kind === "int" ? Math.round(v) : v; changed(true); },
          { unit: kind === "int" || k === "unit" ? "" : "nm" }));
      }
    }
    box.appendChild(g2);
    if (s.type === "custom") {
      const pn = document.createElement("p"); pn.className = "hint";
      pn.textContent = "Custom shapes need Python: the generated script switches to .py. x, y, z are cell-centre coordinates in metres; return a boolean array.";
      box.appendChild(pn);
    }
    if (!T.fixed) {
      const h2 = document.createElement("h4"); h2.textContent = "Transform"; h2.style.marginTop = "10px";
      box.appendChild(h2);
      const g3 = document.createElement("div"); g3.className = "grid3";
      const tf = (label, key, unit) => field(label, s[key], (v) => { s[key] = v; changed(true); }, { unit });
      g3.append(tf("x", "x", "nm"), tf("y", "y", "nm"), tf("z", "z", "nm"),
        tf("rotate z", "rz", "°"), tf("rotate x", "rx", "°"), tf("rotate y", "ry", "°"),
        tf("scale", "s", "×"),
        field("repeat every x", s.rep[0], (v) => { s.rep[0] = v; changed(true); }, { unit: "nm" }),
        field("repeat every y", s.rep[1], (v) => { s.rep[1] = v; changed(true); }, { unit: "nm" }));
      box.appendChild(g3);
    }
    const g4 = document.createElement("div"); g4.className = "grid3"; g4.style.marginTop = "8px";
    g4.append(field("region (0 = none)", s.region, (v) => { s.region = Math.max(0, Math.min(255, Math.round(v))); changed(true); }));
    box.appendChild(g4);
  }

  // ---------------------------------------------------------------- preview (viewer canvas)
  let preview = null;   // {hdr, vol, reg, geom}
  let previewKey = "";
  let previewTimer = null;
  let previewSeq = 0;
  let view = null;      // last drawing transform
  function requestPreview(layer) {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => fetchPreview(layer), 220);
  }
  function previewBody(layer) {
    const M = mesh();
    const lang = needsPython() ? "py" : "mx3";
    return { n: M.n, cell: M.cell.map((c) => c * 1e-9), code: code(lang).join("\n"), lang, layer, edgesmooth: M.smooth };
  }
  async function fetchPreview(layer) {
    const body = previewBody(layer);
    const key = JSON.stringify(body);
    if (key === previewKey && preview) { STUDIO.redraw(); return; }
    const seq = ++previewSeq;
    try {
      const buf = await STUDIO.api("/api/preview_geom", { body, binary: true });
      if (seq !== previewSeq) return;
      const dv = new DataView(buf);
      const hl = dv.getUint32(0, true);
      const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
      const n = hdr.nx * hdr.ny;
      const data = new Float32Array(buf, 4 + hl, 2 * n);
      preview = { hdr, vol: data.subarray(0, n), reg: data.subarray(n, 2 * n) };
      previewKey = key;
      STUDIO.redraw();
    } catch (e) {
      if (seq === previewSeq) { preview = null; STUDIO.info("geometry preview: " + e.message); STUDIO.redraw(); }
    }
  }

  const REGION_COLORS = ["#2f4fd8", "#a4481f", "#1d7f49", "#8b3fb3", "#0c8599", "#c08a1e", "#d6453d", "#5c7cfa"];
  function draw(ctx, W, H, layer, css) {
    const M = mesh();
    const [nx, ny, nz] = M.n, [cx, cy, cz] = M.cell;
    const pw = nx * cx, ph = ny * cy;
    const pad = 16;
    const sc = Math.min((W - 2 * pad) / pw, (H - 2 * pad) / ph);
    const dw = pw * sc, dh = ph * sc, x0 = (W - dw) / 2, y0 = (H - dh) / 2;
    view = { x0, y0, dw, dh, sc, pw, ph, nx, ny, cx, cy, layer, z: cz * (layer - (nz - 1) / 2) };
    ctx.fillStyle = css("--panel-2"); ctx.fillRect(x0, y0, dw, dh);
    if (preview && preview.hdr.nx === nx && preview.hdr.ny === ny) {
      const img = new ImageData(nx, ny);
      const ink = hex(css("--ink")), regCols = REGION_COLORS.map(hex);
      for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
        const i = iy * nx + ix, o = ((ny - 1 - iy) * nx + ix) * 4, v = preview.vol[i];
        if (v <= 0) continue;
        const r = preview.reg[i] | 0;
        const col = r > 0 ? regCols[(r - 1) % regCols.length] : ink;
        img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = Math.round(255 * (0.25 + 0.6 * v));
      }
      const off = document.createElement("canvas"); off.width = nx; off.height = ny;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, x0, y0, dw, dh);
    }
    // cell grid when cells are large on screen
    if (cx * sc >= 9 && nx <= 256) {
      ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= nx; i++) { const X = Math.round(x0 + i * cx * sc) + 0.5; ctx.moveTo(X, y0); ctx.lineTo(X, y0 + dh); }
      for (let j = 0; j <= ny; j++) { const Y = Math.round(y0 + j * cy * sc) + 0.5; ctx.moveTo(x0, Y); ctx.lineTo(x0 + dw, Y); }
      ctx.stroke();
    }
    ctx.strokeStyle = css("--line-2"); ctx.strokeRect(x0 - 0.5, y0 - 0.5, dw + 1, dh + 1);
    // selected shape outline
    const s = shapes[sel];
    if (s && s.on) {
      const f = insideFn(s);
      if (f) {
        const R = 2; // outline resolution in px
        const gw = Math.ceil(dw / R), gh = Math.ceil(dh / R);
        const grid = new Uint8Array(gw * gh);
        for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
          const X = -pw / 2 + (i + 0.5) * R / sc, Y = ph / 2 - (j + 0.5) * R / sc;
          grid[j * gw + i] = f(X, Y, view.z) ? 1 : 0;
        }
        ctx.fillStyle = css("--data");
        for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
          const v = grid[j * gw + i];
          if (!v) continue;
          if (i === 0 || j === 0 || i === gw - 1 || j === gh - 1 || !grid[j * gw + i - 1] || !grid[j * gw + i + 1] || !grid[(j - 1) * gw + i] || !grid[(j + 1) * gw + i])
            ctx.fillRect(x0 + i * R, y0 + j * R, R, R);
        }
      }
      // origin cross of the selected shape
      const ox = x0 + (s.x + pw / 2) * sc, oy = y0 + (ph / 2 - s.y) * sc;
      ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ox - 6, oy); ctx.lineTo(ox + 6, oy); ctx.moveTo(ox, oy - 6); ctx.lineTo(ox, oy + 6); ctx.stroke();
    }
    // scale bar
    const target = dw / 5 / sc;
    const mag = Math.pow(10, Math.floor(Math.log10(target)));
    const len = [1, 2, 5, 10].map((m) => m * mag).filter((v) => v <= target).pop() || mag;
    ctx.fillStyle = css("--ink"); ctx.font = "11px " + css("--font-mono");
    const bx = x0, by = y0 + dh + 12;
    if (by < H - 2) {
      ctx.fillRect(bx, by - 2, len * sc, 2);
      ctx.fillText(len >= 1000 ? `${len / 1000} µm` : `${len} nm`, bx + len * sc + 6, by + 2);
    }
    const info = preview ? `${preview.hdr.cells.toLocaleString()} of ${preview.hdr.total.toLocaleString()} cells magnetic · layer ${layer}` +
      (preview.hdr.error ? ` · ${preview.hdr.error}` : "") : "computing preview…";
    return info;
  }
  function hex(c) {
    c = c.trim();
    if (c.startsWith("#")) {
      if (c.length === 4) c = "#" + [...c.slice(1)].map((h) => h + h).join("");
      return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
    }
    const m = c.match(/\d+/g);
    return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
  }
  function toWorld(px, py) {
    if (!view) return null;
    return [(px - view.x0) / view.sc - view.pw / 2, view.ph / 2 - (py - view.y0) / view.sc];
  }
  function hit(px, py) {
    const w = toWorld(px, py);
    if (!w) return -1;
    for (let i = shapes.length - 1; i >= 0; i--) {
      const s = shapes[i];
      if (!s.on) continue;
      const f = insideFn(s);
      if (f && f(w[0], w[1], view.z)) return i;
    }
    return -1;
  }

  // ---------------------------------------------------------------- mouse / keyboard on the viewer
  let drag = null;
  function attach(canvas, wrap, active) {
    canvas.addEventListener("mousedown", (e) => {
      if (!active()) return;
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const i = hit(px, py);
      if (i >= 0 && i !== sel) { sel = i; renderList(); renderEditor(); }
      const s = shapes[sel];
      if (!s || TYPES[s.type].fixed) { STUDIO.redraw(); return; }
      const w = toWorld(px, py);
      drag = { s, w, x: s.x, y: s.y };
      wrap.classList.add("dragging");
      wrap.focus();
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const r = canvas.getBoundingClientRect();
      const w = toWorld(e.clientX - r.left, e.clientY - r.top);
      const M = mesh();
      const snap = (v, c) => (e.altKey ? v : Math.round(v / c) * c);  // snap to cells unless Alt
      drag.s.x = snap(drag.x + w[0] - drag.w[0], M.cell[0] / 2);
      drag.s.y = snap(drag.y + w[1] - drag.w[1], M.cell[1] / 2);
      STUDIO.redraw();
    });
    window.addEventListener("mouseup", () => {
      if (!drag) return;
      drag = null;
      wrap.classList.remove("dragging");
      changed();
    });
    canvas.addEventListener("wheel", (e) => {
      if (!active() || !(e.shiftKey || e.ctrlKey)) return;
      const s = shapes[sel];
      if (!s || TYPES[s.type].fixed) return;
      e.preventDefault();
      const d = (e.deltaY || e.deltaX) > 0 ? -1 : 1;
      if (e.shiftKey) s.rz = Math.round((s.rz + d * 5) * 100) / 100;
      else s.s = Math.max(0.05, +(s.s * (d > 0 ? 1.05 : 1 / 1.05)).toFixed(4));
      changed();
    }, { passive: false });
    wrap.addEventListener("keydown", (e) => {
      if (!active()) return;
      const s = shapes[sel];
      if (!s) return;
      const M = mesh();
      const k = e.key;
      if (k === "Delete" || k === "Backspace") { remove(sel); e.preventDefault(); return; }
      if ((k === "D" || k === "d") && e.shiftKey) { duplicate(sel); e.preventDefault(); return; }
      if (TYPES[s.type].fixed) return;
      const step = e.shiftKey ? 10 : 1;
      const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
      if (moves[k]) {
        s.x += moves[k][0] * M.cell[0] * step; s.y += moves[k][1] * M.cell[1] * step;
        changed(); e.preventDefault();
      } else if (k === "r" || k === "R") { s.rz += e.shiftKey ? -15 : 15; changed(); e.preventDefault(); }
    });
  }

  // ---------------------------------------------------------------- persistence
  function save() { return { shapes, uid }; }
  function load(st) {
    if (!st || !Array.isArray(st.shapes)) return;
    shapes = st.shapes.filter((s) => TYPES[s.type]).map((s) => ({ rep: [0, 0], s: 1, rx: 0, ry: 0, rz: 0, z: 0, ...s }));
    uid = st.uid || shapes.length + 1;
    sel = shapes.length ? 0 : -1;
    renderList(); renderEditor();
  }
  function reset(list) { shapes = []; sel = -1; for (const t of list || []) add(t); renderList(); renderEditor(); }

  function init(cb) {
    onChange = cb;
    $("#addShape").addEventListener("change", (e) => { if (e.target.value) add(e.target.value); e.target.value = ""; });
    renderList();
    renderEditor();
  }

  function exprById(id, lang) { const s = shapes.find((q) => q.id === id); return s ? shapeExpr(s, lang) : null; }

  return { init, code, needsPython, exprById, mesh, draw, attach, requestPreview, previewBody, save, load, reset, regionsUsed,
    get shapes() { return shapes; }, get selected() { return shapes[sel]; } };
})();
