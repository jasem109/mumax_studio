"use strict";
/* OVF reader (OOMMF vector field files, as written by mumax3, mumax+, OOMMF and mumax Studio).
   Handles OVF 1.0 and 2.0, "Data Text", "Data Binary 4" and "Data Binary 8". OVF 1 binary is big-endian,
   OVF 2 little-endian; the check value at the start of the data block tells which, so both are accepted.
   Result: {nx, ny, nz, ncomp, cell:[dx,dy,dz], origin, data: Float32Array(ncomp * N) component-major,
            norm: Float32Array(N), t, title, name} with cell index i = ix + nx*(iy + ny*iz). */
const OVF = (() => {
  const dec = new TextDecoder("latin1");

  function parse(buffer, name = "") {
    const bytes = new Uint8Array(buffer);
    // the header is ASCII; find "# Begin: Data"
    const headLen = Math.min(bytes.length, 1 << 16);
    const headText = dec.decode(bytes.subarray(0, headLen));
    const m = /#\s*Begin:\s*Data\s+(Text|Binary\s*4|Binary\s*8)[^\n]*\n/i.exec(headText);
    if (!m) throw new Error(`${name || "file"}: not an OVF file (no "# Begin: Data" line)`);
    const header = headText.slice(0, m.index);
    const kind = m[1].replace(/\s+/g, "").toLowerCase();
    const dataStart = m.index + m[0].length;
    const H = {};
    let desc = [];
    for (const line of header.split(/\r?\n/)) {
      const mm = /^#\s*([^:]+):\s*(.*)$/.exec(line.trim());
      if (!mm) continue;
      const k = mm[1].trim().toLowerCase(), v = mm[2].trim();
      if (k === "desc") desc.push(v); else H[k] = v;
    }
    const version = /OVF\s*2/i.test(header.split("\n")[0]) ? 2 : 1;
    const nx = parseInt(H.xnodes, 10), ny = parseInt(H.ynodes, 10), nz = parseInt(H.znodes, 10);
    if (!(nx > 0 && ny > 0 && nz > 0)) throw new Error(`${name}: missing or bad xnodes / ynodes / znodes`);
    const ncomp = parseInt(H.valuedim || "3", 10) || 3;
    const unit = (H.meshunit || "m").toLowerCase();
    const scale = { m: 1, nm: 1e-9, um: 1e-6, mm: 1e-3, cm: 1e-2 }[unit] || 1;
    const cell = [num(H.xstepsize, 1), num(H.ystepsize, 1), num(H.zstepsize, 1)].map((v) => v * scale);
    const origin = [num(H.xmin, 0), num(H.ymin, 0), num(H.zmin, 0)].map((v) => v * scale);
    const N = nx * ny * nz, total = N * ncomp;
    const out = new Float32Array(total);
    if (kind === "text") {
      const end = findEnd(bytes, dataStart);
      const text = dec.decode(bytes.subarray(dataStart, end));
      let k = 0;
      const re = /[-+0-9.eEnNaAiIfF]+/g;
      let t;
      while (k < total && (t = re.exec(text)) !== null) {
        if (t[0][0] === "#") continue;
        out[k++] = parseFloat(t[0]);
      }
      if (k < total) throw new Error(`${name}: text data ends after ${k} of ${total} numbers`);
    } else {
      const w = kind === "binary4" ? 4 : 8;
      const need = dataStart + w * (total + 1);
      if (bytes.length < need) throw new Error(`${name}: file is truncated (${bytes.length} of ${need} bytes)`);
      const dv = new DataView(buffer, bytes.byteOffset + dataStart, w * (total + 1));
      const check = w === 4 ? 1234567.0 : 123456789012345.0;
      const rd = (off, le) => (w === 4 ? dv.getFloat32(off, le) : dv.getFloat64(off, le));
      let le = version === 2;
      if (rd(0, le) !== check) {
        le = !le;
        if (rd(0, le) !== check) throw new Error(`${name}: bad binary check value (not a valid OVF data block)`);
      }
      for (let k = 0; k < total; k++) out[k] = rd(w * (k + 1), le);
    }
    // node-major (v0x v0y v0z v1x ...) -> component-major
    const data = new Float32Array(total);
    const norm = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let c = 0; c < ncomp; c++) { const v = out[i * ncomp + c]; data[c * N + i] = v; s += v * v; }
      norm[i] = Math.sqrt(s);
    }
    let t = null;
    for (const d of desc) {
      const mm = /time\s*[:=]\s*([-+0-9.eE]+)/i.exec(d);
      if (mm) t = parseFloat(mm[1]);
    }
    return { nx, ny, nz, ncomp, cell, origin, data, norm, t, title: H.title || "", name, desc, version, kind };
  }
  function num(s, d) { const v = parseFloat(s); return isFinite(v) ? v : d; }
  function findEnd(bytes, from) {
    const s = dec.decode(bytes.subarray(from, Math.min(bytes.length, from + 400 * 1024 * 1024)));
    const i = s.search(/#\s*End:\s*Data/i);
    return i < 0 ? bytes.length : from + i;
  }

  /* the mumax Studio /api/field3d binary format, converted to the same shape */
  function fromStudio(buf) {
    const dv = new DataView(buf);
    const hl = dv.getUint32(0, true);
    const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const N = hdr.nx * hdr.ny * hdr.nz;
    const data = new Float32Array(buf, 4 + hl, hdr.ncomp * N);
    const geom = new Uint8Array(buf, 4 + hl + 4 * hdr.ncomp * N, N);
    const norm = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let c = 0; c < hdr.ncomp; c++) { const v = data[c * N + i]; s += v * v; }
      norm[i] = geom[i] ? Math.sqrt(s) : 0;
    }
    return { nx: hdr.nx, ny: hdr.ny, nz: hdr.nz, ncomp: hdr.ncomp, cell: hdr.cell, origin: [0, 0, 0], data, norm,
             geom, t: hdr.t, title: hdr.name, name: hdr.name, unit: hdr.unit, stride: hdr.stride, fullN: hdr.n };
  }
  return { parse, fromStudio };
})();
