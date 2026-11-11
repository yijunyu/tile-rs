// Glue for the tile-rs wasm bundle. Hand-written because the alternative is
// wasm-bindgen, which is a build prerequisite `tile` will not acquire silently.
//
// The contract is four exports over linear memory: alloc, dealloc, filter, result_len.
// Everything here is defensive — if the module fails to load for any reason the table
// stays exactly as the server rendered it, because a filter box that breaks the page is
// worse than no filter box.
(function () {
  "use strict";
  var mod = null;

  function bytes(mem, ptr, len) {
    return new Uint8Array(mem.buffer, ptr, len);
  }

  function put(m, s) {
    var enc = new TextEncoder().encode(s);
    if (enc.length === 0) return { ptr: 0, len: 0 };
    var p = m.exports.alloc(enc.length);
    if (p === 0) return { ptr: 0, len: 0 };
    bytes(m.exports.memory, p, enc.length).set(enc);
    return { ptr: p, len: enc.length };
  }

  function filter(rows, needle) {
    if (!mod) return null;
    var a = put(mod, rows), b = put(mod, needle);
    try {
      var p = mod.exports.filter(a.ptr, a.len, b.ptr, b.len);
      var n = mod.exports.result_len();
      var out = new TextDecoder().decode(bytes(mod.exports.memory, p, n).slice());
      return JSON.parse(out);
    } catch (e) {
      return null;
    } finally {
      if (a.ptr) mod.exports.dealloc(a.ptr, a.len);
      if (b.ptr) mod.exports.dealloc(b.ptr, b.len);
    }
  }

  function wire() {
    var box = document.getElementById("filter");
    var table = document.getElementById("forms");
    if (!box || !table) return;
    var rows = Array.prototype.slice.call(table.tBodies[0].rows);
    var text = rows.map(function (r) { return r.innerText.replace(/\n/g, " "); }).join("\n");
    box.disabled = false;
    box.placeholder = "filter " + rows.length + " forms (wasm)";
    box.addEventListener("input", function () {
      var keep = filter(text, box.value);
      if (keep === null) return;            // module gone: leave the table alone
      var set = new Set(keep);
      rows.forEach(function (r, i) { r.hidden = !set.has(i); });
      var n = keep.length;
      var count = document.getElementById("filter-count");
      if (count) count.textContent = n + " of " + rows.length;
    });
  }

  // instantiateStreaming needs the right MIME type; the fallback covers a server that
  // does not send it, which is a thing that silently breaks wasm everywhere.
  var req = fetch("/tile_ui.wasm");
  var go = WebAssembly.instantiateStreaming
    ? WebAssembly.instantiateStreaming(req, {})
    : req.then(function (r) { return r.arrayBuffer(); })
         .then(function (b) { return WebAssembly.instantiate(b, {}); });
  go.then(function (res) { mod = res.instance; wire(); })
    .catch(function () { /* server-rendered page stands on its own */ });
})();
