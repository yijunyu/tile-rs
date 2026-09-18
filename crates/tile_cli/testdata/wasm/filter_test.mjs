// Behavioural test for the tile --ui wasm bundle, run by tile_spec through `node`.
//
// The Rust side can only check the module's SHAPE -- magic number, version, export names
// -- because `tile` has no wasm runtime and will not grow one. Shape is not behaviour: a
// bundle that loads and returns nothing useful passes every structural check and leaves a
// filter box that silently does nothing, which is precisely the class of defect this
// project keeps finding. So the behaviour is asserted where a runtime already exists.
//
// Exits non-zero on the first disagreement; the spec step reports its output verbatim.
import { readFileSync } from "node:fs";
const bytes = readFileSync(process.argv[2]);
const { instance } = await WebAssembly.instantiate(bytes, {});
const x = instance.exports;
const mem = () => new Uint8Array(x.memory.buffer);

function put(s) {
  const e = new TextEncoder().encode(s);
  if (!e.length) return { ptr: 0, len: 0 };
  const p = x.alloc(e.length);
  if (p === 0) throw new Error("alloc refused " + e.length);
  mem().set(e, p);
  return { ptr: p, len: e.length };
}
function filter(rows, needle) {
  const a = put(rows), b = put(needle);
  const p = x.filter(a.ptr, a.len, b.ptr, b.len);
  const n = x.result_len();
  const out = new TextDecoder().decode(mem().slice(p, p + n));
  if (a.ptr) x.dealloc(a.ptr, a.len);
  if (b.ptr) x.dealloc(b.ptr, b.len);
  return JSON.parse(out);
}

const rows = ["tile source 3 .rs yes", "mlir mlir 2 .mlir yes", "msl source 1 .metal no",
              "gpu source 1 .cu no", "PICO source 1 .pico.s no"].join("\n");
let fail = 0;
const eq = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) { fail++; console.log("FAIL", name, "got", JSON.stringify(got), "want", JSON.stringify(want)); }
  else console.log("ok  ", name, JSON.stringify(got));
};

eq("empty needle matches all", filter(rows, ""), [0,1,2,3,4]);
eq("substring", filter(rows, "metal"), [2]);
eq("case-insensitive needle", filter(rows, "MLIR"), [1]);
eq("case-insensitive row", filter(rows, "pico"), [4]);      // row is 'PICO'
eq("multiple matches", filter(rows, "source"), [0,2,3,4]);
eq("no match", filter(rows, "zzzz"), []);
eq("empty rows", filter("", "x"), []);
// Repeated calls must not corrupt the stored result.
for (let i = 0; i < 200; i++) filter(rows, i % 2 ? "s" : "mlir");
eq("stable after 200 calls", filter(rows, "metal"), [2]);
// Alloc cap is enforced rather than trapping.
console.log("alloc(0) =", x.alloc(0), " alloc(too big) =", x.alloc(64 << 20));
// A needle with non-ASCII must not trap.
eq("unicode needle", filter(rows, "métal"), []);
console.log(fail === 0 ? "\nALL PASS" : `\n${fail} FAILED`);
process.exit(fail === 0 ? 0 : 1);
