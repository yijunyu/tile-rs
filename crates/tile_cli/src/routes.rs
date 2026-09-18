//! The transformation graph, as data.
//!
//! Nodes are forms; edges carry a kind (derived from the level difference), a capability
//! requirement, a cost and a fidelity class. Route selection is a shortest path over the
//! subgraph whose capabilities are satisfiable right now. `--route`, `--via`, `-O`, the
//! daemon and the UI all read this one graph — if routes lived in the argument parser,
//! each of those would reimplement them.
//!
//! Two rules that are not obvious and are load-bearing:
//!
//! * **Lifts do not auto-compose into lowering routes.** `msl -> cpp` is refused even
//!   where an `msl -> tile` lifter exists; the composed route must be named with
//!   `--via`. A synthesised intermediate silently feeding a lowering is exactly the
//!   invisible wrongness this tool exists to prevent.
//! * **Same-form optimization needs a READER.** `msl -> msl` would mean writing a Metal
//!   frontend. tile-rs has 1.5 readers and 16 writers, and the graph says so.

use crate::forms::{self, Evidence, Fidelity, Kind};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

/// What an edge needs before it can be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// Nothing: a pure in-process string transform.
    Nothing,
    /// A provisioned toolchain (M4/M6).
    Toolchain(&'static str),
    /// A target compiled into THIS build. Distinct from "no such form", and the
    /// distinction is the whole difference between a packaging bug and a typo.
    Feature(&'static str),
    /// Real hardware of a family.
    Device(&'static str),
    /// A READER for this form — a frontend that turns the form back into IR.
    ///
    /// This is the one need nobody can satisfy by installing, buying or rebuilding: it
    /// is a piece of work that has not been written. tile-rs has 3 readers and 19
    /// writers, so most of the graph's missing edges are this, and the tool must say so
    /// in that tense rather than reporting a flat "no route".
    Frontend(&'static str),
}

/// Whether the caller can do something about a missing need today.
///
/// This is the axis exit codes are grouped on: [`Availability::Acquirable`] means retry
/// after acquiring, [`Availability::Unwritten`] means retrying is pointless until
/// somebody builds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    Acquirable,
    Unwritten,
}

impl Need {
    pub fn availability(&self) -> Availability {
        match self {
            // Nothing to do; a satisfied need never reaches here.
            Need::Nothing => Availability::Acquirable,
            // Install it, rebuild with it, or plug the hardware in.
            Need::Toolchain(_) | Need::Feature(_) | Need::Device(_) => Availability::Acquirable,
            // Somebody has to write it.
            Need::Frontend(_) => Availability::Unwritten,
        }
    }
}

impl fmt::Display for Need {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Need::Nothing => write!(f, "nothing"),
            Need::Toolchain(t) => write!(f, "toolchain:{t}"),
            Need::Feature(x) => write!(f, "feature:{x}"),
            Need::Device(d) => write!(f, "device:{d}"),
            Need::Frontend(x) => write!(f, "frontend:{x}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: &'static str,
    pub to: &'static str,
    pub need: Need,
    pub cost: u32,
    pub fidelity: Fidelity,
}

impl Edge {
    pub fn kind(&self) -> Kind {
        let f = forms::by_id(self.from).expect("edge from");
        let t = forms::by_id(self.to).expect("edge to");
        forms::kind_of(f, t)
    }
}

const RUSTC_SO: Need = Need::Toolchain("rustc_codegen_tile");
const ASCENDC_TO_RS: Need = Need::Toolchain("ascendc-to-rs");

/// The capability names the graph can ask for.
///
/// `Caps` holds `&'static str` while an `Env` holds `String`s that may come from a
/// user's `TILE_SIMULATE`. This table is where the two meet, so a typo in a spec can
/// never masquerade as a capability the build does not have.
const FEATURE_NAMES: &[&str] = &["emitters", "ascend", "lift", "pico"];

/// How a caller could obtain a capability this build lacks — which is what the exit code
/// is promising, so it has to be true.
///
/// Both wrong answers were live: `feature:lift` and `feature:ascend` each reported "exit
/// 4, rebuild with `--features X`", and NEITHER feature is declared in Cargo.toml, so
/// that command fails with "the package does not contain this feature". They fail for
/// different reasons, though, and collapsing them loses the useful half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HowToGet {
    /// Declared in `Cargo.toml`: rebuilding this tree really does turn it on.
    Rebuild,
    /// The code exists but not in this repository — a closed-source emitter. A release
    /// binary can have it; a rebuild from this tree cannot produce it.
    AnotherBuild,
    /// Nobody has written it. No command, rebuild or download changes that today.
    Unwritten,
}

/// Features `Cargo.toml` declares. Kept honest by
/// `every_advertised_feature_is_really_a_cargo_feature`, which parses the manifest.
pub const BUILDABLE_FEATURES: &[&str] = &["emitters", "pico", "stats"];

/// Features whose code lives outside this repository.
///
/// `ascend` gates the AscendC emitter, and `crates/rustc_codegen_tile/src/` contains no
/// `mlir_to_cce.rs` -- 15 emitters are open and that one is not. So "exists but not here"
/// is exactly right, and the remedy is a build that includes it, never a rebuild of this.
pub const CLOSED_SOURCE_FEATURES: &[&str] = &["ascend"];

pub fn how_to_get(f: &str) -> HowToGet {
    if BUILDABLE_FEATURES.contains(&f) {
        HowToGet::Rebuild
    } else if CLOSED_SOURCE_FEATURES.contains(&f) {
        HowToGet::AnotherBuild
    } else {
        HowToGet::Unwritten
    }
}
const TOOLCHAIN_NAMES: &[&str] = &["rustc_codegen_tile", "llvm-20", "ascendc-to-rs"];

/// Every edge that exists. Adding a 17th backend is a `Form` row plus a row here.
pub fn edges() -> Vec<Edge> {
    let mut v = Vec::new();

    // The one reader that needs a toolchain: tile-rs source -> MLIR, through the
    // prebuilt codegen backend. Everything else downstream is a pure string transform.
    v.push(Edge {
        from: "tile",
        to: "mlir",
        need: RUSTC_SO,
        cost: 10,
        fidelity: Fidelity::Exact,
    });

    // MLIR -> target source. The open emitters, in-process, no LLVM.
    for id in [
        "msl", "gpu", "musa", "spirv", "nki", "aie", "tpu", "bang", "gaudi", "hexagon", "ttmetal",
        "csl", "linalg", "rvv", "debug",
    ] {
        let f = forms::by_id(id).expect("form");
        v.push(Edge {
            from: "mlir",
            to: id,
            need: Need::Feature("emitters"),
            cost: 1,
            fidelity: f.fidelity,
        });
    }

    // PICO is the 16th target and lives in a sibling repository (tile-rs-pico). Its
    // emitter takes the uniform `convert_mlir_to_*` signature and imports nothing but
    // std, so wiring it in is one `#[path]` include -- gated behind `pico` because the
    // checkout is not part of this tree.
    {
        let f = forms::by_id("pico").expect("form");
        v.push(Edge {
            from: "mlir",
            to: "pico",
            need: Need::Feature("pico"),
            cost: 1,
            fidelity: f.fidelity,
        });
    }

    // `pto` is NOT closed. Its emitter is `crates/rustc_codegen_tile/src/mlir_to_pto.rs`,
    // it is compiled into every build, and `emit::emitter_exists("pto")` returns true
    // unconditionally -- yet this table demanded `feature:ascend` for it, so
    // `tile k.mlir -t pto` refused with "exists, but not in this build" while the
    // emitter sat right there. The refusal is the expensive kind: exit 4 says "get
    // another build", so a caller stops rather than retrying, and the PTO path is the
    // one that reaches AscendC through ptoas.
    //
    // `cpp` really is closed: there is no `mlir_to_cce.rs` in this repository and
    // `emitter_exists` does not list it. One of the two belonged here, not both.
    {
        let f = forms::by_id("pto").expect("form");
        v.push(Edge {
            from: "mlir",
            to: "pto",
            need: Need::Feature("emitters"),
            cost: 1,
            fidelity: f.fidelity,
        });
    }

    // The closed target registers the same way; it is absent from a default build,
    // and the planner must say "not compiled into this build", not "no such form".
    {
        let id = "cpp";
        let f = forms::by_id(id).expect("form");
        v.push(Edge {
            from: "mlir",
            to: id,
            need: Need::Feature("ascend"),
            cost: 1,
            fidelity: f.fidelity,
        });
    }

    // rvv has no translator of its own -- it wraps the linalg egress and stamps the
    // RISC-V triple -- but that wrapping happens INSIDE `convert_mlir_to_rvv`, which
    // consumes the same tile MLIR every other emitter does. It is therefore a peer edge
    // out of `mlir`, above, and NOT a hop out of `linalg`.
    //
    // Modelling it the other way looked right and was wrong: `linalg -> rvv` planned
    // cleanly and then failed at hop 2 with "No entry-point kernel functions found",
    // because linalg output has no `hacc.entry` for the rvv emitter to find. The form
    // table's `rvv refines linalg` is about IDENTIFYING a file; it says nothing about
    // how one is produced, and conflating the two cost a working route.

    // Same-form optimization, for the two forms tile-rs can read.
    v.push(Edge {
        from: "mlir",
        to: "mlir",
        need: Need::Nothing,
        cost: 1,
        fidelity: Fidelity::Exact,
    });
    v.push(Edge {
        from: "tile",
        to: "tile",
        need: RUSTC_SO,
        cost: 5,
        fidelity: Fidelity::Exact,
    });

    // Lifts. Deliberately almost empty: lifting is program synthesis, not the reverse
    // edge with a different cost. Each one that arrives is justified individually.
    v.push(Edge {
        from: "pto",
        to: "tile",
        need: Need::Feature("lift"),
        cost: 50,
        fidelity: Fidelity::Synthesised,
    });
    // AscendC back to the DSL. Justified individually, as the comment above asks:
    // this one has an implementation behind it, which `pto -> tile` still does not.
    //
    // A TOOLCHAIN rather than a feature, because the lifter is an external program.
    // See `lift_cpp` for why it is not compiled in -- briefly, this crate's whole
    // discipline is one dependency, and a vendored second copy of a 728 KB `lower.rs`
    // would drift from the first.
    v.push(Edge {
        from: "cpp",
        to: "tile",
        need: ASCENDC_TO_RS,
        cost: 50,
        fidelity: Fidelity::Synthesised,
    });

    v
}

/// What this build can actually do right now.
#[derive(Clone, Debug, Default)]
pub struct Caps {
    pub features: Vec<&'static str>,
    pub toolchains: Vec<&'static str>,
    pub devices: Vec<String>,
}

impl Caps {
    /// The capabilities of the binary as compiled, before any provisioning.
    /// The capabilities of the binary as compiled, before any provisioning.
    ///
    /// Prefer [`Caps::from_env`] on any path a user reaches: reading `cfg!` directly at
    /// the point of use makes the missing-capability branches untestable, which is
    /// precisely how they rot until a user on a bare machine finds them.
    pub fn of_this_build() -> Caps {
        Caps::from_env(&crate::simenv::Env::detect())
    }

    /// Capabilities derived from an [`Env`](crate::simenv::Env) — real or simulated.
    ///
    /// This is the seam that lets `TILE_SIMULATE=no-emitters` exercise the "not compiled
    /// into this build" refusal on a machine where it very much is compiled in.
    pub fn from_env(env: &crate::simenv::Env) -> Caps {
        let mut toolchains: Vec<&'static str> = TOOLCHAIN_NAMES
            .iter()
            .filter(|t| env.toolchains.iter().any(|x| x == *t))
            .copied()
            .collect();
        // A provisioned backend on disk IS the toolchain, whether or not anything told
        // the environment about it. Asking the filesystem is the honest check; requiring
        // a separate declaration would refuse a route that would in fact work.
        if env.vendor_clis
            && !toolchains.contains(&"rustc_codegen_tile")
            && crate::lower_rs::available()
        {
            toolchains.push("rustc_codegen_tile");
        }
        // Same rule for the AscendC lifter: a binary on disk IS the toolchain.
        if env.vendor_clis && !toolchains.contains(&"ascendc-to-rs") && crate::lift_cpp::available()
        {
            toolchains.push("ascendc-to-rs");
        }
        Caps {
            features: FEATURE_NAMES
                .iter()
                .filter(|f| env.has_feature(f))
                .copied()
                .collect(),
            toolchains,
            devices: env.devices.clone(),
        }
    }
    pub fn satisfies(&self, n: &Need) -> bool {
        match n {
            Need::Nothing => true,
            Need::Feature(f) => self.features.contains(f),
            Need::Toolchain(t) => self.toolchains.contains(t),
            Need::Device(d) => self.devices.iter().any(|x| x == d),
            // No build satisfies this. It is satisfiable only by writing the reader.
            Need::Frontend(_) => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Route {
    pub hops: Vec<Edge>,
}

impl Route {
    pub fn forms(&self) -> Vec<&'static str> {
        let mut v = vec![self.hops[0].from];
        v.extend(self.hops.iter().map(|h| h.to));
        v
    }
    pub fn cost(&self) -> u32 {
        self.hops.iter().map(|h| h.cost).sum()
    }
    /// The route's fidelity: the weakest *doubt* any hop raises, and — if none does — the
    /// hardware the OUTPUT was validated on.
    ///
    /// "Weakest hop wins" is right for the two classes that express doubt: one synthesised
    /// hop makes the whole route synthesised, and one hop never run on its target makes it
    /// unvalidated. It is NOT right for `Exact` against `Validated`, and the distinction is
    /// worth stating because the two look like neighbours on a ladder and are not.
    /// `Exact` is a claim about the emit path being mechanically checked; every mlir->mlir
    /// pass is Exact and none of them could be anything else, because a compiler pass has
    /// no hardware to run on. Degrading to Exact whenever a route contains a pass would
    /// mean no route through the optimizer could ever report where its output was
    /// validated, which is the fact worth printing.
    ///
    /// When two hops are both validated, the LAST one wins, not the first. The claim is
    /// about the artifact the caller ends up holding, so it must name the hardware at the
    /// end of the route rather than somewhere in the middle. No route reaches this today —
    /// `linalg`, the only validated form that is also readable, has no outgoing edges — but
    /// picking the first was an arbitrary answer to a question with a right one.
    pub fn fidelity(&self) -> Fidelity {
        let mut worst = Fidelity::Exact;
        for h in &self.hops {
            worst = match (worst, h.fidelity) {
                (Fidelity::Synthesised, _) | (_, Fidelity::Synthesised) => Fidelity::Synthesised,
                (Fidelity::Unvalidated, _) | (_, Fidelity::Unvalidated) => Fidelity::Unvalidated,
                (Fidelity::Validated(a, ea), Fidelity::Exact) => Fidelity::Validated(a, ea),
                (Fidelity::Exact, Fidelity::Validated(b, eb)) => Fidelity::Validated(b, eb),
                // #012. The hardware name is the last hop's -- that is the machine the
                // route ends on. The EVIDENCE is not: a route is only recorded-here if
                // every validated hop in it is, so one inherited claim anywhere makes the
                // whole route's claim inherited. Taking the last hop's evidence would let
                // a route that passes through an unattributed backend end up claiming a
                // file that says nothing about it.
                (Fidelity::Validated(_, ea), Fidelity::Validated(b, eb)) => {
                    let ev = match (ea, eb) {
                        (Evidence::Here(_), Evidence::Here(w)) => Evidence::Here(w),
                        _ => Evidence::Inherited,
                    };
                    Fidelity::Validated(b, ev)
                }
                (Fidelity::Exact, Fidelity::Exact) => Fidelity::Exact,
            };
        }
        worst
    }
    /// Capabilities this route needs that the caller does not have.
    pub fn missing(&self, caps: &Caps) -> Vec<Need> {
        self.hops
            .iter()
            .map(|h| h.need)
            .filter(|n| !caps.satisfies(n))
            .collect()
    }
    pub fn describe(&self) -> String {
        format!("{}  [{}]", self.forms().join(" -> "), self.fidelity())
    }
}

#[derive(Debug)]
pub enum RouteError {
    /// Nothing leaves `from` because tile-rs cannot READ it — there is no frontend for
    /// that form. Carries the route that WOULD work if somebody wrote one, so the
    /// refusal names a missing piece of work instead of implying impossibility.
    NeedsFrontend {
        from: &'static str,
        to: &'static str,
        would_be: Option<Route>,
    },
    /// No path at all, and a reader is not what is missing.
    NoRoute {
        from: &'static str,
        to: &'static str,
        nearest: Vec<Route>,
    },
    /// A path exists and is takeable; the tool declines to take it unasked because it
    /// passes through a lift. A different command works right now.
    OnlyThroughLift {
        from: &'static str,
        to: &'static str,
        via: Vec<&'static str>,
    },
    /// `--via` named forms that no available route visits.
    ViaNotOnAnyRoute {
        from: &'static str,
        to: &'static str,
        via: Vec<String>,
        available: Vec<String>,
    },
}

/// All routes from `from` to `to`, cheapest first.
///
/// `allow_lifts` names the forms the user explicitly pinned with `--via`; a lift into a
/// form not on that list is never taken automatically.
pub fn routes(from: &str, to: &str, allow_lifts: &[String]) -> Vec<Route> {
    routes_with(&[], from, to, allow_lifts)
}

/// `routes`, plus edges that do NOT exist — used to answer "what would this take?".
///
/// The graph itself stays honest: an edge is in it only if it is real. A refusal that
/// merely says "no" teaches nothing, so the refusal path re-runs the search with the
/// missing capability hypothetically present and reports the route that WOULD work.
pub fn routes_with(extra: &[Edge], from: &str, to: &str, allow_lifts: &[String]) -> Vec<Route> {
    let mut all = edges();
    all.extend_from_slice(extra);
    let mut adj: BTreeMap<&'static str, Vec<Edge>> = BTreeMap::new();
    for e in &all {
        adj.entry(e.from).or_default().push(*e);
    }

    let mut found: Vec<Route> = Vec::new();
    // Breadth-first over simple paths. The graph is ~25 edges and shallow; an
    // exhaustive walk is both affordable and easier to reason about than Dijkstra
    // with tie-breaking, and it terminates because a form is never revisited.
    let mut queue: VecDeque<(Vec<Edge>, Vec<&'static str>)> = VecDeque::new();
    queue.push_back((Vec::new(), vec![from_static(from)]));
    while let Some((hops, seen)) = queue.pop_front() {
        let here = *seen.last().unwrap();
        if !hops.is_empty() && here == to {
            found.push(Route { hops });
            continue;
        }
        if hops.len() >= 4 {
            continue;
        }
        for e in adj.get(here).into_iter().flatten() {
            // Self-edges (optimization) are only meaningful as a whole route.
            if e.from == e.to && !(hops.is_empty() && from == to) {
                continue;
            }
            if e.from != e.to && seen.contains(&e.to) {
                continue;
            }
            if e.kind() == Kind::Lift && !allow_lifts.iter().any(|v| v == e.to) {
                continue;
            }
            let mut h2 = hops.clone();
            h2.push(*e);
            let mut s2 = seen.clone();
            if e.from != e.to {
                s2.push(e.to);
            }
            queue.push_back((h2, s2));
        }
    }
    found.sort_by_key(|r| (r.cost(), r.hops.len()));
    found
}

fn from_static(s: &str) -> &'static str {
    forms::by_id(s).map(|f| f.id).unwrap_or("mlir")
}

/// Plan a route with an optimization prologue.
///
/// At `-O1` and above, a readable input gets a same-form optimize hop BEFORE the
/// lowering. That is not bookkeeping: it makes the optimization a real, nameable,
/// keepable step on the route, so `--route` shows it, `-k` writes it out, and a user can
/// diff what the passes did rather than taking the report's word for it.
///
/// A form tile-rs cannot read gets no prologue — there is nothing to optimize with.
pub fn plan_with_opt(
    from: &str,
    to: &str,
    allow_lifts: &[String],
    level: u8,
) -> Result<Route, RouteError> {
    Ok(with_opt_prologue(plan(from, to, allow_lifts)?, level))
}

/// Put the optimize hop on the front of a route, when the level calls for one and the
/// input form can be read.
///
/// Separate from `plan_with_opt` so `--route` can apply the same transformation to every
/// candidate it lists. A `--route` that shows the bare lowering while the run inserts an
/// optimize hop is a different answer from `-o`, which is the one thing it must not be.
pub fn with_opt_prologue(mut route: Route, level: u8) -> Route {
    let Some(first) = route.hops.first().copied() else {
        return route;
    };
    let readable = forms::by_id(first.from).is_some_and(|f| f.readable);
    if level >= 1 && readable && first.from != first.to {
        route.hops.insert(
            0,
            Edge {
                from: first.from,
                to: first.from,
                need: Need::Nothing,
                cost: 0,
                fidelity: Fidelity::Exact,
            },
        );
    }
    route
}

/// Does this route actually pass through every form the user pinned, in order?
///
/// `--via` had only ever meant "a lift into one of these is permitted". The help text
/// said "pin the intermediate forms", so `--via linalg` on a pair with a direct edge was
/// accepted and then ignored: the route printed was the unpinned one, and nothing said
/// the instruction had been dropped. A flag that is silently a no-op is worse than one
/// that refuses, because the user believes they constrained the run.
fn passes_through(route: &Route, via: &[String]) -> bool {
    let mut want = via.iter();
    let Some(mut next) = want.next() else {
        return true;
    };
    for h in &route.hops {
        if h.to == next.as_str() {
            match want.next() {
                Some(n) => next = n,
                None => return true,
            }
        }
    }
    false
}

/// Plan a route, or explain precisely why there is none.
pub fn plan(from: &str, to: &str, allow_lifts: &[String]) -> Result<Route, RouteError> {
    let rs = routes(from, to, allow_lifts);
    // Honour the pin. A named form that no route visits is a contradiction between the
    // command and the graph, and the remedy is to type something else -- which is what
    // NoRoute reports, with the routes that DO exist.
    let pinned: Vec<Route> = rs
        .iter()
        .filter(|r| passes_through(r, allow_lifts))
        .cloned()
        .collect();
    if !allow_lifts.is_empty() && pinned.is_empty() && !rs.is_empty() {
        return Err(RouteError::ViaNotOnAnyRoute {
            from: from_static(from),
            to: from_static(to),
            via: allow_lifts.to_vec(),
            available: rs
                .iter()
                .map(|r| r.forms().join(" -> "))
                .collect::<Vec<_>>(),
        });
    }
    if let Some(r) = pinned.into_iter().next() {
        return Ok(r);
    }
    let rs = routes(from, to, allow_lifts);
    if let Some(r) = rs.into_iter().next() {
        return Ok(r);
    }
    // Would it exist if lifts composed? If so, say which lift, and that the user must
    // ask for it by name.
    let lift_targets: Vec<String> = forms::FORMS.iter().map(|f| f.id.to_string()).collect();
    let with_lifts = routes(from, to, &lift_targets);
    if let Some(r) = with_lifts.first() {
        let via: Vec<&'static str> = r
            .hops
            .iter()
            .filter(|h| h.kind() == Kind::Lift)
            .map(|h| h.to)
            .collect();
        return Err(RouteError::OnlyThroughLift {
            from: from_static(from),
            to: from_static(to),
            via,
        });
    }
    // Is a READER what is missing? For most form pairs in this tool it is: 16 of the 19
    // forms are write-only. Re-run the search with a hypothetical frontend so the answer
    // can name the work that would unlock it, in the right tense.
    let ff = forms::by_id(from).map(|f| f.id).unwrap_or("mlir");
    if let Some(form) = forms::by_id(from) {
        if !form.readable {
            let hypothetical = Edge {
                from: form.id,
                to: "mlir",
                need: Need::Frontend(form.id),
                cost: 20,
                fidelity: Fidelity::Synthesised,
            };
            let lift_all: Vec<String> = forms::FORMS.iter().map(|f| f.id.to_string()).collect();
            let would_be = if from == to {
                // A round trip. The search refuses to revisit a form, correctly — but
                // same-form optimization IS a round trip through the IR, so the route
                // that would unlock it has to be built rather than searched for.
                edges()
                    .iter()
                    .find(|e| e.from == "mlir" && e.to == form.id)
                    .map(|back| Route {
                        hops: vec![hypothetical, *back],
                    })
            } else {
                routes_with(&[hypothetical], from, to, &lift_all)
                    .into_iter()
                    .next()
            };
            return Err(RouteError::NeedsFrontend {
                from: ff,
                to: from_static(to),
                would_be,
            });
        }
    }

    // Nearest routes, so the refusal teaches instead of just refusing. For most form
    // pairs in this tool the refusal IS the product.
    let mut nearest = Vec::new();
    for f in forms::FORMS {
        if f.id == to {
            continue;
        }
        if let Some(r) = routes(from, f.id, allow_lifts).into_iter().next() {
            nearest.push(r);
        }
    }
    nearest.sort_by_key(|r| r.cost());
    nearest.truncate(3);
    Err(RouteError::NoRoute {
        from: from_static(from),
        to: from_static(to),
        nearest,
    })
}

/// The reader/writer matrix, for `tile --list-forms`.
///
/// A sparse graph is fine. A sparse *undiscoverable* graph is what makes a tool feel
/// broken, and this is the one command that makes the shape visible.
pub fn list_forms() -> String {
    let all = edges();
    let mut s = String::new();
    s.push_str(
        "form      role        lvl  ext         read  write  optimize  lifts-from  fidelity\n",
    );
    s.push_str(
        "--------  ----------  ---  ----------  ----  -----  --------  ----------  --------\n",
    );
    for f in forms::FORMS {
        let lifts: Vec<&str> = all
            .iter()
            .filter(|e| e.to == f.id && e.kind() == Kind::Lift)
            .map(|e| e.from)
            .collect();
        s.push_str(&format!(
            "{:<8}  {:<10}  {:>3}  .{:<9}  {:<4}  {:<5}  {:<8}  {:<10}  {}\n",
            f.id,
            f.role.to_string(),
            f.level,
            f.primary_ext(),
            if f.readable { "yes" } else { "no" },
            if f.writable { "yes" } else { "no" },
            if f.can_optimize_in_place() {
                "yes"
            } else {
                "no"
            },
            if lifts.is_empty() {
                "-".to_string()
            } else {
                lifts.join(",")
            },
            f.fidelity
        ));
    }
    // Counted, never written down: a hardcoded "reads 2, writes 18" drifts the moment a
    // form is added, and a footer that lies about the shape of the graph is worse than no
    // footer at all.
    let readable: Vec<&str> = forms::FORMS
        .iter()
        .filter(|f| f.readable)
        .map(|f| f.id)
        .collect();
    let writable = forms::FORMS.iter().filter(|f| f.writable).count();
    let source = forms::FORMS
        .iter()
        .filter(|f| f.role == forms::Role::Source)
        .count();
    s.push_str(&format!(
        "\ntile-rs reads {} forms and writes {writable}. This is a fan-out with pandoc's\n\
         grammar, not pandoc's capability: most forms are write-only, and a conversion INTO\n\
         one of them is a one-way trip. Same-form optimization needs a reader, so it exists\n\
         for {} only.\n",
        readable.len(),
        readable.join(", ")
    ));
    // The role column says WHY that shape is a design and not a shortfall. Without it the
    // 16 `read: no` rows read as 16 unwritten readers, which is how they were being
    // recorded -- four backlog issues in one area, all of them the same category error.
    s.push_str(&format!(
        "\nrole says who consumes a form, which is what decides whether a missing reader is\n\
         a gap. {source} `source` forms are handed to a downstream toolchain and owe no\n\
         reader; the `mlir` forms are the ones tile-rs passes process, so those must stay\n\
         readable. No form is `binary` yet -- nothing tile-rs writes can be loaded and run\n\
         without a compiler, which is why `-r` builds its kernel at runtime.\n"
    ));
    s
}

#[cfg(test)]
mod tests {

    /// What a multi-hop route is allowed to claim about hardware.
    ///
    /// The two classes that express doubt dominate: a synthesised or unvalidated hop
    /// anywhere makes the whole route that. `Exact` does not, because every mlir->mlir
    /// pass is Exact and a compiler pass has no hardware to be validated on — degrading on
    /// it would mean no optimized route could ever name where its output was checked.
    ///
    /// And when two hops are validated the LAST wins: the claim describes the artifact the
    /// caller ends up with. No route reaches that case today, which is exactly why it was
    /// answered arbitrarily until now.
    #[test]
    fn a_route_names_the_hardware_its_output_was_validated_on() {
        let hop = |fid| Edge {
            from: "mlir",
            to: "msl",
            need: Need::Nothing,
            cost: 1,
            fidelity: fid,
        };
        let route = |fids: Vec<Fidelity>| Route {
            hops: fids.into_iter().map(hop).collect(),
        };

        // A pass followed by a hardware-validated emit still names the hardware.
        assert_eq!(
            route(vec![
                Fidelity::Exact,
                Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"))
            ])
            .fidelity(),
            Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"))
        );
        // Order does not matter for that.
        assert_eq!(
            route(vec![
                Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md")),
                Fidelity::Exact
            ])
            .fidelity(),
            Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"))
        );
        // Two validated hops: the one at the END of the route is the artifact's.
        assert_eq!(
            route(vec![
                Fidelity::Validated("CPU", Evidence::Here("docs/cli/INTEGRATION.md")),
                Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"))
            ])
            .fidelity(),
            Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md")),
            "the claim is about the output, so it names the last hardware, not the first"
        );
        // Doubt dominates, from either position.
        assert_eq!(
            route(vec![
                Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md")),
                Fidelity::Unvalidated
            ])
            .fidelity(),
            Fidelity::Unvalidated
        );
        assert_eq!(
            route(vec![
                Fidelity::Synthesised,
                Fidelity::Validated("Apple GPU", Evidence::Here("docs/cli/INTEGRATION.md"))
            ])
            .fidelity(),
            Fidelity::Synthesised
        );
    }
    use super::*;

    #[test]
    fn every_advertised_feature_is_really_a_cargo_feature() {
        // The tool tells people to run `cargo build --features X`. If X is not declared,
        // that command fails and the exit code lied about being acquirable.
        let manifest = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
        )
        .expect("Cargo.toml");
        let features = manifest
            .split("[features]")
            .nth(1)
            .expect("a [features] section");
        let declared: Vec<&str> = features
            .lines()
            .take_while(|l| !l.trim_start().starts_with('['))
            .filter_map(|l| l.split('=').next())
            .map(str::trim)
            .filter(|n| !n.is_empty() && !n.starts_with('#'))
            .collect();
        for f in BUILDABLE_FEATURES {
            assert!(
                declared.contains(f),
                "{f} is advertised as buildable but Cargo.toml declares {declared:?}"
            );
        }
        // The other two must NOT be, or the rebuild instruction is a dead end again.
        for f in CLOSED_SOURCE_FEATURES {
            assert!(
                !declared.contains(f),
                "{f} is declared here, so it is not closed-source any more"
            );
        }
        assert_eq!(how_to_get("lift"), HowToGet::Unwritten);
        // And the cpp lift is deliberately NOT that: it is a toolchain, so the
        // refusal is exit 4 with a command, not exit 3 with an apology.
        assert_eq!(how_to_get("ascend"), HowToGet::AnotherBuild);
        assert_eq!(how_to_get("pico"), HowToGet::Rebuild);
    }

    /// The AscendC lift is the first one with an implementation behind it. It must be
    /// a TOOLCHAIN need -- acquirable, exit 4, with a command that works -- and never a
    /// `Feature`, which is the shape that means "nobody wrote it" and cannot be fixed
    /// by the person hitting it.
    #[test]
    fn the_cpp_lift_is_acquirable_not_unwritten() {
        let e = edges()
            .into_iter()
            .find(|e| e.from == "cpp" && e.to == "tile")
            .expect("cpp -> tile edge");
        assert_eq!(e.need, Need::Toolchain("ascendc-to-rs"));
        assert_eq!(e.need.availability(), Availability::Acquirable);
        assert_eq!(e.fidelity, Fidelity::Synthesised);
    }

    /// A build that can see the lifter can take the route; one that cannot, cannot.
    /// Both directions, because a capability that is always true is not a capability.
    #[test]
    fn the_cpp_lift_needs_the_binary() {
        let with = Caps {
            toolchains: vec!["ascendc-to-rs"],
            ..Default::default()
        };
        let without = Caps::default();
        let n = Need::Toolchain("ascendc-to-rs");
        assert!(with.satisfies(&n));
        assert!(!without.satisfies(&n));
    }

    #[test]
    fn every_edge_names_forms_that_exist() {
        for e in edges() {
            assert!(
                forms::by_id(e.from).is_some(),
                "unknown from-form {}",
                e.from
            );
            assert!(forms::by_id(e.to).is_some(), "unknown to-form {}", e.to);
        }
    }

    #[test]
    fn lowering_from_tile_to_metal_is_two_hops() {
        let r = plan("tile", "msl", &[]).expect("route");
        assert_eq!(r.forms(), vec!["tile", "mlir", "msl"]);
        assert_eq!(r.hops[0].kind(), Kind::Lower);
    }

    #[test]
    fn a_lift_never_composes_into_a_lowering_route() {
        // pto -> tile exists as a lift; pto -> msl would need it. Refused by default.
        let e = plan("pto", "msl", &[]).unwrap_err();
        match e {
            RouteError::OnlyThroughLift { via, .. } => assert_eq!(via, vec!["tile"]),
            other => panic!("expected OnlyThroughLift, got {other:?}"),
        }
    }

    #[test]
    fn the_composed_route_runs_when_it_is_named() {
        let r = plan("pto", "msl", &["tile".to_string()]).expect("route");
        assert_eq!(r.forms(), vec!["pto", "tile", "mlir", "msl"]);
        // One synthesised hop makes the whole route synthesised.
        assert_eq!(r.fidelity(), Fidelity::Synthesised);
    }

    #[test]
    fn a_write_only_form_reports_the_missing_reader_not_a_flat_no_route() {
        // The distinction the user has to be able to act on: this is not "impossible",
        // it is "nobody has written the frontend". The refusal carries the route that
        // would exist if somebody did, so the answer names the work.
        let e = plan("msl", "cpp", &[]).unwrap_err();
        match e {
            RouteError::NeedsFrontend { from, to, would_be } => {
                assert_eq!(from, "msl");
                assert_eq!(to, "cpp");
                let r = would_be.expect("the hypothetical route");
                assert_eq!(r.forms(), vec!["msl", "mlir", "cpp"]);
                assert_eq!(r.fidelity(), Fidelity::Synthesised);
            }
            other => panic!("expected NeedsFrontend, got {other:?}"),
        }
    }

    #[test]
    fn same_form_optimization_on_an_unreadable_form_shows_the_round_trip() {
        // The search will not revisit a form, correctly — but same-form optimization IS
        // a round trip through the IR, so this route has to be constructed.
        let e = plan("msl", "msl", &[]).unwrap_err();
        match e {
            RouteError::NeedsFrontend { would_be, .. } => {
                let r = would_be.expect("the round trip");
                assert_eq!(r.forms(), vec!["msl", "mlir", "msl"]);
            }
            other => panic!("expected NeedsFrontend, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_reader_is_never_acquirable_but_a_missing_feature_is() {
        // This is the axis the exit codes are grouped on. A feature is one download
        // away; a frontend is somebody's unwritten work.
        assert_eq!(
            Need::Frontend("msl").availability(),
            Availability::Unwritten
        );
        assert_eq!(
            Need::Feature("ascend").availability(),
            Availability::Acquirable
        );
        assert_eq!(
            Need::Toolchain("llvm-20").availability(),
            Availability::Acquirable
        );
        assert_eq!(
            Need::Device("nvidia").availability(),
            Availability::Acquirable
        );
    }

    #[test]
    fn no_build_can_satisfy_a_frontend_need() {
        let mut caps = Caps::of_this_build();
        caps.features.push("emitters");
        caps.toolchains.push("rustc_codegen_tile");
        assert!(!caps.satisfies(&Need::Frontend("msl")));
    }

    #[test]
    fn same_form_optimization_exists_only_where_there_is_a_reader() {
        assert!(plan("mlir", "mlir", &[]).is_ok());
        assert!(plan("tile", "tile", &[]).is_ok());
        assert!(plan("msl", "msl", &[]).is_err());
        assert!(plan("gpu", "gpu", &[]).is_err());
    }

    #[test]
    fn rvv_is_emitted_from_mlir_directly_not_routed_through_linalg() {
        // The emitter wraps the linalg egress internally; it still consumes the tile
        // MLIR. A `linalg -> rvv` hop plans cleanly and then fails at emit time, because
        // linalg output carries no `hacc.entry` for the rvv emitter to find.
        let r = plan("mlir", "rvv", &[]).expect("route");
        assert_eq!(r.forms(), vec!["mlir", "rvv"]);
        assert!(
            !edges().iter().any(|e| e.from == "linalg" && e.to == "rvv"),
            "a linalg -> rvv hop is a route the emitter cannot take"
        );
    }

    #[test]
    fn refining_a_form_says_nothing_about_how_it_is_produced() {
        // `rvv refines linalg` is a SNIFFING fact: an rvv module carries linalg's marker.
        // It is not a routing fact, and reading it as one is what created the broken hop.
        assert_eq!(forms::by_id("rvv").unwrap().refines, Some("linalg"));
        let via_mlir = plan("mlir", "rvv", &[]).unwrap();
        assert_eq!(via_mlir.hops.len(), 1);
    }

    #[test]
    fn the_planner_terminates_on_every_pair() {
        // Property: no pair of forms sends the walk into a cycle.
        for a in forms::FORMS {
            for b in forms::FORMS {
                let _ = plan(a.id, b.id, &[]);
            }
        }
    }

    #[test]
    fn a_default_build_lacks_the_emitters_and_says_which_capability_is_missing() {
        let caps = Caps::default();
        let r = plan("mlir", "msl", &[]).expect("route");
        let missing = r.missing(&caps);
        assert_eq!(missing, vec![Need::Feature("emitters")]);
    }

    #[test]
    fn an_optimize_prologue_is_added_for_a_readable_input() {
        let plain = plan("mlir", "msl", &[]).unwrap();
        assert_eq!(plain.forms(), vec!["mlir", "msl"]);
        let opt = plan_with_opt("mlir", "msl", &[], 1).unwrap();
        assert_eq!(opt.forms(), vec!["mlir", "mlir", "msl"]);
        assert_eq!(opt.hops[0].kind(), Kind::Optimize);
        // The prologue is free and changes nothing about what may be trusted.
        assert_eq!(opt.cost(), plain.cost());
        assert_eq!(opt.fidelity(), plain.fidelity());
    }

    #[test]
    fn o0_gets_no_prologue_because_o0_is_verbatim() {
        let r = plan_with_opt("mlir", "msl", &[], 0).unwrap();
        assert_eq!(r.forms(), vec!["mlir", "msl"]);
    }

    #[test]
    fn an_unreadable_input_gets_no_prologue_even_at_o2() {
        // Nothing can optimize a form tile-rs cannot read, and pretending otherwise would
        // put a hop on the route that has no implementation behind it.
        let e = plan_with_opt("msl", "cpp", &[], 2);
        assert!(e.is_err(), "msl has no reader, so there is no route at all");
    }

    #[test]
    fn a_same_form_route_is_not_given_a_second_optimize_hop() {
        let r = plan_with_opt("mlir", "mlir", &[], 2).unwrap();
        assert_eq!(r.forms(), vec!["mlir", "mlir"], "one optimize hop, not two");
    }

    #[test]
    fn one_unvalidated_hop_downgrades_the_whole_route() {
        let r = plan("mlir", "rvv", &[]).expect("route");
        assert_eq!(r.fidelity(), Fidelity::Unvalidated);
    }

    #[test]
    fn list_forms_shows_that_most_forms_are_write_only() {
        let s = list_forms();
        assert!(s.contains("write-only"), "{s}");
        let readable = forms::FORMS.iter().filter(|f| f.readable).count();
        assert_eq!(readable, 3, "tile, mlir, linalg — and nothing else");
    }
}
