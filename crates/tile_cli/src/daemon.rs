//! Daemon mode: the same capabilities, over MCP, for agents.
//!
//! Requirement (7). The CLI and the daemon are two renderings of one core, so an agent
//! can do exactly what a person can do **and no more** — that is a property of the
//! layering, not a promise anyone has to keep by hand: every tool below is a thin wrapper
//! over the same function `src/bin/tile.rs` calls.
//!
//! ## Three rules that only matter in daemon mode
//!
//! * **stdout belongs to the protocol.** Every diagnostic goes to stderr. A well-meaning
//!   log line on stdout corrupts the framing, and the failure looks like a parser bug on
//!   the other end.
//! * **No acquisition on an agent's behalf.** An MCP client asking a question must not be
//!   able to start a multi-gigabyte download. `install` exists as a tool, so the client
//!   can ask *explicitly*; nothing else acquires anything.
//! * **stdio only.** A TCP listener serving license-gated data with "binds localhost" as
//!   its whole security story is not a security story. `--listen` needs a token, and
//!   until there is one it is refused.
//!
//! No async runtime: the protocol is a request/response loop over stdin, and a blocking
//! read is exactly the right shape for it. That keeps the core synchronous, which was the
//! point of D4.

use crate::json::Json;
use crate::{emit, forms, optimize, platform, profile, provision, routes, simenv};
use std::io::{BufRead, Write};

pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// The tools this server advertises. One row per capability, and each one names the
/// CLI flag it mirrors so the two cannot drift apart unnoticed.
pub fn tool_list() -> Json {
    let tool = |name: &str, desc: &str, props: Vec<(&str, &str, &str)>, required: Vec<&str>| {
        let mut schema_props = Vec::new();
        for (n, ty, d) in props {
            schema_props.push((
                n,
                Json::obj(vec![("type", Json::s(ty)), ("description", Json::s(d))]),
            ));
        }
        Json::obj(vec![
            ("name", Json::s(name)),
            ("description", Json::s(desc)),
            (
                "inputSchema",
                Json::obj(vec![
                    ("type", Json::s("object")),
                    ("properties", Json::obj(schema_props)),
                    (
                        "required",
                        Json::Arr(required.into_iter().map(Json::s).collect()),
                    ),
                ]),
            ),
        ])
    };

    Json::Arr(vec![
        tool(
            "doctor",
            "Report the platform, the accelerators present, their SDKs, and the default \
             output form. Mirrors `tile doctor`.",
            vec![],
            vec![],
        ),
        tool(
            "list_forms",
            "The reader/writer matrix: every kernel form, whether tile-rs can read it, \
             write it, optimize it in place, and lift from it. Mirrors `tile --list-forms`.",
            vec![],
            vec![],
        ),
        tool(
            "identify",
            "Decide what kind of kernel some source is, and say how it was decided. \
             Mirrors the form resolution behind every command.",
            vec![
                ("source", "string", "The kernel source text."),
                ("filename", "string", "Its name, for the extension hint."),
                ("from", "string", "Force a form instead of sniffing."),
            ],
            vec!["source", "filename"],
        ),
        tool(
            "profile",
            "What a kernel is made of: dtypes and promotion, extents, the tile plan, the \
             RAW/WAR/WAW hazard edges and required barriers, and the target's resource \
             bounds -- or the refusal, when the architecture has never been measured. \
             Mirrors `tile <kernel> -i`.",
            vec![
                ("source", "string", "The kernel source text."),
                ("filename", "string", "Its name, for the extension hint."),
                (
                    "target",
                    "string",
                    "Accelerator family to check bounds against.",
                ),
            ],
            vec!["source", "filename"],
        ),
        tool(
            "routes",
            "Every route between two forms, with its cost, its fidelity class and the \
             capabilities it needs. Mirrors `tile --route`.",
            vec![
                ("from", "string", "The input form id."),
                ("to", "string", "The output form id."),
                ("level", "number", "Optimization level, 0 to 4."),
            ],
            vec!["from", "to"],
        ),
        tool(
            "convert",
            "Lower, lift or optimize a kernel and return the result. Never writes a file \
             and never acquires a toolchain. Mirrors `tile <kernel> -t <form> -o -`.",
            vec![
                ("source", "string", "The kernel source text."),
                ("filename", "string", "Its name, for the extension hint."),
                ("to", "string", "The output form id."),
                ("level", "number", "Optimization level, 0 to 4. Default 2."),
            ],
            vec!["source", "filename", "to"],
        ),
        tool(
            "install",
            "Provision a toolchain, or list what this platform can be given. This is the \
             ONLY tool that acquires anything, and it must be asked for explicitly.",
            vec![("tool", "string", "The tool id. Omit to list.")],
            vec![],
        ),
    ])
}

fn text_result(s: String, is_error: bool) -> Json {
    Json::obj(vec![
        (
            "content",
            Json::Arr(vec![Json::obj(vec![
                ("type", Json::s("text")),
                ("text", Json::s(s)),
            ])]),
        ),
        ("isError", Json::Bool(is_error)),
    ])
}

fn arg<'a>(args: &'a Json, k: &str) -> Option<&'a str> {
    args.get(k).and_then(|v| v.as_str())
}

/// Run one tool call. Pure with respect to the filesystem except for `install`.
pub fn call_tool(name: &str, args: &Json, env: &simenv::Env) -> Json {
    match name {
        "doctor" => {
            let p = platform::detect_in(env);
            let form = crate::default_form_for(p.primary_family());
            text_result(
                format!("{p}default output form: {} ({})\n", form.id, form.display),
                false,
            )
        }
        "list_forms" => text_result(routes::list_forms(), false),
        "identify" => {
            let (Some(src), Some(name)) = (arg(args, "source"), arg(args, "filename")) else {
                return text_result("identify needs `source` and `filename`".into(), true);
            };
            match forms::resolve_input(name, src, arg(args, "from")) {
                Ok(r) => text_result(
                    format!(
                        "form: {} ({})\nlevel: {}\nreadable: {}\nfidelity: {}\n",
                        r.form.id, r.how, r.form.level, r.form.readable, r.form.fidelity
                    ),
                    false,
                ),
                Err(e) => text_result(e.to_string(), true),
            }
        }
        "profile" => {
            let (Some(src), Some(name)) = (arg(args, "source"), arg(args, "filename")) else {
                return text_result("profile needs `source` and `filename`".into(), true);
            };
            let form = match forms::resolve_input(name, src, arg(args, "from")) {
                Ok(r) => r,
                Err(e) => return text_result(e.to_string(), true),
            };
            let family = arg(args, "target").unwrap_or("none");
            let p = profile::profile(name, src, form.form, form.how, family, 8192);
            text_result(p.render(), false)
        }
        "routes" => {
            let (Some(from), Some(to)) = (arg(args, "from"), arg(args, "to")) else {
                return text_result("routes needs `from` and `to`".into(), true);
            };
            let level = args.get("level").and_then(|v| v.as_u8()).unwrap_or(2);
            let all = routes::routes(from, to, &[]);
            if all.is_empty() {
                return match routes::plan(from, to, &[]) {
                    Ok(_) => text_result("no routes".into(), true),
                    Err(e) => text_result(format!("{e:?}"), true),
                };
            }
            let caps = routes::Caps::from_env(env);
            let mut out = String::new();
            for r in all {
                let r = routes::with_opt_prologue(r, level);
                let missing = r.missing(&caps);
                out.push_str(&format!(
                    "{}  cost {}  {}\n",
                    r.describe(),
                    r.cost(),
                    if missing.is_empty() {
                        "available".to_string()
                    } else {
                        format!(
                            "needs {}",
                            missing
                                .iter()
                                .map(|n| n.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                ));
            }
            text_result(out, false)
        }
        "convert" => {
            let (Some(src), Some(name), Some(to)) =
                (arg(args, "source"), arg(args, "filename"), arg(args, "to"))
            else {
                return text_result("convert needs `source`, `filename` and `to`".into(), true);
            };
            let level = args.get("level").and_then(|v| v.as_u8()).unwrap_or(2);
            let resolved = match forms::resolve_input(name, src, arg(args, "from")) {
                Ok(r) => r,
                Err(e) => return text_result(e.to_string(), true),
            };
            let Some(out_form) = forms::by_id(to) else {
                return text_result(format!("no such form \"{to}\""), true);
            };
            let route = match routes::plan_with_opt(resolved.form.id, out_form.id, &[], level) {
                Ok(r) => r,
                Err(e) => return text_result(describe_route_error(&e), true),
            };
            // An agent must not be able to start a download by asking a question.
            let caps = routes::Caps::from_env(env);
            let missing = route.missing(&caps);
            if !missing.is_empty() {
                return text_result(
                    format!(
                        "the route {} needs {}, which this server does not have. \
                         Nothing is acquired on your behalf -- call the `install` tool \
                         explicitly if that is what you want.",
                        route.describe(),
                        missing
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    true,
                );
            }
            let mut current = src.to_string();
            for hop in &route.hops {
                if hop.from == hop.to {
                    current = optimize::optimize(&current, level).0;
                    continue;
                }
                let to_form = forms::by_id(hop.to).expect("edge target");
                match emit::emit(to_form, &current) {
                    Ok(s) => current = s,
                    Err(e) => return text_result(format!("{} -> {}: {e}", hop.from, hop.to), true),
                }
            }
            text_result(current, false)
        }
        "install" => {
            let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
            match arg(args, "tool") {
                None => {
                    let mut out = format!("tools for {os}/{arch}:\n");
                    for (t, done) in provision::listing(os, arch) {
                        out.push_str(&format!(
                            "  {:<20} {:<8} {}\n",
                            t.id,
                            t.version,
                            if done {
                                "installed".into()
                            } else if let Some(b) = &t.barrier {
                                format!("needs a person ({b})")
                            } else if !t.pinned() {
                                "not pinned yet".to_string()
                            } else {
                                "available".to_string()
                            }
                        ));
                    }
                    text_result(out, false)
                }
                Some(id) => {
                    let mut said = String::new();
                    match provision::ensure(id, os, arch, provision::Policy::OnDemand, &mut |m| {
                        said.push_str(m);
                        said.push('\n');
                    }) {
                        Ok(o) => text_result(format!("{said}{o:?}"), false),
                        Err(e) => text_result(format!("{said}{e}"), true),
                    }
                }
            }
        }
        other => text_result(format!("no such tool \"{other}\""), true),
    }
}

fn describe_route_error(e: &routes::RouteError) -> String {
    match e {
        routes::RouteError::NeedsFrontend { from, to, would_be } => format!(
            "cannot go from \"{from}\" to \"{to}\" yet: tile-rs can emit \"{from}\" but \
             cannot read it -- no {from} frontend has been written. That is missing work, \
             not a missing possibility.{}",
            would_be
                .as_ref()
                .map(|r| format!(" With one, the route would be: {}", r.describe()))
                .unwrap_or_default()
        ),
        routes::RouteError::ViaNotOnAnyRoute {
            from,
            to,
            via,
            available,
        } => format!(
            "--via {} names a form no route from \"{from}\" to \"{to}\" visits. \
             Available: {}",
            via.join(","),
            available.join(" | ")
        ),
        routes::RouteError::OnlyThroughLift { from, to, via } => format!(
            "no direct route from \"{from}\" to \"{to}\". One exists through a lift ({}), \
             but a lift never composes automatically -- ask for it by name.",
            via.join(", ")
        ),
        routes::RouteError::NoRoute { from, to, .. } => {
            format!("no route from \"{from}\" to \"{to}\" has been built yet")
        }
    }
}

/// Handle one JSON-RPC request. `None` means the message was a notification.
pub fn handle(req: &Json, env: &simenv::Env) -> Option<Json> {
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = req.get("id").cloned();
    let params = req.get("params").cloned().unwrap_or(Json::Null);

    // A notification has no id and gets no reply. Replying to one is a protocol error
    // that the client sees as an unsolicited message.
    let respond = |result: Json| -> Option<Json> {
        let id = id.clone()?;
        Some(Json::obj(vec![
            ("jsonrpc", Json::s("2.0")),
            ("id", id),
            ("result", result),
        ]))
    };

    match method {
        "initialize" => respond(Json::obj(vec![
            ("protocolVersion", Json::s(PROTOCOL_VERSION)),
            (
                "capabilities",
                Json::obj(vec![("tools", Json::obj(vec![]))]),
            ),
            (
                "serverInfo",
                Json::obj(vec![
                    ("name", Json::s("tile")),
                    ("version", Json::s(crate::VERSION)),
                ]),
            ),
        ])),
        "tools/list" => respond(Json::obj(vec![("tools", tool_list())])),
        "tools/call" => {
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Json::Null);
            respond(call_tool(name, &args, env))
        }
        "ping" => respond(Json::obj(vec![])),
        _ if id.is_none() => None,
        _ => {
            let id = id?;
            Some(Json::obj(vec![
                ("jsonrpc", Json::s("2.0")),
                ("id", id),
                (
                    "error",
                    Json::obj(vec![
                        ("code", Json::Num(-32601.0)),
                        ("message", Json::s(format!("no such method \"{method}\""))),
                    ]),
                ),
            ]))
        }
    }
}

/// Serve MCP over stdio until the input closes.
pub fn serve_stdio(env: &simenv::Env, verbose: u8) -> i32 {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    // Everything diagnostic goes to stderr. stdout is the protocol's, exclusively.
    if verbose > 0 {
        eprintln!("tile: MCP over stdio, protocol {PROTOCOL_VERSION}");
    }
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match crate::json::parse(&line) {
            Ok(req) => handle(&req, env),
            Err(e) => Some(Json::obj(vec![
                ("jsonrpc", Json::s("2.0")),
                ("id", Json::Null),
                (
                    "error",
                    Json::obj(vec![
                        ("code", Json::Num(-32700.0)),
                        ("message", Json::s(format!("parse error: {e}"))),
                    ]),
                ),
            ])),
        };
        if let Some(r) = reply {
            if writeln!(stdout, "{r}").is_err() || stdout.flush().is_err() {
                break;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> simenv::Env {
        simenv::Env::detect()
    }

    fn call(name: &str, args: Vec<(&str, Json)>) -> (String, bool) {
        let r = call_tool(name, &Json::obj(args), &env());
        let text = r
            .get("content")
            .and_then(|c| match c {
                Json::Arr(a) => a.first().cloned(),
                _ => None,
            })
            .and_then(|c| c.get("text").and_then(|t| t.as_str()).map(str::to_string))
            .unwrap_or_default();
        (
            text,
            r.get("isError").and_then(|e| e.as_bool()).unwrap_or(false),
        )
    }

    const MLIR: &str = include_str!("../testdata/forms/softmax.mlir");

    #[test]
    fn initialize_answers_with_the_protocol_version_and_the_server_name() {
        let req = crate::json::parse(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#).unwrap();
        let r = handle(&req, &env()).expect("a reply");
        assert_eq!(
            r.get("result")
                .and_then(|x| x.get("protocolVersion"))
                .and_then(|v| v.as_str()),
            Some(PROTOCOL_VERSION)
        );
        assert_eq!(
            r.get("result")
                .and_then(|x| x.get("serverInfo"))
                .and_then(|s| s.get("name"))
                .and_then(|n| n.as_str()),
            Some("tile")
        );
    }

    #[test]
    fn a_notification_gets_no_reply() {
        // Replying to a notification is a protocol error the client sees as an
        // unsolicited message.
        let req = crate::json::parse(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .unwrap();
        assert!(handle(&req, &env()).is_none());
    }

    #[test]
    fn an_unknown_method_with_an_id_gets_an_error_not_silence() {
        let req = crate::json::parse(r#"{"jsonrpc":"2.0","id":9,"method":"nope"}"#).unwrap();
        let r = handle(&req, &env()).expect("a reply");
        assert!(r.get("error").is_some(), "{r}");
    }

    #[test]
    fn every_advertised_tool_has_a_schema_and_answers_when_called() {
        let Json::Arr(tools) = tool_list() else {
            panic!("tools/list is not an array")
        };
        assert!(!tools.is_empty());
        for t in &tools {
            let name = t.get("name").and_then(|n| n.as_str()).expect("a name");
            assert!(t.get("description").is_some(), "{name} has no description");
            let schema = t.get("inputSchema").expect("a schema");
            assert_eq!(schema.get("type").and_then(|x| x.as_str()), Some("object"));
            // Called with nothing, a tool must answer -- with an error if it needs
            // arguments, never a panic.
            let _ = call_tool(name, &Json::Null, &env());
        }
    }

    #[cfg(feature = "emitters")]
    #[test]
    fn convert_over_mcp_matches_what_the_library_produces() {
        // The parity that makes "an agent can do what a person can do" structural rather
        // than a promise: same core function, same bytes.
        let (text, err) = call(
            "convert",
            vec![
                ("source", Json::s(MLIR)),
                ("filename", Json::s("softmax.mlir")),
                ("to", Json::s("msl")),
                ("level", Json::Num(0.0)),
            ],
        );
        assert!(!err, "{text}");
        let direct = emit::emit(forms::by_id("msl").unwrap(), MLIR).unwrap();
        assert_eq!(text, direct);
    }

    #[test]
    fn convert_never_acquires_anything_on_an_agents_behalf() {
        // An MCP client asking a question must not be able to start a download.
        let (text, err) = call(
            "convert",
            vec![
                ("source", Json::s(MLIR)),
                ("filename", Json::s("softmax.mlir")),
                ("to", Json::s("cpp")),
            ],
        );
        assert!(err, "{text}");
        assert!(
            text.contains("Nothing is acquired on your behalf"),
            "{text}"
        );
        assert!(text.contains("install"), "{text}");
    }

    #[test]
    fn a_refusal_keeps_its_tense_over_the_protocol_too() {
        let (text, err) = call(
            "convert",
            vec![
                ("source", Json::s("kernel void f() {}")),
                ("filename", Json::s("k.metal")),
                ("to", Json::s("msl")),
            ],
        );
        assert!(err, "{text}");
        assert!(
            text.contains("missing work, not a missing possibility"),
            "{text}"
        );
    }

    #[test]
    fn profile_over_mcp_refuses_an_unmeasured_architecture() {
        let (text, err) = call(
            "profile",
            vec![
                ("source", Json::s(MLIR)),
                ("filename", Json::s("softmax.mlir")),
                ("target", Json::s("ascend-950")),
            ],
        );
        assert!(!err, "{text}");
        assert!(text.contains("UNMEASURED"), "{text}");
    }

    #[test]
    fn identify_reports_how_it_decided() {
        let (text, err) = call(
            "identify",
            vec![
                ("source", Json::s("@nki.jit\ndef k(): pass\n")),
                ("filename", Json::s("k.py")),
            ],
        );
        assert!(!err, "{text}");
        assert!(text.contains("form: nki"), "{text}");
        assert!(text.contains("sniffed"), "{text}");
    }

    #[test]
    fn a_missing_argument_is_an_error_result_not_a_panic() {
        let (text, err) = call("convert", vec![("source", Json::s("x"))]);
        assert!(err);
        assert!(text.contains("needs"), "{text}");
    }
}
