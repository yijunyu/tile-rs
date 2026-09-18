//! `-u`: the same core, rendered for eyes.
//!
//! ## Why a server rather than a window
//!
//! The design says `-u` serves a bundle on loopback and opens a browser, and `--ui native`
//! opens an eframe window. The server half is the part that has to exist either way, and
//! it is the part that can be **tested**: a headless CI box can fetch a page and assert
//! what is on it; it cannot look at a window.
//!
//! It is also the half with no build dependency. An egui-wasm bundle needs a wasm
//! toolchain — a *build* prerequisite the tool would have to provision, and M4's rule is
//! that nothing is acquired that cannot be verified. So the bundle slots in later as a
//! static asset behind this same server, and until it does the page is rendered here.
//!
//! ## Headless is the normal case
//!
//! This is a compiler tool. It runs over ssh, in CI, and on machines with no display far
//! more often than not, so "no windowing subsystem" degrades to **printing the URL**, not
//! to an error. Exit 7 is reserved for a subsystem that was asked for by name and could
//! not start.
//!
//! The listener binds loopback only. A tool that will one day show license-gated data
//! does not get to bind `0.0.0.0` because it was convenient.

use crate::{forms, platform, profile, routes, simenv};
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};

/// Everything the page shows, gathered once from the core.
pub struct View {
    pub title: String,
    pub profile: Option<String>,
    pub platform: String,
    pub forms: String,
    pub routes: String,
}

impl View {
    pub fn build(env: &simenv::Env, input: Option<(&str, &str)>, family: &str) -> View {
        let plat = platform::detect_in(env);
        let (title, prof, route_text) = match input {
            Some((name, text)) => {
                let r = forms::resolve_input(name, text, None);
                match r {
                    Ok(r) => {
                        let p = profile::profile(name, text, r.form, r.how, family, 8192);
                        let out = crate::default_form_for(family);
                        let routes = routes::routes(r.form.id, out.id, &[])
                            .into_iter()
                            .map(|x| format!("{}  cost {}", x.describe(), x.cost()))
                            .collect::<Vec<_>>()
                            .join("\n");
                        (name.to_string(), Some(p.render()), routes)
                    }
                    Err(e) => (name.to_string(), Some(e.to_string()), String::new()),
                }
            }
            None => ("tile".to_string(), None, String::new()),
        };
        View {
            title,
            profile: prof,
            platform: plat.to_string(),
            forms: routes::list_forms(),
            routes: route_text,
        }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The page. Hand-written, self-contained, no network fetches — a viewer with no
/// connectivity still sees everything, which a CDN-linked page would not manage.
pub fn page(v: &View) -> String {
    let section = |title: &str, body: &str| {
        if body.trim().is_empty() {
            String::new()
        } else {
            format!("<h2>{}</h2><pre>{}</pre>", esc(title), esc(body))
        }
    };
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <title>tile — {}</title><style>\
         :root{{color-scheme:light dark}}\
         body{{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:2rem;\
         max-width:70rem;line-height:1.5}}\
         h1{{font-size:1.4rem;margin:0 0 .25rem}}\
         h2{{font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;opacity:.6;\
         margin:2rem 0 .5rem}}\
         pre{{white-space:pre-wrap;margin:0;padding:1rem;border:1px solid;\
         border-color:color-mix(in srgb, currentColor 20%, transparent);border-radius:4px}}\
         .sub{{opacity:.6;margin:0 0 1rem}}\
         table{{border-collapse:collapse;font-size:.85rem;width:100%}}\
         th,td{{text-align:left;padding:.3rem .6rem;border-bottom:1px solid;\
         border-color:color-mix(in srgb, currentColor 12%, transparent)}}\
         th{{font-weight:600;opacity:.6;font-size:.7rem;text-transform:uppercase;\
         letter-spacing:.08em}}\
         input{{font:inherit;padding:.25rem .5rem;border-radius:4px;border:1px solid;\
         border-color:color-mix(in srgb, currentColor 30%, transparent);\
         background:transparent;color:inherit}}\
         input:disabled{{opacity:.5}}\
         </style></head><body>\
         <h1>{}</h1><p class=\"sub\">tile {}</p>{}{}{}{}</body></html>",
        esc(&v.title),
        esc(&v.title),
        crate::VERSION,
        section("this machine", &v.platform),
        v.profile
            .as_deref()
            .map(|p| section("profile", p))
            .unwrap_or_default(),
        section("routes", &v.routes),
        forms_table(&v.forms),
    )
}

/// The form matrix as a real table, plus the box that filters it.
///
/// A `<pre>` cannot be filtered row by row, and filtering is the one thing the wasm
/// bundle buys over the server-rendered page. The box ships DISABLED and the loader
/// enables it: if the module never arrives, the reader sees a complete table and no
/// control that does nothing.
fn forms_table(text: &str) -> String {
    let mut rows = String::new();
    let mut n = 0usize;
    for line in text.lines() {
        // Skip the header, the rule, and the prose footer: the table is the rows that
        // start with a form id, which are the ones indented by nothing and column-shaped.
        let cells: Vec<&str> = line.split_whitespace().collect();
        // A row is a line whose first cell IS a form id. Counting columns instead let the
        // prose footer through -- every one of its lines has eight or more words -- and
        // produced eight extra "forms" named `grammar,` and `readable.`.
        if cells.len() < 8 || forms::by_id(cells[0]).is_none_or(|f| f.id != cells[0]) {
            continue;
        }
        rows.push_str("<tr>");
        for c in &cells[..7] {
            rows.push_str(&format!("<td>{}</td>", esc(c)));
        }
        rows.push_str(&format!("<td>{}</td>", esc(&cells[7..].join(" "))));
        rows.push_str("</tr>");
        n += 1;
    }
    // The prose footer, kept. It is the part that says WHY 17 rows read "no" -- that a
    // source form owes no reader -- and dropping it while keeping the table would leave
    // the page stating the shape without the reason, which is how the fan-out gets read
    // as a shortfall.
    let mut note = String::new();
    for line in text.lines() {
        let first = line.split_whitespace().next().unwrap_or("");
        if first.is_empty()
            || forms::by_id(first).is_some_and(|f| f.id == first)
            || first == "form"
            || first.starts_with('-')
        {
            continue;
        }
        note.push_str(&esc(line));
        note.push(' ');
    }

    format!(
        "<h2>forms</h2>\
         <p class=\"sub\"><input id=\"filter\" disabled \
         placeholder=\"filter (loading wasm…)\" size=\"32\"> \
         <span id=\"filter-count\">{n} of {n}</span></p>\
         <table id=\"forms\"><thead><tr>\
         <th>form</th><th>role</th><th>lvl</th><th>ext</th><th>read</th>\
         <th>write</th><th>optimize</th><th>fidelity</th></tr></thead>\
         <tbody>{rows}</tbody></table>\
         <p class=\"sub\">{note}</p>\
         <script src=\"/loader.js\"></script>"
    )
}

/// Bind an ephemeral loopback port, or the one asked for.
pub fn bind(port: u16) -> std::io::Result<TcpListener> {
    // Loopback, always. Not a default that a flag can widen.
    TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
}

/// The wasm bundle, and the glue that loads it.
///
/// Checked in as built artifacts and included here, exactly like the emitters: they come
/// from `crates/tile_ui_wasm`, which is a separate workspace with a wasm target, and
/// building it at `tile` build time would make a wasm toolchain a prerequisite for
/// `tile k.mlir -o k.metal`. `the_checked_in_wasm_bundle_is_a_real_module` is the drift
/// alarm, the same role the golden files play for the emitters.
pub const WASM_BUNDLE: &[u8] = include_bytes!("../assets/ui/tile_ui.wasm");
pub const WASM_LOADER: &str = include_str!("../assets/ui/loader.js");

/// Serve `body` to each request until `limit` requests have been answered.
///
/// `limit` exists for the tests: a server that only ever runs forever cannot be asserted
/// against without a thread and a kill.
pub fn serve(listener: &TcpListener, body: &str, limit: Option<usize>) -> std::io::Result<usize> {
    let mut served = 0;
    for stream in listener.incoming() {
        let mut stream = stream?;
        respond(&mut stream, body)?;
        served += 1;
        if limit.is_some_and(|l| served >= l) {
            break;
        }
    }
    Ok(served)
}

/// What a request path should be answered with.
///
/// The server used to ignore the path entirely and hand the same HTML to every request,
/// which is fine for one page and useless the moment the page needs an asset: the browser
/// would have fetched `/tile_ui.wasm` and been given HTML with a 200, and
/// `instantiateStreaming` fails on that with a message about a magic number.
pub enum Asset {
    Page,
    Wasm,
    Loader,
}

pub fn route(path: &str) -> Asset {
    match path {
        "/tile_ui.wasm" => Asset::Wasm,
        "/loader.js" => Asset::Loader,
        _ => Asset::Page,
    }
}

fn respond(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    // Read the request line so the client is not left writing into a closed socket.
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    let _ = reader.read_line(&mut line);
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    while let Ok(n) = reader.read_line(&mut line) {
        if n <= 2 {
            break;
        }
    }
    match route(&path) {
        Asset::Wasm => {
            // The MIME type is load-bearing: instantiateStreaming refuses anything that
            // is not application/wasm, and the failure names a magic number rather than a
            // content type, which sends people looking in the wrong place.
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/wasm\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                WASM_BUNDLE.len()
            )?;
            std::io::Write::write_all(stream, WASM_BUNDLE)?;
        }
        Asset::Loader => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/javascript; charset=utf-8\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{WASM_LOADER}",
                WASM_LOADER.len()
            )?;
        }
        Asset::Page => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )?;
        }
    }
    stream.flush()
}

/// Is there anything that could show a window?
///
/// Deliberately conservative: on Linux a display is `DISPLAY` or `WAYLAND_DISPLAY`;
/// elsewhere assume there is one, because being wrong means printing a URL the user did
/// not need rather than failing on a machine that would have worked.
pub fn windowing_available() -> bool {
    if cfg!(target_os = "linux") {
        std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()
    } else {
        true
    }
}

/// Ask the desktop to open a URL. Failure is not an error: the URL was printed.
pub fn open_browser(url: &str) -> bool {
    let (prog, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        ("open", &[])
    } else if cfg!(target_os = "windows") {
        ("cmd", &["/C", "start"])
    } else {
        ("xdg-open", &[])
    };
    std::process::Command::new(prog)
        .args(args)
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn a_view() -> View {
        View {
            title: "softmax.mlir".into(),
            profile: Some("  form: mlir\n  hazards: 6 edges".into()),
            platform: "platform: aarch64-apple-darwin".into(),
            forms: "msl  1  .metal  no  yes".into(),
            routes: "mlir -> msl  [exact]  cost 1".into(),
        }
    }

    #[test]
    fn the_page_carries_what_the_core_reported() {
        let p = page(&a_view());
        for want in [
            "softmax.mlir",
            "hazards: 6 edges",
            "aarch64-apple-darwin",
            "mlir -&gt; msl",
        ] {
            assert!(p.contains(want), "the page omits {want:?}");
        }
    }

    #[test]
    fn the_page_needs_nothing_from_the_network() {
        // A viewer with no connectivity sees everything. A CDN-linked page would not.
        let p = page(&a_view());
        assert!(!p.contains("http://"), "an external reference: {p}");
        assert!(!p.contains("https://"), "an external reference");
        // A SAME-ORIGIN script is not a network dependency: /loader.js is served by this
        // same process on loopback. What the scenario is about is a CDN, and the two
        // asserts above are what actually say that. Checked explicitly so nobody
        // "restores" the old blanket ban and silently drops the wasm bundle.
        assert!(!p.contains("src=\"http"), "an off-machine script");
        assert!(
            p.matches("<script").count() <= 1,
            "more scripts than the one loader: {p}"
        );
    }

    #[test]
    fn kernel_source_cannot_inject_markup_into_the_page() {
        let mut v = a_view();
        v.profile = Some("<script>alert(1)</script>".into());
        let p = page(&v);
        assert!(
            !p.contains("<script>alert"),
            "unescaped markup reached the page"
        );
        assert!(p.contains("&lt;script&gt;"), "it should be shown as text");
    }

    #[test]
    fn an_empty_section_is_omitted_rather_than_shown_blank() {
        let mut v = a_view();
        v.routes = String::new();
        assert!(!page(&v).contains("ROUTES"));
    }

    #[test]
    fn the_listener_binds_loopback_and_nothing_else() {
        // Not a default a flag can widen: a tool that will one day show license-gated
        // data does not get to bind 0.0.0.0 because it was convenient.
        let l = bind(0).expect("bind");
        assert!(l.local_addr().unwrap().ip().is_loopback());
    }

    #[test]
    fn a_request_gets_the_page_back() {
        let l = bind(0).expect("bind");
        let addr = l.local_addr().unwrap();
        let body = page(&a_view());
        let expect = body.clone();
        let t = std::thread::spawn(move || serve(&l, &body, Some(1)));

        let mut s = std::net::TcpStream::connect(addr).expect("connect");
        write!(s, "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut got = String::new();
        s.read_to_string(&mut got).unwrap();

        assert_eq!(t.join().unwrap().unwrap(), 1);
        assert!(got.starts_with("HTTP/1.1 200 OK"), "{got}");
        assert!(got.contains("Content-Length:"), "a client needs the length");
        assert!(got.ends_with(&expect), "the body did not arrive intact");
    }

    #[test]
    fn a_view_of_a_kernel_carries_its_profile_and_its_routes() {
        let src = include_str!("../testdata/forms/softmax.mlir");
        let v = View::build(
            &simenv::Env::detect(),
            Some(("softmax.mlir", src)),
            "apple-gpu",
        );
        let p = v.profile.expect("a profile");
        assert!(p.contains("form: mlir"), "{p}");
        assert!(p.contains("hazards:"), "{p}");
        assert!(v.routes.contains("msl"), "{}", v.routes);
    }

    #[test]
    fn the_checked_in_wasm_bundle_is_a_real_module() {
        // The bundle is a BUILT ARTIFACT checked in, exactly like the emitter goldens,
        // because building it here would make a wasm toolchain a prerequisite for
        // `tile k.mlir -o k.metal`. Nothing else would notice it going stale or empty,
        // so this is its drift alarm: the magic number, the version, and the four
        // entry points the loader actually calls.
        assert!(
            WASM_BUNDLE.len() > 1000,
            "bundle is {} bytes",
            WASM_BUNDLE.len()
        );
        assert_eq!(&WASM_BUNDLE[..4], b"\0asm", "not a wasm module");
        assert_eq!(&WASM_BUNDLE[4..8], &[1, 0, 0, 0], "not wasm version 1");
        // The export names are plain bytes in the export section; finding them is enough
        // to catch a rename, which is the failure that would leave a silent dead box.
        for name in ["alloc", "dealloc", "filter", "result_len", "memory"] {
            assert!(
                WASM_BUNDLE
                    .windows(name.len())
                    .any(|w| w == name.as_bytes()),
                "the bundle exports no {name}"
            );
        }
        // And the loader must call exactly those.
        for name in ["alloc", "dealloc", "filter", "result_len", "memory"] {
            assert!(WASM_LOADER.contains(name), "the loader never calls {name}");
        }
    }

    #[test]
    fn the_assets_are_served_from_their_own_paths() {
        // The server used to ignore the path and hand HTML to every request, so a browser
        // fetching /tile_ui.wasm got a page with a 200 and instantiateStreaming failed
        // with a message about a magic number -- which sends you looking at the module.
        assert!(matches!(route("/tile_ui.wasm"), Asset::Wasm));
        assert!(matches!(route("/loader.js"), Asset::Loader));
        assert!(matches!(route("/"), Asset::Page));
        assert!(matches!(route("/anything-else"), Asset::Page));
    }

    #[test]
    fn the_form_table_has_a_row_per_form_and_a_disabled_box() {
        let html = forms_table(&routes::list_forms());
        // One row per form in the table, so the filter indices line up with what the
        // wasm is told about.
        let rows = html.matches("<tr>").count() - 1; // minus the header row
        assert_eq!(rows, forms::FORMS.len(), "{html}");
        // Disabled until the module arrives: a control that does nothing is worse than
        // no control, and the page has to stand up without wasm at all.
        assert!(html.contains("id=\"filter\" disabled"), "{html}");
        assert!(html.contains("/loader.js"), "{html}");
    }

    #[test]
    fn a_view_with_no_kernel_still_shows_the_machine_and_the_forms() {
        let v = View::build(&simenv::Env::detect(), None, "none");
        assert!(v.profile.is_none());
        assert!(v.platform.contains("platform:"));
        assert!(v.forms.contains("write-only"));
    }
}
