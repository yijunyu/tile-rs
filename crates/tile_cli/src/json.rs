//! Just enough JSON, by hand.
//!
//! The daemon speaks JSON-RPC, and a serialization crate would cost more than the
//! protocol does: `serde` + `serde_json` is a dozen crates against a default binary
//! budget of fifteen, for a message shape that fits on a page.
//!
//! Deliberately small and deliberately strict. It parses the subset MCP actually sends —
//! objects, arrays, strings, numbers, booleans, null — and refuses everything else rather
//! than guessing. A parser that guesses is worse than no parser when the thing on the
//! other end is an agent.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn s(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(k),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            Json::Num(n) => Some(*n as u8),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => write!(f, "null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
            Json::Str(s) => write!(f, "\"{}\"", escape(s)),
            Json::Arr(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Json::Obj(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "\"{}\":{v}", escape(k))?;
                }
                write!(f, "}}")
            }
        }
    }
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub fn parse(text: &str) -> Result<Json, String> {
    let b: Vec<char> = text.chars().collect();
    let mut i = 0;
    let v = value(&b, &mut i)?;
    skip_ws(&b, &mut i);
    if i != b.len() {
        return Err(format!("trailing input at char {i}"));
    }
    Ok(v)
}

fn skip_ws(b: &[char], i: &mut usize) {
    while *i < b.len() && b[*i].is_whitespace() {
        *i += 1;
    }
}

fn value(b: &[char], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    let Some(&c) = b.get(*i) else {
        return Err("unexpected end of input".into());
    };
    match c {
        '{' => object(b, i),
        '[' => array(b, i),
        '"' => Ok(Json::Str(string(b, i)?)),
        't' => lit(b, i, "true", Json::Bool(true)),
        'f' => lit(b, i, "false", Json::Bool(false)),
        'n' => lit(b, i, "null", Json::Null),
        c if c == '-' || c.is_ascii_digit() => number(b, i),
        c => Err(format!("unexpected character {c:?} at {i}")),
    }
}

fn lit(b: &[char], i: &mut usize, word: &str, v: Json) -> Result<Json, String> {
    for (n, wc) in word.chars().enumerate() {
        if b.get(*i + n) != Some(&wc) {
            return Err(format!("expected {word} at {i}"));
        }
    }
    *i += word.len();
    Ok(v)
}

fn number(b: &[char], i: &mut usize) -> Result<Json, String> {
    let start = *i;
    if b.get(*i) == Some(&'-') {
        *i += 1;
    }
    while b.get(*i).is_some_and(|c| {
        c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '+' || *c == '-'
    }) {
        *i += 1;
    }
    let s: String = b[start..*i].iter().collect();
    s.parse::<f64>().map(Json::Num).map_err(|e| e.to_string())
}

fn string(b: &[char], i: &mut usize) -> Result<String, String> {
    if b.get(*i) != Some(&'"') {
        return Err(format!("expected a string at {i}"));
    }
    *i += 1;
    let mut out = String::new();
    while let Some(&c) = b.get(*i) {
        *i += 1;
        match c {
            '"' => return Ok(out),
            '\\' => {
                let Some(&e) = b.get(*i) else {
                    return Err("unterminated escape".into());
                };
                *i += 1;
                match e {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hex: String = b
                            .get(*i..*i + 4)
                            .ok_or("short \\u escape")?
                            .iter()
                            .collect();
                        *i += 4;
                        let n = u32::from_str_radix(&hex, 16).map_err(|e| e.to_string())?;
                        out.push(char::from_u32(n).ok_or("bad code point")?);
                    }
                    e => return Err(format!("unknown escape \\{e}")),
                }
            }
            c => out.push(c),
        }
    }
    Err("unterminated string".into())
}

fn array(b: &[char], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut out = Vec::new();
    skip_ws(b, i);
    if b.get(*i) == Some(&']') {
        *i += 1;
        return Ok(Json::Arr(out));
    }
    loop {
        out.push(value(b, i)?);
        skip_ws(b, i);
        match b.get(*i) {
            Some(',') => *i += 1,
            Some(']') => {
                *i += 1;
                return Ok(Json::Arr(out));
            }
            _ => return Err(format!("expected , or ] at {i}")),
        }
    }
}

fn object(b: &[char], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut map = BTreeMap::new();
    skip_ws(b, i);
    if b.get(*i) == Some(&'}') {
        *i += 1;
        return Ok(Json::Obj(map));
    }
    loop {
        skip_ws(b, i);
        let k = string(b, i)?;
        skip_ws(b, i);
        if b.get(*i) != Some(&':') {
            return Err(format!("expected : at {i}"));
        }
        *i += 1;
        map.insert(k, value(b, i)?);
        skip_ws(b, i);
        match b.get(*i) {
            Some(',') => *i += 1,
            Some('}') => {
                *i += 1;
                return Ok(Json::Obj(map));
            }
            _ => return Err(format!("expected , or }} at {i}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_the_shapes_the_protocol_uses() {
        for src in [
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            r#"{"a":[1,2,3],"b":{"c":true,"d":null}}"#,
            r#"[]"#,
            r#"{}"#,
            r#""plain string""#,
        ] {
            let v = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
            let printed = v.to_string();
            let again = parse(&printed).unwrap();
            assert_eq!(v, again, "{src} did not survive a round trip as {printed}");
        }
    }

    #[test]
    fn strings_with_awkward_characters_survive() {
        // Kernel source goes through this: newlines, tabs, quotes and backslashes are
        // the normal case, not an edge case.
        let src = "kernel void f() {\n\t\"a\\b\"\n}";
        let printed = Json::s(src).to_string();
        assert_eq!(parse(&printed).unwrap().as_str().unwrap(), src);
    }

    #[test]
    fn a_control_character_is_escaped_not_emitted_raw() {
        let printed = Json::s("a\u{1}b").to_string();
        assert!(printed.contains("\\u0001"), "{printed}");
        assert_eq!(parse(&printed).unwrap().as_str().unwrap(), "a\u{1}b");
    }

    #[test]
    fn integers_do_not_come_back_with_a_decimal_point() {
        // An agent comparing `"id":1` against `"id":1.0` is a real interop failure.
        assert_eq!(Json::Num(1.0).to_string(), "1");
        assert_eq!(Json::Num(-42.0).to_string(), "-42");
    }

    #[test]
    fn object_keys_are_ordered_so_output_is_reproducible() {
        let a = Json::obj(vec![("b", Json::Num(1.0)), ("a", Json::Num(2.0))]);
        assert_eq!(a.to_string(), r#"{"a":2,"b":1}"#);
    }

    #[test]
    fn malformed_input_is_refused_rather_than_guessed_at() {
        for bad in [
            "{",
            "{\"a\":}",
            "[1,]",
            "tru",
            "\"unterminated",
            "{\"a\":1} trailing",
            "",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn nested_structures_parse_to_the_right_shape() {
        let v = parse(r#"{"params":{"name":"convert","arguments":{"to":"msl"}}}"#).unwrap();
        assert_eq!(
            v.get("params")
                .and_then(|p| p.get("arguments"))
                .and_then(|a| a.get("to"))
                .and_then(|t| t.as_str()),
            Some("msl")
        );
    }
}
