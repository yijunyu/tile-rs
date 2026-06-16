//! Shared MLIR text-parsing helpers consumed by the `mlir_to_cpp` (AscendC
//! C++) and `mlir_to_msl` (Apple Metal Shading Language) emitters, as well as
//! any future MLIR back end that needs to walk the same `@__tile_*`
//! intrinsic shape produced by `rustc_codegen_tile`.
//!
//! ## Why a separate file?
//!
//! Both `mlir_to_cpp.rs` and `mlir_to_msl.rs` are re-included via
//! `#[path = "..."] mod ...;` from lightweight test crates that have zero
//! third-party dependencies. So this module is **strictly std-only** — no
//! MLIR, LLVM, melior or rustc deps. Any helper that needs richer types
//! belongs in the consuming emitter, not here.
//!
//! ## Scope at extraction time (B1)
//!
//! - `MlirModule`, `MlirFunc`, `MlirGlobal`, `FuncArg` value types
//! - `parse_module` and its parsing primitives
//! - Brace/paren counters used to track block depth
//!
//! `KernelAnalysis` (the cpp-tile pipeline metadata bag) and
//! `determine_element_type` (which depends on it) stay in `mlir_to_cpp.rs`
//! — they are not reusable by the MSL emitter.
//!
//! Per-consumer dead code is expected: when this file is `#[path]`-included
//! from a test crate that exercises only one emitter (e.g.
//! `mlir_to_msl_tests`), helpers used only by the other emitter look
//! unreferenced. The crate-level `allow(dead_code)` keeps those builds clean.

#![allow(dead_code)]

/// Count `{` characters in `s`. Used by parsers to track block depth.
pub fn count_open_braces(s: &str) -> usize {
    s.chars().filter(|&c| c == '{').count()
}

/// Count `}` characters in `s`. Used by parsers to track block depth.
pub fn count_close_braces(s: &str) -> usize {
    s.chars().filter(|&c| c == '}').count()
}

/// Count `(` characters in `s`. Used by parsers to track region nesting.
pub fn count_open_parens(s: &str) -> usize {
    s.chars().filter(|&c| c == '(').count()
}

/// Count `)` characters in `s`. Used by parsers to track region nesting.
pub fn count_close_parens(s: &str) -> usize {
    s.chars().filter(|&c| c == ')').count()
}
/// Map an MLIR scalar type string to a C++ type string.
pub fn mlir_scalar_type_to_cpp(ty: &str) -> String {
    match ty {
        "i8" => "int8_t".to_string(),
        "i16" => "int16_t".to_string(),
        "i32" => "int32_t".to_string(),
        "i64" => "int64_t".to_string(),
        "f32" => "float".to_string(),
        "f16" => "half".to_string(),
        _ => {
            // For array types, return element type
            if let Some(content) = ty
                .strip_prefix("!llvm.array<")
                .and_then(|s| s.strip_suffix('>'))
            {
                if let Some(x_pos) = content.find(" x ") {
                    return mlir_scalar_type_to_cpp(content[x_pos + 3..].trim());
                }
            }
            // tensor<NxT> format from dense attributes
            if let Some(content) = ty.strip_prefix("tensor<").and_then(|s| s.strip_suffix('>')) {
                if let Some(x_pos) = content.find('x') {
                    return mlir_scalar_type_to_cpp(content[x_pos + 1..].trim());
                }
            }
            "int32_t".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// MLIR text parsing
// ---------------------------------------------------------------------------

pub struct MlirFunc {
    pub name: String,
    pub args: Vec<FuncArg>,
    pub is_entry: bool,
    pub body_lines: Vec<String>,
}

pub struct FuncArg {
    pub name: String, // e.g. "%arg0"
    pub ty: String,   // e.g. "!llvm.ptr<1>"
    pub is_gm: bool,  // true if ty contains "ptr<1>"
}

/// A global variable parsed from `llvm.mlir.global` declarations.
pub struct MlirGlobal {
    pub name: String,
    pub ty: String,                // e.g. "i32", "f32", "!llvm.array<5 x i32>"
    pub value: Option<String>,     // e.g. "42", "3.14159", "dense<[10, 20, 30]>"
    pub elem_type: Option<String>, // e.g. "i32" for arrays, None for scalars
    pub count: Option<usize>,      // array element count if applicable
}

pub struct MlirModule {
    pub functions: Vec<MlirFunc>,
    pub globals: Vec<MlirGlobal>,
}

pub fn parse_module(mlir_text: &str) -> Result<MlirModule, String> {
    let mut functions = Vec::new();
    let mut globals = Vec::new();
    let lines: Vec<&str> = mlir_text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Match global: llvm.mlir.global ... @name(...) ... { ... }
        if line.contains("llvm.mlir.global") && line.contains("@") {
            if let Some(g) = parse_global_decl(line) {
                globals.push(g);
            }
            i += 1;
        // Match function header in pretty format: llvm.func @name(args) attributes {...} {
        } else if line.contains("llvm.func @") {
            let name = extract_func_name(line)
                .ok_or_else(|| format!("Cannot parse function name from: {}", line))?;
            let args = extract_func_args(line);
            let is_entry = line.contains("hacc.entry");

            // Collect function body by tracking brace nesting
            let mut body_lines = Vec::new();
            let mut depth = count_open_braces(line) as i32 - count_close_braces(line) as i32;

            i += 1;
            while i < lines.len() && depth > 0 {
                let body_line = lines[i];
                depth += count_open_braces(body_line) as i32;
                depth -= count_close_braces(body_line) as i32;

                if depth > 0 {
                    // Don't include the final closing brace
                    body_lines.push(body_line.trim().to_string());
                }
                i += 1;
            }

            functions.push(MlirFunc {
                name,
                args,
                is_entry,
                body_lines,
            });
        // Match function in MLIR generic format:
        //   "llvm.func"() <{sym_name = "name", function_type = ..., ...}> ({
        //   ^bb0(%arg0: type, ...):
        //     body...
        //   }) {hacc.entry, ...} : () -> ()
        } else if line.starts_with("\"llvm.func\"()") {
            let name = extract_generic_sym_name(line);
            let func_type_args = extract_generic_func_type_args(line);

            // Collect body and the closing line (which has discardable attrs)
            let mut body_lines = Vec::new();
            let mut is_entry = false;
            let mut bb_args = Vec::new();

            // Track parenthesized region nesting: the region starts with ({
            // and ends with }) on a line.
            let mut depth = count_open_parens(line) as i32 - count_close_parens(line) as i32;

            i += 1;
            while i < lines.len() && depth > 0 {
                let body_line = lines[i].trim();
                depth += count_open_parens(body_line) as i32;
                depth -= count_close_parens(body_line) as i32;

                if depth <= 0 {
                    // This is the closing line: }) {hacc.entry, ...} : () -> ()
                    is_entry = body_line.contains("hacc.entry");
                } else if body_line.starts_with("^bb0(") || body_line.starts_with("^bb0:") {
                    // Extract args from ^bb0(%arg0: !llvm.ptr<1>, ...)
                    bb_args = extract_bb_args(body_line);
                } else {
                    body_lines.push(normalize_generic_to_pretty(body_line));
                }
                i += 1;
            }

            // Prefer ^bb0 args (have names), fall back to function_type args
            let args = if !bb_args.is_empty() {
                bb_args
            } else {
                func_type_args
            };

            if let Some(name) = name {
                functions.push(MlirFunc {
                    name,
                    args,
                    is_entry,
                    body_lines,
                });
            }
        } else {
            i += 1;
        }
    }

    Ok(MlirModule { functions, globals })
}

/// Extract `sym_name = "..."` from generic format properties `<{..., sym_name = "foo", ...}>`.
pub fn extract_generic_sym_name(line: &str) -> Option<String> {
    let marker = "sym_name = \"";
    let start = line.find(marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract function argument types from generic format `function_type = !llvm.func<void (ptr<1>, ptr<1>)>`.
/// Returns FuncArgs with synthetic names (%arg0, %arg1, ...).
pub fn extract_generic_func_type_args(line: &str) -> Vec<FuncArg> {
    let marker = "function_type = !llvm.func<";
    let start = match line.find(marker) {
        Some(p) => p + marker.len(),
        None => return Vec::new(),
    };
    let rest = &line[start..];
    // Find the matching > for the outer func type
    let mut depth = 1i32;
    let mut end = 0;
    for (j, ch) in rest.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    end = j;
                    break;
                }
            }
            _ => {}
        }
    }
    let func_sig = &rest[..end]; // e.g. "void (ptr<1>, ptr<1>, ptr<1>)"

    // Find the argument types between ( and )
    let open = match func_sig.find('(') {
        Some(p) => p,
        None => return Vec::new(),
    };
    let arg_str = &func_sig[open + 1..func_sig.len().saturating_sub(1)];
    if arg_str.trim().is_empty() {
        return Vec::new();
    }

    // Split by comma, handling nested <>
    let mut args = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0i32;
    let mut arg_idx = 0;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 => {
                let ty = current.trim().to_string();
                if !ty.is_empty() {
                    let is_gm = ty.contains("ptr<1>");
                    args.push(FuncArg {
                        name: format!("%arg{}", arg_idx),
                        ty,
                        is_gm,
                    });
                    arg_idx += 1;
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let ty = current.trim().to_string();
    if !ty.is_empty() {
        let is_gm = ty.contains("ptr<1>");
        args.push(FuncArg {
            name: format!("%arg{}", arg_idx),
            ty,
            is_gm,
        });
    }
    args
}

/// Extract arguments from a basic block header: ^bb0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>):
pub fn extract_bb_args(line: &str) -> Vec<FuncArg> {
    let open = match line.find('(') {
        Some(p) => p,
        None => return Vec::new(),
    };
    // Find matching close paren
    let mut depth = 0;
    let mut close = open;
    for (j, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }
    let arg_str = &line[open + 1..close];
    if arg_str.trim().is_empty() {
        return Vec::new();
    }

    // Split by comma, handling nested <>
    let mut args = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0i32;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 => {
                if let Some(arg) = parse_single_bb_arg(&current) {
                    args.push(arg);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if let Some(arg) = parse_single_bb_arg(&current) {
        args.push(arg);
    }
    args
}

/// Parse a single BB arg like "%arg1704: !llvm.ptr<1>" into a FuncArg.
pub fn parse_single_bb_arg(s: &str) -> Option<FuncArg> {
    let s = s.trim();
    let colon = s.find(':')?;
    let name = s[..colon].trim().to_string();
    let ty = s[colon + 1..].trim().to_string();
    let is_gm = ty.contains("ptr<1>");
    Some(FuncArg { name, ty, is_gm })
}

/// Normalize a generic-format MLIR body line to pseudo-pretty-format.
///
/// Converts patterns like:
///   `%r = "llvm.call"(%a, %b) <{callee = @func, ...}> : (T1, T2) -> T3`
///   → `%r = llvm.call @func(%a, %b) : (T1, T2) -> T3`
///
///   `%r = "llvm.mlir.constant"() <{value = 42 : i64}> : () -> i32`
///   → `%r = llvm.mlir.constant(42 : i64) : i32`
///
///   `%r = "llvm.load"(%ptr) <{alignment = 4}> : (!llvm.ptr<1>) -> i32`
///   → `%r = llvm.load %ptr : !llvm.ptr<1> -> i32`
///
///   `"llvm.store"(%val, %ptr) <{alignment = 4}> : (i32, !llvm.ptr<1>) -> ()`
///   → `llvm.store %val, %ptr : i32, !llvm.ptr<1>`
///
///   `%r = "llvm.bitcast"(%val) : (i32) -> i32`
///   → `%r = llvm.bitcast %val : i32 to i32`
///
///   `%r = "llvm.alloca"(%size) <{elem_type = i8, ...}> : (i32) -> !llvm.ptr<1>`
///   → `%r = llvm.alloca %size x i8 : (i32) -> !llvm.ptr<1>`
///
///   `"llvm.br"(%a, %b)[^bb1] <{}> : (i32, i32)`
///   → `llvm.br ^bb1(%a, %b : i32, i32)`
///
///   `"llvm.cond_br"(%cond)[^bb1, ^bb2] <{...}> : (i1)`
///   → `llvm.cond_br %cond, ^bb1(...), ^bb2(...)`
///
///   `%r = "llvm.getelementptr"(%ptr, %idx) <{elem_type = f32, ...}> : (TYPE, TYPE) -> TYPE`
///   → `%r = llvm.getelementptr %ptr[%idx] : (TYPE, TYPE) -> TYPE, f32`
pub fn normalize_generic_to_pretty(line: &str) -> String {
    let trimmed = line.trim();

    // Handle "cf.br" and "cf.cond_br" in generic form before the "llvm." check
    if trimmed.contains("\"cf.br\"") || trimmed.contains("\"cf.cond_br\"") {
        return normalize_cf_branch(trimmed);
    }

    // Only process lines containing the generic op format ("op.name")
    if !trimmed.contains("\"llvm.") && !trimmed.contains("\"arith.") {
        return trimmed.to_string();
    }

    // Extract result prefix if present: %name =
    let (result_prefix, rest) = if let Some(eq_pos) = trimmed.find('=') {
        let before = trimmed[..eq_pos].trim();
        if before.starts_with('%') {
            (format!("{} = ", before), trimmed[eq_pos + 1..].trim())
        } else {
            (String::new(), trimmed)
        }
    } else {
        (String::new(), trimmed)
    };

    // --- llvm.call: "llvm.call"(args) <{callee = @func, ...}> : types ---
    if rest.starts_with("\"llvm.call\"") {
        let after_op = &rest["\"llvm.call\"".len()..];
        let operands = extract_paren_contents(after_op);
        let callee = extract_property(rest, "callee");
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.call {}({}) : {}",
            result_prefix,
            callee.unwrap_or_default(),
            operands,
            type_sig
        );
    }

    // --- llvm.mlir.constant: "llvm.mlir.constant"() <{value = V : T}> : () -> RT ---
    if rest.starts_with("\"llvm.mlir.constant\"") {
        if let Some(value) = extract_property(rest, "value") {
            let result_type = extract_arrow_result_type(rest);
            return format!(
                "{}llvm.mlir.constant({}) : {}",
                result_prefix, value, result_type
            );
        }
    }

    // --- llvm.load: "llvm.load"(%ptr) <{...}> : (PT) -> RT ---
    if rest.starts_with("\"llvm.load\"") {
        let after_op = &rest["\"llvm.load\"".len()..];
        let operand = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Convert "(PT) -> RT" to "PT -> RT" for the pretty format parser
        let pretty_type = type_sig.trim_start_matches('(').replacen(") ->", " ->", 1);
        return format!("{}llvm.load {} : {}", result_prefix, operand, pretty_type);
    }

    // --- llvm.store: "llvm.store"(%val, %ptr) <{...}> : (VT, PT) -> () ---
    if rest.starts_with("\"llvm.store\"") {
        let after_op = &rest["\"llvm.store\"".len()..];
        let operands = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Convert operands "a, b" to "a, b" and type "(VT, PT) -> ()" to "VT, PT"
        let types = type_sig
            .split("->")
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        return format!("{}llvm.store {} : {}", result_prefix, operands, types);
    }

    // --- llvm.bitcast: "llvm.bitcast"(%val) : (FT) -> TT ---
    if rest.starts_with("\"llvm.bitcast\"") {
        let after_op = &rest["\"llvm.bitcast\"".len()..];
        let operand = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Parse "(from_type) -> to_type"
        let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
        let from_type = parts
            .first()
            .unwrap_or(&"")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        let to_type = parts.get(1).unwrap_or(&"").trim();
        return format!(
            "{}llvm.bitcast {} : {} to {}",
            result_prefix, operand, from_type, to_type
        );
    }

    // --- llvm.alloca: "llvm.alloca"(%size) <{elem_type = T, ...}> : types ---
    if rest.starts_with("\"llvm.alloca\"") {
        let after_op = &rest["\"llvm.alloca\"".len()..];
        let operand = extract_paren_contents(after_op);
        let elem_type = extract_property(rest, "elem_type").unwrap_or_else(|| "i8".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.alloca {} x {} : {}",
            result_prefix, operand, elem_type, type_sig
        );
    }

    // --- llvm.getelementptr: "llvm.getelementptr"(%ptr, %idx) <{elem_type = T, ...}> : types ---
    if rest.starts_with("\"llvm.getelementptr\"") {
        let after_op = &rest["\"llvm.getelementptr\"".len()..];
        let operands = extract_paren_contents(after_op);
        let parts: Vec<&str> = operands.splitn(2, ',').collect();
        let base = parts.first().unwrap_or(&"").trim();
        let indices = if parts.len() > 1 { parts[1].trim() } else { "" };
        let elem_type = extract_property(rest, "elem_type").unwrap_or_else(|| "i8".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.getelementptr {}[{}] : {}, {}",
            result_prefix, base, indices, type_sig, elem_type
        );
    }

    // --- llvm.extractvalue: "llvm.extractvalue"(%agg) <{position = array<i64: 0>}> : types ---
    if rest.starts_with("\"llvm.extractvalue\"") {
        let after_op = &rest["\"llvm.extractvalue\"".len()..];
        let operand = extract_paren_contents(after_op);
        // Extract position from <{position = array<i64: N>}>
        let pos = extract_property(rest, "position")
            .and_then(|p| {
                // Parse "array<i64: 0>" → "0"
                p.find(':')
                    .map(|c| p[c + 1..].trim_end_matches('>').trim().to_string())
            })
            .unwrap_or_else(|| "0".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.extractvalue {}[{}] : {}",
            result_prefix, operand, pos, type_sig
        );
    }

    // --- llvm.insertvalue: "llvm.insertvalue"(%val, %agg) <{position = ...}> : types ---
    if rest.starts_with("\"llvm.insertvalue\"") {
        let after_op = &rest["\"llvm.insertvalue\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pos = extract_property(rest, "position")
            .and_then(|p| {
                p.find(':')
                    .map(|c| p[c + 1..].trim_end_matches('>').trim().to_string())
            })
            .unwrap_or_else(|| "0".to_string());
        let type_sig = extract_type_signature(rest);
        let parts: Vec<&str> = operands.splitn(2, ',').collect();
        let val = parts.first().unwrap_or(&"").trim();
        let agg = parts.get(1).unwrap_or(&"").trim();
        return format!(
            "{}llvm.insertvalue {} into {}[{}] : {}",
            result_prefix, val, agg, pos, type_sig
        );
    }

    // --- llvm.br: "llvm.br"(args)[^bb] <{}> : types ---
    if rest.starts_with("\"llvm.br\"") {
        let after_op = &rest["\"llvm.br\"".len()..];
        let operands = extract_paren_contents(after_op);
        // Extract successor block from [^bb1]
        let target = extract_bracket_contents(after_op);
        let type_sig = extract_type_signature(rest);
        if operands.is_empty() {
            return format!("{}llvm.br {}", result_prefix, target);
        }
        return format!(
            "{}llvm.br {}({} : {})",
            result_prefix,
            target,
            operands,
            type_sig.trim_start_matches('(').trim_end_matches(')')
        );
    }

    // --- llvm.cond_br: "llvm.cond_br"(%cond, args...)[^t, ^f] <{...}> : types ---
    if rest.starts_with("\"llvm.cond_br\"") {
        let after_op = &rest["\"llvm.cond_br\"".len()..];
        let operands = extract_paren_contents(after_op);
        let successors = extract_bracket_contents(after_op);
        let succ_parts: Vec<&str> = successors.split(',').collect();
        let true_block = succ_parts.first().unwrap_or(&"^bb0").trim();
        let false_block = succ_parts.get(1).unwrap_or(&"^bb0").trim();

        // First operand is condition, rest are block args
        let op_parts: Vec<&str> = operands.splitn(2, ',').collect();
        let cond = op_parts.first().unwrap_or(&"%0").trim();

        // Get operandSegmentSizes to split args between true/false blocks
        // For simplicity, pass all remaining args to the true block
        let remaining_args = if op_parts.len() > 1 {
            op_parts[1].trim()
        } else {
            ""
        };

        if remaining_args.is_empty() {
            return format!(
                "{}llvm.cond_br {}, {}, {}",
                result_prefix, cond, true_block, false_block
            );
        }
        return format!(
            "{}llvm.cond_br {}, {}({}), {}",
            result_prefix, cond, true_block, remaining_args, false_block
        );
    }

    // --- llvm.intr.*: "llvm.intr.X"(args) <{...}> : types ---
    if let Some(intr_start) = rest.find("\"llvm.intr.") {
        let after_quote = &rest[intr_start + 1..];
        let end_quote = after_quote.find('"').unwrap_or(after_quote.len());
        let intr_name = &after_quote[..end_quote]; // e.g. "llvm.intr.umul.with.overflow"
        let after_op_end = &rest[intr_start + end_quote + 2..]; // after closing "
        let operands = extract_paren_contents(after_op_end);
        let type_sig = extract_type_signature(rest);
        return format!("{}{} {} : {}", result_prefix, intr_name, operands, type_sig);
    }

    // --- Unary type cast ops: "llvm.trunc"(%val) <{...}> : (FROM) -> TO ---
    // Covers: trunc, zext, sext, fptoui, fptosi, uitofp, sitofp, fpext, fptrunc, ptrtoint, inttoptr
    {
        let unary_cast_ops = [
            "llvm.trunc",
            "llvm.zext",
            "llvm.sext",
            "llvm.fptoui",
            "llvm.fptosi",
            "llvm.uitofp",
            "llvm.sitofp",
            "llvm.fpext",
            "llvm.fptrunc",
            "llvm.ptrtoint",
            "llvm.inttoptr",
        ];
        for op in &unary_cast_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operand = extract_paren_contents(after_op);
                let type_sig = extract_type_signature(rest);
                // Parse "(FROM) -> TO" into "FROM to TO"
                let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
                let from_type = parts
                    .first()
                    .unwrap_or(&"")
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim();
                let to_type = parts.get(1).unwrap_or(&"").trim();
                return format!(
                    "{}{} {} : {} to {}",
                    result_prefix, op, operand, from_type, to_type
                );
            }
        }
    }

    // --- llvm.fneg: "llvm.fneg"(%val) <{fastmathFlags = ...}> : (f32) -> f32 ---
    // → llvm.fneg %val : f32
    if rest.starts_with("\"llvm.fneg\"") {
        let after_op = &rest["\"llvm.fneg\"".len()..];
        let operand = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!("{}llvm.fneg {} : {}", result_prefix, operand, result_type);
    }

    // --- arith.negf: "arith.negf"(%val) <{fastmath = ...}> : (f32) -> f32 ---
    // → arith.negf %val : f32
    if rest.starts_with("\"arith.negf\"") {
        let after_op = &rest["\"arith.negf\"".len()..];
        let operand = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!("{}arith.negf {} : {}", result_prefix, operand, result_type);
    }

    // --- Binary ops: "llvm.add"(%a, %b) <{overflowFlags = ...}> : (T, T) -> T ---
    // Covers: add, sub, mul, udiv, sdiv, urem, srem, and, or, xor, shl, lshr, ashr
    {
        let binary_ops = [
            "llvm.add",
            "llvm.sub",
            "llvm.mul",
            "llvm.udiv",
            "llvm.sdiv",
            "llvm.urem",
            "llvm.srem",
            "llvm.and",
            "llvm.or",
            "llvm.xor",
            "llvm.shl",
            "llvm.lshr",
            "llvm.ashr",
            "llvm.fdiv",
            "llvm.frem",
        ];
        for op in &binary_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operands = extract_paren_contents(after_op);
                let result_type = extract_arrow_result_type(rest);
                return format!("{}{} {} : {}", result_prefix, op, operands, result_type);
            }
        }
    }

    // --- llvm.icmp: "llvm.icmp"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    // → llvm.icmp "pred_name" %a, %b : T
    if rest.starts_with("\"llvm.icmp\"") {
        let after_op = &rest["\"llvm.icmp\"".len()..];
        let operands = extract_paren_contents(after_op);
        // Extract predicate number from <{predicate = N : i64}>
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "eq",
                1 => "ne",
                2 => "slt",
                3 => "sle",
                4 => "sgt",
                5 => "sge",
                6 => "ult",
                7 => "ule",
                8 => "ugt",
                9 => "uge",
                _ => "eq",
            }
        } else {
            "eq"
        };
        // Get the input type (first type in the signature)
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("i32")
            .trim()
            .to_string();
        return format!(
            "{}llvm.icmp \"{}\" {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- llvm.select: "llvm.select"(%c, %a, %b) <{...}> : (i1, T, T) -> T ---
    if rest.starts_with("\"llvm.select\"") {
        let after_op = &rest["\"llvm.select\"".len()..];
        let operands = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!(
            "{}llvm.select {} : {}",
            result_prefix, operands, result_type
        );
    }

    // --- arith.constant: "arith.constant"() <{value = V : T}> : () -> RT ---
    // Normalize to arith.constant(V : T) : RT (matching llvm.mlir.constant format)
    if rest.starts_with("\"arith.constant\"") {
        if let Some(value) = extract_property(rest, "value") {
            let result_type = extract_arrow_result_type(rest);
            return format!(
                "{}arith.constant({}) : {}",
                result_prefix, value, result_type
            );
        }
    }

    // --- arith binary ops: "arith.addi"(%a, %b) : (T, T) -> T ---
    // Integer: addi, subi, muli, divui, divsi, remui, remsi, andi, ori, xori, shli, shrui, shrsi
    // Float: addf, subf, mulf, divf, remf
    {
        let arith_binary_ops = [
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divui",
            "arith.divsi",
            "arith.remui",
            "arith.remsi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrui",
            "arith.shrsi",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.remf",
        ];
        for op in &arith_binary_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operands = extract_paren_contents(after_op);
                let result_type = extract_arrow_result_type(rest);
                return format!("{}{} {} : {}", result_prefix, op, operands, result_type);
            }
        }
    }

    // --- arith.cmpi: "arith.cmpi"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    // → arith.cmpi pred_name, %a, %b : T
    if rest.starts_with("\"arith.cmpi\"") {
        let after_op = &rest["\"arith.cmpi\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "eq",
                1 => "ne",
                2 => "slt",
                3 => "sle",
                4 => "sgt",
                5 => "sge",
                6 => "ult",
                7 => "ule",
                8 => "ugt",
                9 => "uge",
                _ => "eq",
            }
        } else {
            "eq"
        };
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("i32")
            .trim()
            .to_string();
        return format!(
            "{}arith.cmpi {}, {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- arith.cmpf: "arith.cmpf"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    if rest.starts_with("\"arith.cmpf\"") {
        let after_op = &rest["\"arith.cmpf\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "false",
                1 => "oeq",
                2 => "ogt",
                3 => "oge",
                4 => "olt",
                5 => "ole",
                6 => "one",
                7 => "ord",
                8 => "ueq",
                9 => "ugt",
                10 => "uge",
                11 => "ult",
                12 => "ule",
                13 => "une",
                14 => "uno",
                15 => "true",
                _ => "oeq",
            }
        } else {
            "oeq"
        };
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("f32")
            .trim()
            .to_string();
        return format!(
            "{}arith.cmpf {}, {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- arith unary cast ops: "arith.extui"(%val) : (FROM) -> TO ---
    // Covers: extui, extsi, trunci, sitofp, uitofp, fptosi, fptoui, fpext, fptrunc, bitcast, index_cast
    {
        let arith_cast_ops = [
            "arith.extui",
            "arith.extsi",
            "arith.trunci",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "arith.fpext",
            "arith.fptrunc",
            "arith.bitcast",
            "arith.index_cast",
        ];
        for op in &arith_cast_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operand = extract_paren_contents(after_op);
                let type_sig = extract_type_signature(rest);
                let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
                let from_type = parts
                    .first()
                    .unwrap_or(&"")
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim();
                let to_type = parts.get(1).unwrap_or(&"").trim();
                return format!(
                    "{}{} {} : {} to {}",
                    result_prefix, op, operand, from_type, to_type
                );
            }
        }
    }

    // --- arith.select: "arith.select"(%c, %a, %b) : (i1, T, T) -> T ---
    if rest.starts_with("\"arith.select\"") {
        let after_op = &rest["\"arith.select\"".len()..];
        let operands = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!(
            "{}arith.select {} : {}",
            result_prefix, operands, result_type
        );
    }

    // --- llvm.return: "llvm.return"(%val) : (T) -> () → llvm.return %val : T ---
    // Void: "llvm.return"() : () -> () → llvm.return
    if rest.starts_with("\"llvm.return\"") {
        let after_op = &rest["\"llvm.return\"".len()..];
        let operands = extract_paren_contents(after_op);
        if operands.is_empty() {
            return "llvm.return".to_string();
        }
        // Extract the result type from (T) -> ()
        let type_sig = extract_type_signature(rest);
        // type_sig is "(f32) -> ()" — extract just "f32"
        let ret_type = type_sig
            .split("->")
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        return format!("llvm.return {} : {}", operands, ret_type);
    }

    // --- Fallback: strip quotes from op name ---
    // "llvm.X"(args) ... → llvm.X args ...
    if let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        if let Some(end) = after.find('"') {
            let op_name = &after[..end];
            let remainder = &after[end + 1..];
            return format!("{}{}{}", result_prefix, op_name, remainder);
        }
    }

    trimmed.to_string()
}

/// Normalize generic-form `"cf.br"` and `"cf.cond_br"` to pretty form.
///
///   `"cf.br"()[^bb1] <{}> : () -> ()` → `cf.br ^bb1`
///   `"cf.br"(%a, %b)[^bb1] <{}> : (i32, i32) -> ()` → `cf.br ^bb1(%a : i32, %b : i32)`
///   `"cf.cond_br"(%c)[^bb1, ^bb2] <{...}> : (i1) -> ()` → `cf.cond_br %c, ^bb1, ^bb2`
///   `"cf.cond_br"(%c, %a, %b)[^bb1, ^bb2] <{...}> : (i1, i32, i32) -> ()` → with args
pub fn normalize_cf_branch(line: &str) -> String {
    if let Some(pos) = line.find("\"cf.cond_br\"") {
        let after = &line[pos + "\"cf.cond_br\"".len()..];
        // Extract operands (%c, %a, ...) from parentheses
        let operands = extract_paren_contents(after);
        let operand_list: Vec<&str> = if operands.is_empty() {
            Vec::new()
        } else {
            operands.split(',').map(|s| s.trim()).collect()
        };
        // Extract successors [^bb1, ^bb2] from brackets
        let bracket_open = after.find('[').unwrap_or(0);
        let bracket_close = after.find(']').unwrap_or(after.len());
        let successors_str = &after[bracket_open + 1..bracket_close];
        let successors: Vec<&str> = successors_str.split(',').map(|s| s.trim()).collect();
        let cond = operand_list.first().copied().unwrap_or("%unknown");
        let bb_true = successors.first().copied().unwrap_or("^bb0");
        let bb_false = successors.get(1).copied().unwrap_or("^bb0");
        return format!("cf.cond_br {}, {}, {}", cond, bb_true, bb_false);
    }
    if let Some(pos) = line.find("\"cf.br\"") {
        let after = &line[pos + "\"cf.br\"".len()..];
        let operands = extract_paren_contents(after);
        let operand_list: Vec<&str> = if operands.is_empty() {
            Vec::new()
        } else {
            operands.split(',').map(|s| s.trim()).collect()
        };
        // Extract successor [^bbN] from brackets
        let bracket_open = after.find('[').unwrap_or(0);
        let bracket_close = after.find(']').unwrap_or(after.len());
        let successor = after[bracket_open + 1..bracket_close].trim();
        // Extract type annotations to pair with operands
        let type_sig = extract_type_signature(after);
        let type_list: Vec<&str> = if type_sig.is_empty() {
            Vec::new()
        } else {
            let input_types = type_sig.split("->").next().unwrap_or("").trim();
            let inner = input_types.trim_start_matches('(').trim_end_matches(')');
            if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|s| s.trim()).collect()
            }
        };
        if operand_list.is_empty() {
            return format!("cf.br {}", successor);
        }
        // Build block args: ^bbN(%a : type, %b : type)
        let args: Vec<String> = operand_list
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let ty = type_list.get(i).copied().unwrap_or("i64");
                format!("{} : {}", op, ty)
            })
            .collect();
        return format!("cf.br {}({})", successor, args.join(", "));
    }
    line.to_string()
}

/// Extract content between first pair of parentheses: "(a, b)" → "a, b"
pub fn extract_paren_contents(s: &str) -> String {
    let open = match s.find('(') {
        Some(p) => p,
        None => return String::new(),
    };
    let mut depth = 0;
    let mut close = open;
    for (j, ch) in s[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }
    s[open + 1..close].trim().to_string()
}

/// Extract content between first pair of square brackets: "[^bb1, ^bb2]" → "^bb1, ^bb2"
pub fn extract_bracket_contents(s: &str) -> String {
    let open = match s.find('[') {
        Some(p) => p,
        None => return String::new(),
    };
    let close = match s[open..].find(']') {
        Some(p) => open + p,
        None => return String::new(),
    };
    s[open + 1..close].trim().to_string()
}

/// Extract a named property from `<{..., name = VALUE, ...}>`.
pub fn extract_property(line: &str, name: &str) -> Option<String> {
    let pattern = format!("{} = ", name);
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..];
    // Value extends until the next comma, or closing }>, or end of line
    // Handle nested <> and () and {}
    let mut depth_angle = 0i32;
    let mut depth_brace = 0i32;
    let mut end = rest.len();
    for (j, ch) in rest.char_indices() {
        match ch {
            '<' => depth_angle += 1,
            '>' if depth_angle > 0 => depth_angle -= 1,
            '{' => depth_brace += 1,
            '}' if depth_brace > 0 => depth_brace -= 1,
            '}' if depth_brace == 0 => {
                end = j;
                break;
            }
            ',' if depth_angle == 0 && depth_brace == 0 => {
                end = j;
                break;
            }
            _ => {}
        }
    }
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Extract type signature after the last ` : ` (outside nested structures).
/// e.g., `... : (i32, !llvm.ptr<1>) -> i32` → `(i32, !llvm.ptr<1>) -> i32`
pub fn extract_type_signature(line: &str) -> String {
    // Find the last ` : ` that's outside of <{}>
    let bytes = line.as_bytes();
    let mut depth_angle = 0i32;
    let mut depth_brace = 0i32;
    let mut last_colon = None;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth_angle += 1,
            b'>' if depth_angle > 0 => depth_angle -= 1,
            b'{' => depth_brace += 1,
            b'}' if depth_brace > 0 => depth_brace -= 1,
            b':' if depth_angle == 0 && depth_brace == 0 => {
                // Check for " : " pattern
                if i > 0 && i + 1 < bytes.len() && bytes[i - 1] == b' ' && bytes[i + 1] == b' ' {
                    last_colon = Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    match last_colon {
        Some(pos) => line[pos + 2..].trim().to_string(),
        None => String::new(),
    }
}

/// Extract the result type after `->` in a type signature.
/// e.g., `() -> i32` → `i32`
pub fn extract_arrow_result_type(line: &str) -> String {
    if let Some(arrow) = line.rfind("-> ") {
        line[arrow + 3..].trim().to_string()
    } else {
        "i32".to_string()
    }
}

/// Parse a single `llvm.mlir.global` line into an MlirGlobal.
/// Examples:
///   llvm.mlir.global constant @MAGIC {value = 42 : i32, ...}
///   llvm.mlir.global constant @TABLE {value = dense<[10, 20, 30]> : tensor<3xi32>, ...}
pub fn parse_global_decl(line: &str) -> Option<MlirGlobal> {
    // Extract symbol name: @name
    let at_pos = line.find('@')?;
    let rest = &line[at_pos + 1..];
    let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')?;
    let name = rest[..end].to_string();

    // Extract type from global_type attribute or from value attribute
    let mut ty = String::new();
    let mut value = None;
    let mut elem_type = None;
    let mut count = None;

    // Try to extract value from the line
    if let Some(val_pos) = line.find("value = ") {
        let val_rest = &line[val_pos + 8..];
        // Parse the value — could be "42 : i32" or "dense<[...]> : tensor<NxT>"
        if val_rest.starts_with("dense<") {
            // Dense array: dense<[v1, v2, ...]> : tensor<NxT>
            if let Some(bracket_end) = val_rest.find("]>") {
                let dense_content = &val_rest[6..bracket_end + 1]; // [v1, v2, ...]
                                                                   // Find the type after " : "
                let type_marker = &val_rest[bracket_end + 2..];
                if let Some(colon_pos) = type_marker.find(" : ") {
                    let type_str = type_marker[colon_pos + 3..].trim();
                    // Trim trailing }, etc.
                    let type_str = type_str
                        .trim_end_matches(|c: char| c == '}' || c == ',' || c.is_whitespace());
                    ty = type_str.to_string();

                    // Parse array element type and count from "!llvm.array<N x T>" or "tensor<NxT>"
                    if let Some(arr_content) = type_str
                        .strip_prefix("!llvm.array<")
                        .and_then(|s| s.strip_suffix('>'))
                    {
                        if let Some(x_pos) = arr_content.find(" x ") {
                            count = arr_content[..x_pos].trim().parse().ok();
                            elem_type = Some(arr_content[x_pos + 3..].trim().to_string());
                        }
                    } else if let Some(tensor_content) = type_str
                        .strip_prefix("tensor<")
                        .and_then(|s| s.strip_suffix('>'))
                    {
                        // tensor<NxT> format (no spaces around 'x')
                        if let Some(x_pos) = tensor_content.find('x') {
                            count = tensor_content[..x_pos].trim().parse().ok();
                            elem_type = Some(tensor_content[x_pos + 1..].trim().to_string());
                        }
                    }
                }
                value = Some(format!("dense<{}>", dense_content));
            }
        } else {
            // Scalar: "42 : i32" or "3.14159 : f32"
            if let Some(colon_pos) = val_rest.find(" : ") {
                let scalar_val = val_rest[..colon_pos].trim().to_string();
                let type_str = val_rest[colon_pos + 3..].trim();
                let type_str =
                    type_str.trim_end_matches(|c: char| c == '}' || c == ',' || c.is_whitespace());
                ty = type_str.to_string();
                value = Some(scalar_val);
            }
        }
    }

    // If we couldn't determine the type from value, try global_type attribute
    if ty.is_empty() {
        if let Some(gt_pos) = line.find("global_type = ") {
            let gt_rest = &line[gt_pos + 14..];
            let gt_end = gt_rest
                .find(|c: char| c == ',' || c == '}')
                .unwrap_or(gt_rest.len());
            ty = gt_rest[..gt_end].trim().to_string();
        }
    }

    Some(MlirGlobal {
        name,
        ty,
        value,
        elem_type,
        count,
    })
}

/// Keep the old API for backward compatibility (used by tests etc.)
pub fn parse_functions(mlir_text: &str) -> Result<Vec<MlirFunc>, String> {
    Ok(parse_module(mlir_text)?.functions)
}

pub fn extract_func_name(line: &str) -> Option<String> {
    let at_pos = line.find("@")?;
    let rest = &line[at_pos + 1..];
    let end = rest.find('(')?;
    Some(rest[..end].to_string())
}

pub fn extract_func_args(line: &str) -> Vec<FuncArg> {
    let mut args = Vec::new();

    // Find the argument list between the first ( and matching )
    let open = match line.find('(') {
        Some(p) => p,
        None => return args,
    };

    let mut depth = 0;
    let mut close = open;
    for (j, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }

    let arg_str = &line[open + 1..close];
    if arg_str.trim().is_empty() {
        return args;
    }

    // Split by comma, handling nested <> and ()
    let mut arg_parts = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0;
    let mut paren_depth = 0;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 && paren_depth == 0 => {
                arg_parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        arg_parts.push(current.trim().to_string());
    }

    for part in &arg_parts {
        // Format: %argN: TYPE
        let colon = match part.find(':') {
            Some(p) => p,
            None => continue,
        };
        let name = part[..colon].trim().to_string();
        let ty = part[colon + 1..].trim().to_string();
        let is_gm = ty.contains("ptr<1>");
        args.push(FuncArg { name, ty, is_gm });
    }

    args
}

// ---- additional helpers used by mlir_to_pto / mlir_to_msl (ds4-msl-pipeline) ----

pub fn extract_result_ssa(line: &str) -> Option<String> {
    let line = line.trim();
    if line.starts_with('%') {
        let end = line.find(" = ")?;
        Some(line[..end].trim().to_string())
    } else {
        None
    }
}

pub fn extract_call_args(line: &str) -> Option<Vec<String>> {
    let open = line.find('(')?;
    let args_section = &line[open + 1..];
    let close = args_section.find(") :")?;
    let inner = &args_section[..close];
    Some(inner.split(',').map(|s| s.trim().to_string()).collect())
}

pub fn parse_const_arg(s: &str) -> u32 {
    let s = s.trim();
    if let Ok(n) = s.parse::<u32>() {
        return n;
    }
    let digits_start = if let Some(rest) = s.strip_prefix("%c") {
        rest
    } else if let Some(rest) = s.strip_prefix('%') {
        rest
    } else {
        s
    };
    let digit_str: String = digits_start
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if !digit_str.is_empty() {
        return digit_str.parse().unwrap_or(0);
    }
    0
}

pub fn is_builtin_helper(name: &str) -> bool {
    matches!(
        name,
        "get_block_idx"
            | "get_block_num"
            | "get_sub_block_idx"
            | "get_sub_block_num"
            | "__rust_eh_personality"
    ) || name.starts_with("__tile_")
        || name.starts_with("__rust_")
}

// ---------------------------------------------------------------------------
// Unit tests for the shared, std-only MLIR parser helpers. These run wherever
// this file is `#[path]`-included (tile_spec, codegen_tests, the mlir_to_*_tests
// crates) — NO LLVM toolchain needed — and pin the open emit surface's parser
// contract.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn brace_and_paren_counts() {
        assert_eq!(count_open_braces("a { b { c"), 2);
        assert_eq!(count_close_braces("} } x }"), 3);
        assert_eq!(count_open_parens("f((x)"), 2);
        assert_eq!(count_close_parens("f((x))"), 2);
        assert_eq!(count_open_braces(""), 0);
    }

    #[test]
    fn scalar_type_to_cpp_primitives() {
        assert_eq!(mlir_scalar_type_to_cpp("i8"), "int8_t");
        assert_eq!(mlir_scalar_type_to_cpp("i16"), "int16_t");
        assert_eq!(mlir_scalar_type_to_cpp("i32"), "int32_t");
        assert_eq!(mlir_scalar_type_to_cpp("i64"), "int64_t");
        assert_eq!(mlir_scalar_type_to_cpp("f32"), "float");
        assert_eq!(mlir_scalar_type_to_cpp("f16"), "half");
    }

    #[test]
    fn scalar_type_to_cpp_arrays_and_tensors_and_default() {
        assert_eq!(mlir_scalar_type_to_cpp("!llvm.array<5 x f32>"), "float");
        assert_eq!(mlir_scalar_type_to_cpp("tensor<3xi16>"), "int16_t");
        // Unknown type falls back to int32_t.
        assert_eq!(mlir_scalar_type_to_cpp("something_else"), "int32_t");
    }

    #[test]
    fn extract_generic_sym_name_present_and_absent() {
        let line = "llvm.mlir.global constant @x() {sym_name = \"foo\", other = 1}";
        assert_eq!(extract_generic_sym_name(line).as_deref(), Some("foo"));
        assert_eq!(extract_generic_sym_name("no marker here"), None);
    }

    #[test]
    fn extract_generic_func_type_args_counts_gm_ptrs() {
        let line = "x {function_type = !llvm.func<void (ptr<1>, ptr<1>, i32)>} y";
        let args = extract_generic_func_type_args(line);
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].name, "%arg0");
        assert!(args[0].is_gm); // ptr<1>
        assert!(args[1].is_gm);
        assert!(!args[2].is_gm); // i32
                                 // No marker -> empty.
        assert!(extract_generic_func_type_args("nope").is_empty());
        // Empty arg list -> empty.
        assert!(extract_generic_func_type_args("function_type = !llvm.func<void ()>").is_empty());
    }

    #[test]
    fn extract_func_name_and_args() {
        let line = "llvm.func @kernel(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {}";
        assert_eq!(extract_func_name(line).as_deref(), Some("kernel"));
        let args = extract_func_args(line);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].name, "%arg0");
        assert!(args[0].is_gm);
        assert_eq!(args[1].ty, "i32");
        assert!(!args[1].is_gm);
        // No '@' -> None.
        assert_eq!(extract_func_name("plain line"), None);
        // No parens -> empty args.
        assert!(extract_func_args("@kernel no parens").is_empty());
    }

    #[test]
    fn extract_bb_args_named() {
        let args = extract_bb_args("^bb0(%a: !llvm.ptr<1>, %b: f32):");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].name, "%a");
        assert!(args[0].is_gm);
        assert_eq!(args[1].ty, "f32");
        assert!(extract_bb_args("^bb0:").is_empty());
    }

    #[test]
    fn extract_paren_and_bracket_contents() {
        assert_eq!(extract_paren_contents("call(a, (b), c) : x"), "a, (b), c");
        assert_eq!(extract_paren_contents("no parens"), "");
        assert_eq!(extract_bracket_contents("br [^bb1, ^bb2]"), "^bb1, ^bb2");
        assert_eq!(extract_bracket_contents("no brackets"), "");
    }

    #[test]
    fn extract_property_handles_nesting() {
        let line = "op <{value = 42 : i32, other = dense<[1, 2]>}>";
        assert_eq!(extract_property(line, "value").as_deref(), Some("42 : i32"));
        assert_eq!(extract_property(line, "missing"), None);
    }

    #[test]
    fn extract_type_signature_after_last_colon() {
        let line = "%r = llvm.call @f(%a) : (i32, !llvm.ptr<1>) -> i32";
        assert_eq!(extract_type_signature(line), "(i32, !llvm.ptr<1>) -> i32");
        assert_eq!(extract_type_signature("no colon here"), "");
    }

    #[test]
    fn extract_arrow_result_type_present_and_default() {
        assert_eq!(extract_arrow_result_type("(i32) -> f32"), "f32");
        // No arrow -> defaults to i32.
        assert_eq!(extract_arrow_result_type("(i32)"), "i32");
    }

    #[test]
    fn extract_result_ssa_and_call_args() {
        assert_eq!(
            extract_result_ssa("%t1 = llvm.call @f(%a) : (i32) -> i32").as_deref(),
            Some("%t1")
        );
        assert_eq!(extract_result_ssa("llvm.return"), None);
        let args = extract_call_args("llvm.call @f(%a, %b, %c) : (i32, i32, i32) -> i32").unwrap();
        assert_eq!(args, vec!["%a", "%b", "%c"]);
        assert_eq!(extract_call_args("no parens"), None);
    }

    #[test]
    fn parse_const_arg_decimal_and_named() {
        assert_eq!(parse_const_arg("1024"), 1024);
        assert_eq!(parse_const_arg("%c256"), 256);
        assert_eq!(parse_const_arg("%42"), 42);
        assert_eq!(parse_const_arg("%arg0"), 0); // no leading digits
        assert_eq!(parse_const_arg("   7  "), 7);
    }

    #[test]
    fn is_builtin_helper_matches() {
        assert!(is_builtin_helper("get_block_idx"));
        assert!(is_builtin_helper("get_sub_block_num"));
        assert!(is_builtin_helper("__tile_load_f32"));
        assert!(is_builtin_helper("__rust_alloc"));
        assert!(!is_builtin_helper("my_kernel"));
    }

    #[test]
    fn parse_global_decl_scalar_and_dense() {
        let scalar =
            parse_global_decl("llvm.mlir.global constant @MAGIC() {value = 42 : i32} : i32")
                .expect("scalar global");
        assert_eq!(scalar.name, "MAGIC");
        assert_eq!(scalar.value.as_deref(), Some("42"));

        let dense = parse_global_decl(
            "llvm.mlir.global constant @TABLE() {value = dense<[10, 20, 30]> : tensor<3xi32>}",
        )
        .expect("dense global");
        assert_eq!(dense.name, "TABLE");
        assert_eq!(dense.count, Some(3));
        // No '@' -> None.
        assert!(parse_global_decl("not a global").is_none());
    }

    #[test]
    fn parse_module_extracts_function_and_entry() {
        let mlir = r#"
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return
  }
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions[0];
        assert_eq!(f.name, "softmax_1d");
        assert!(f.is_entry, "hacc.entry must mark the function as an entry");
        assert_eq!(f.args.len(), 2);
        assert!(f.args[0].is_gm);
    }

    #[test]
    fn parse_module_extracts_global() {
        let mlir = r#"
module {
  llvm.mlir.global constant @TBL() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>}
  llvm.func @k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    llvm.return
  }
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.globals.len(), 1);
        assert_eq!(m.globals[0].name, "TBL");
    }

    // ── normalize_generic_to_pretty: generic-print-form → pretty-form coverage ──
    // The generic MLIR print form (`"op.name"(args) <{attrs}> : types`) is rarely
    // emitted but must round-trip into the pretty form the rest of the parser
    // consumes. Each case below feeds one generic-form line per op family and
    // asserts the exact pretty-form normalization (OSS ratchet — pure string fn).
    #[test]
    fn test_normalize_generic_op_families() {
        let cases: &[(&str, &str)] = &[
            // call
            (
                "%0 = \"llvm.call\"(%a, %b) <{callee = @foo}> : (i32, i32) -> i32",
                "llvm.call @foo(%a, %b) : (i32, i32) -> i32",
            ),
            // constant
            (
                "%1 = \"llvm.mlir.constant\"() <{value = 7 : i32}> : () -> i32",
                "llvm.mlir.constant(7 : i32) : i32",
            ),
            // load
            (
                "%2 = \"llvm.load\"(%p) <{}> : (!llvm.ptr) -> f32",
                "llvm.load %p : !llvm.ptr -> f32",
            ),
            // store
            (
                "\"llvm.store\"(%v, %p) <{}> : (f32, !llvm.ptr) -> ()",
                "llvm.store %v, %p",
            ),
            // bitcast
            (
                "%3 = \"llvm.bitcast\"(%x) <{}> : (i32) -> f32",
                "llvm.bitcast %x : i32 to f32",
            ),
            // alloca
            (
                "%4 = \"llvm.alloca\"(%n) <{elem_type = f32}> : (i64) -> !llvm.ptr",
                "llvm.alloca %n x f32",
            ),
            // getelementptr
            (
                "%5 = \"llvm.getelementptr\"(%p, %i) <{elem_type = f32}> : (!llvm.ptr, i64) -> !llvm.ptr",
                "llvm.getelementptr %p[%i]",
            ),
            // extractvalue
            (
                "%6 = \"llvm.extractvalue\"(%agg) <{position = array<i64: 1>}> : (!llvm.struct) -> i32",
                "llvm.extractvalue %agg[1]",
            ),
            // insertvalue
            (
                "%7 = \"llvm.insertvalue\"(%v, %agg) <{position = array<i64: 0>}> : (i32, !llvm.struct) -> !llvm.struct",
                "llvm.insertvalue %v into %agg[0]",
            ),
            // unconditional br (no operands)
            (
                "\"llvm.br\"()[^bb1] <{}> : () -> ()",
                "llvm.br ^bb1",
            ),
            // unary cast (one of the table)
            (
                "%8 = \"llvm.sext\"(%x) <{}> : (i16) -> i32",
                "llvm.sext %x : i16 to i32",
            ),
            // fneg
            (
                "%9 = \"llvm.fneg\"(%x) <{}> : (f32) -> f32",
                "llvm.fneg %x : f32",
            ),
            // arith.negf
            (
                "%10 = \"arith.negf\"(%x) <{}> : (f32) -> f32",
                "arith.negf %x : f32",
            ),
            // binary op (table)
            (
                "%11 = \"llvm.add\"(%a, %b) <{}> : (i32, i32) -> i32",
                "llvm.add %a, %b : i32",
            ),
            // icmp with predicate 4 (sgt)
            (
                "%12 = \"llvm.icmp\"(%a, %b) <{predicate = 4 : i64}> : (i32, i32) -> i1",
                "llvm.icmp \"sgt\" %a, %b : i32",
            ),
            // select
            (
                "%13 = \"llvm.select\"(%c, %a, %b) <{}> : (i1, i32, i32) -> i32",
                "llvm.select %c, %a, %b : i32",
            ),
            // arith.constant
            (
                "%14 = \"arith.constant\"() <{value = 3 : index}> : () -> index",
                "arith.constant(3 : index) : index",
            ),
            // arith binary (table)
            (
                "%15 = \"arith.addf\"(%a, %b) <{}> : (f32, f32) -> f32",
                "arith.addf %a, %b : f32",
            ),
            // return with value
            (
                "\"llvm.return\"(%v) <{}> : (i32) -> ()",
                "llvm.return %v : i32",
            ),
            // void return
            (
                "\"llvm.return\"() <{}> : () -> ()",
                "llvm.return",
            ),
        ];
        for (input, expect) in cases {
            let got = normalize_generic_to_pretty(input);
            assert!(
                got.contains(expect),
                "normalize_generic_to_pretty({input:?})\n  got:    {got:?}\n  expect substring: {expect:?}"
            );
            // determinism
            assert_eq!(
                got,
                normalize_generic_to_pretty(input),
                "non-deterministic for {input:?}"
            );
        }
    }

    #[test]
    fn test_normalize_generic_passthrough_and_fallback() {
        // A pretty-form line (no quoted generic op) passes through unchanged.
        let pretty = "%0 = llvm.add %a, %b : i32";
        assert_eq!(normalize_generic_to_pretty(pretty), pretty);
        // Unknown quoted op hits the quote-stripping fallback.
        let unknown = "%0 = \"llvm.frobnicate\"(%a) : (i32) -> i32";
        let got = normalize_generic_to_pretty(unknown);
        assert!(
            got.contains("llvm.frobnicate"),
            "fallback should strip quotes: {got}"
        );
        assert!(
            !got.contains("\"llvm.frobnicate\""),
            "fallback should remove quotes: {got}"
        );
    }

    #[test]
    fn test_normalize_cf_branch_forms() {
        // Unconditional cf.br with no args.
        let br = "\"cf.br\"()[^bb1] <{}> : () -> ()";
        assert_eq!(normalize_cf_branch(br), "cf.br ^bb1");
        // cf.br with block args carries the operand:type pairs.
        let br_args = "\"cf.br\"(%a, %b)[^bb2] <{}> : (i32, i64) -> ()";
        let got = normalize_cf_branch(br_args);
        assert!(got.contains("cf.br ^bb2("), "missing block args: {got}");
        assert!(
            got.contains("%a : i32") && got.contains("%b : i64"),
            "missing typed args: {got}"
        );
        // cf.cond_br emits cond + two successors.
        let cbr = "\"cf.cond_br\"(%c)[^bbt, ^bbf] <{}> : (i1) -> ()";
        assert_eq!(normalize_cf_branch(cbr), "cf.cond_br %c, ^bbt, ^bbf");
        // routed through the dispatcher in normalize_generic_to_pretty too.
        assert_eq!(
            normalize_generic_to_pretty(cbr),
            "cf.cond_br %c, ^bbt, ^bbf"
        );
    }

    #[test]
    fn test_normalize_generic_control_flow_and_intr() {
        // llvm.br carrying block args → typed arg list.
        let br_args = "\"llvm.br\"(%a, %b)[^bb1] <{}> : (i32, i64) -> ()";
        let got = normalize_generic_to_pretty(br_args);
        assert!(got.contains("llvm.br ^bb1("), "br with args: {got}");
        assert!(
            got.contains("i32") && got.contains("i64"),
            "br arg types: {got}"
        );

        // llvm.cond_br with no extra args → cond + two successors.
        let cbr = "\"llvm.cond_br\"(%c)[^bbt, ^bbf] <{}> : (i1) -> ()";
        assert_eq!(
            normalize_generic_to_pretty(cbr),
            "llvm.cond_br %c, ^bbt, ^bbf"
        );

        // llvm.cond_br with block args after the condition.
        let cbr_args = "\"llvm.cond_br\"(%c, %x)[^bbt, ^bbf] <{}> : (i1, i32) -> ()";
        let got = normalize_generic_to_pretty(cbr_args);
        assert!(
            got.contains("llvm.cond_br %c, ^bbt("),
            "cond_br args: {got}"
        );

        // llvm.intr.* passthrough.
        let intr = "%0 = \"llvm.intr.smax\"(%a, %b) <{}> : (i32, i32) -> i32";
        let got = normalize_generic_to_pretty(intr);
        assert!(got.contains("llvm.intr.smax"), "intr: {got}");
    }

    #[test]
    fn test_normalize_generic_icmp_predicates() {
        // Every signed/unsigned integer compare predicate maps to its name.
        let cases: &[(u32, &str)] = &[
            (0, "eq"),
            (1, "ne"),
            (2, "slt"),
            (3, "sle"),
            (5, "sge"),
            (6, "ult"),
            (7, "ule"),
            (8, "ugt"),
            (9, "uge"),
        ];
        for (n, name) in cases {
            let line = format!(
                "%0 = \"llvm.icmp\"(%a, %b) <{{predicate = {n} : i64}}> : (i32, i32) -> i1"
            );
            let got = normalize_generic_to_pretty(&line);
            assert!(
                got.contains(&format!("llvm.icmp \"{name}\"")),
                "icmp pred {n} → {name}: {got}"
            );
        }
        // Out-of-range predicate falls back to "eq".
        let oob = "%0 = \"llvm.icmp\"(%a, %b) <{predicate = 99 : i64}> : (i32, i32) -> i1";
        assert!(
            normalize_generic_to_pretty(oob).contains("\"eq\""),
            "icmp oob → eq"
        );
    }

    #[test]
    fn test_normalize_generic_arith_cmp() {
        // arith.cmpi with a signed predicate.
        let cmpi = "%0 = \"arith.cmpi\"(%a, %b) <{predicate = 4 : i64}> : (i32, i32) -> i1";
        let got = normalize_generic_to_pretty(cmpi);
        assert!(got.contains("arith.cmpi sgt, %a, %b : i32"), "cmpi: {got}");

        // arith.cmpf with an ordered predicate (olt = 4).
        let cmpf = "%1 = \"arith.cmpf\"(%a, %b) <{predicate = 4 : i64}> : (f32, f32) -> i1";
        let got = normalize_generic_to_pretty(cmpf);
        assert!(got.contains("arith.cmpf olt, %a, %b : f32"), "cmpf: {got}");

        // arith.cmpf "true"/"false" sentinel predicates.
        let cmpf_true = "%2 = \"arith.cmpf\"(%a, %b) <{predicate = 15 : i64}> : (f32, f32) -> i1";
        assert!(
            normalize_generic_to_pretty(cmpf_true).contains("arith.cmpf true"),
            "cmpf pred 15 → true"
        );
    }

    #[test]
    fn test_normalize_generic_arith_binary_table() {
        // Each arith binary op in the table normalizes to its pretty infix form.
        let ops = [
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.remsi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrsi",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
        ];
        for op in ops {
            let line = format!("%0 = \"{op}\"(%a, %b) <{{}}> : (i32, i32) -> i32");
            let got = normalize_generic_to_pretty(&line);
            assert!(
                got.contains(&format!("{op} %a, %b")),
                "{op} normalization: {got}"
            );
        }
    }

    #[test]
    fn parse_functions_backward_compat_wrapper() {
        // `parse_functions` is the pub backward-compat shim over parse_module;
        // it returns just the function list. Drive a two-arg kernel and assert
        // the name + arg count round-trip.
        let mlir = r#"
module {
  llvm.func @kern(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    llvm.return
  }
}
"#;
        let funcs = parse_functions(mlir).expect("parse_functions should succeed");
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "kern");
        assert_eq!(funcs[0].args.len(), 2);
    }
}
