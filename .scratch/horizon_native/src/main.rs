use std::env;
use std::io::{self, BufRead, Write};

use horizon_native::{decide, Action, Cell, Config, Edge, Model, Phase, Status};

fn parse_u64(value: &str, what: &str) -> Result<u64, String> {
    value.parse().map_err(|_| format!("invalid {what}: {value}"))
}

fn parse_i32(value: &str, what: &str) -> Result<i32, String> {
    value.parse().map_err(|_| format!("invalid {what}: {value}"))
}

fn parse_cell_id(value: &str) -> Result<u16, String> {
    value.parse().map_err(|_| format!("invalid cell index: {value}"))
}

fn parse_bool(value: &str, what: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("invalid {what} flag: {value}")),
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--timeout-ms" => {
                i += 1;
                let raw = args.get(i).ok_or("--timeout-ms needs a value")?;
                let ms = parse_u64(raw, "timeout")?;
                config.default_timeout_ms = (ms != 0).then_some(ms);
            }
            "--max-cache" => {
                i += 1;
                let raw = args.get(i).ok_or("--max-cache needs a value")?;
                config.max_cache_entries = raw.parse().map_err(|_| format!("invalid max cache: {raw}"))?;
            }
            "--help" | "-h" => {
                println!("horizon_native [--timeout-ms N] [--max-cache N] < protocol.txt");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(config)
}

struct Builder {
    id: String,
    horizon: u8,
    phase: Phase,
    timeout_ms: Option<u64>,
    cells: Vec<Cell>,
    ta: Vec<Edge>,
    oa: Vec<Edge>,
    near: Vec<Edge>,
    preferred: Option<Action>,
    preferred_required: Option<u16>,
}

impl Builder {
    fn finish(self) -> Model {
        Model { id: self.id, horizon: self.horizon, phase: self.phase, timeout_ms: self.timeout_ms, cells: self.cells, target_anchored: self.ta, opponent_anchored: self.oa, near: self.near, preferred: self.preferred, preferred_required: self.preferred_required }
    }
}

fn main() {
    let config = match parse_args() {
        Ok(config) => config,
        Err(message) => { eprintln!("{message}"); std::process::exit(2); }
    };
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut current: Option<Builder> = None;
    let mut saw_header = false;
    for (line_no, raw) in stdin.lock().lines().enumerate() {
        let raw = match raw {
            Ok(line) => line,
            Err(error) => { eprintln!("input error: {error}"); std::process::exit(2); }
        };
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let result: Result<(), String> = (|| {
            match parts[0] {
                "HORIZON_NATIVE_V1" => {
                    if parts.len() != 1 || saw_header || current.is_some() { return Err("misplaced protocol header".to_string()); }
                    saw_header = true;
                }
                "MODEL" => {
                    if !saw_header { return Err("missing HORIZON_NATIVE_V1 header".to_string()); }
                    if current.is_some() { return Err("nested MODEL".to_string()); }
                    if !(parts.len() == 4 || parts.len() == 5) { return Err("MODEL needs id, horizon, phase, and optional timeout_ms".to_string()); }
                    let horizon: u8 = parts[2].parse().map_err(|_| format!("invalid horizon: {}", parts[2]))?;
                    let phase = match parts[3] { "first" => Phase::First, "second" => Phase::Second, x => return Err(format!("invalid phase: {x}")) };
                    let timeout_ms = if parts.len() == 5 { let ms = parse_u64(parts[4], "model timeout")?; (ms != 0).then_some(ms) } else { None };
                    current = Some(Builder { id: parts[1].to_string(), horizon, phase, timeout_ms, cells: vec![], ta: vec![], oa: vec![], near: vec![], preferred: None, preferred_required: None });
                }
                "CELL" => {
                    if parts.len() != 5 { return Err("CELL needs q r anchored root_legal".to_string()); }
                    let b = current.as_mut().ok_or("CELL outside MODEL")?;
                    b.cells.push(Cell { q: parse_i32(parts[1], "q")?, r: parse_i32(parts[2], "r")?, anchored: parse_bool(parts[3], "anchored")?, root_legal: parse_bool(parts[4], "root_legal")? });
                }
                "TA" | "OA" | "NE" => {
                    let b = current.as_mut().ok_or("edge outside MODEL")?;
                    let ids = parts[1..].iter().map(|x| parse_cell_id(x)).collect::<Result<Vec<_>, _>>()?;
                    let edge = Edge::new(&ids)?;
                    match parts[0] { "TA" => b.ta.push(edge), "OA" => b.oa.push(edge), _ => b.near.push(edge) }
                }
                "PREF" => {
                    if !(parts.len() == 2 || parts.len() == 3) { return Err("PREF needs one or two cell indices".to_string()); }
                    let b = current.as_mut().ok_or("PREF outside MODEL")?;
                    let a = parse_cell_id(parts[1])?;
                    b.preferred = Some(if parts.len() == 2 {
                        Action::one(a)
                    } else {
                        let second = parse_cell_id(parts[2])?;
                        if a == second { return Err("PREF contains a duplicate cell".to_string()); }
                        Action::pair(a, second)
                    });
                }
                "PREF_CELL" => {
                    if parts.len() != 2 { return Err("PREF_CELL needs one cell index".to_string()); }
                    let b = current.as_mut().ok_or("PREF_CELL outside MODEL")?;
                    b.preferred_required = Some(parse_cell_id(parts[1])?);
                }
                "END" => {
                    if parts.len() != 1 { return Err("END takes no fields".to_string()); }
                    let model = current.take().ok_or("END outside MODEL")?.finish();
                    let decision = decide(model, config);
                    writeln!(stdout, "{}", decision.json_line()).map_err(|e| e.to_string())?;
                    stdout.flush().map_err(|e| e.to_string())?;
                }
                other => return Err(format!("unknown record: {other}")),
            }
            Ok(())
        })();
        if let Err(message) = result {
            let phase = current.as_ref().map(|x| x.phase).unwrap_or(Phase::First);
            let id = current.as_ref().map(|x| x.id.clone()).unwrap_or_else(|| format!("line_{}", line_no + 1));
            let error_model = Model { id, horizon: current.as_ref().map(|x| x.horizon).unwrap_or(13), phase, timeout_ms: None, cells: vec![], target_anchored: vec![], opponent_anchored: vec![], near: vec![], preferred: None, preferred_required: None };
            let mut decision = decide(error_model, config);
            decision.status = Status::Error(format!("line {}: {}", line_no + 1, message));
            let _ = writeln!(stdout, "{}", decision.json_line());
            let _ = stdout.flush();
            std::process::exit(2);
        }
    }
    if current.is_some() {
        eprintln!("unterminated MODEL at EOF");
        std::process::exit(2);
    }
}
