use anyhow::{Context, Result};
use clap::Parser;
use edgefirst_tflite::{Interpreter, Library, Model};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "tflite-smoke", about = "Smoke test running a .tflite model via TFLite C API")]
struct Args {
    /// Path to a .tflite file (e.g. models/_litertlm_dump/embedded_01.tflite)
    #[arg(long)]
    model: PathBuf,

    /// Number of threads for the interpreter
    #[arg(long, default_value_t = 4)]
    threads: i32,

    /// Invoke the model after allocating tensors (may fail without proper inputs/state).
    #[arg(long, default_value_t = false)]
    invoke: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Uses edgefirst-tflite library discovery; with edgefirst-tflite-sys vendored feature
    // this should work on macOS without a system-installed libtensorflowlite_c.
    let lib = Library::new().context("load TFLite C library")?;
    let model =
        Model::from_file(&lib, &args.model).with_context(|| format!("load {:?}", args.model))?;

    let mut interpreter = Interpreter::builder(&lib)?
        .num_threads(args.threads)
        .build(&model)
        .context("build interpreter")?;

    // Allocate tensors so that we can inspect shapes.
    interpreter
        .allocate_tensors()
        .context("allocate tensors")?;

    let inputs = interpreter.inputs().context("read inputs")?;
    let outputs = interpreter.outputs().context("read outputs")?;

    println!("Model: {:?}", args.model);
    println!("Inputs: {}", inputs.len());
    for (i, t) in inputs.iter().enumerate() {
        println!("  in[{i}]: {t}");
    }
    println!("Outputs: {}", outputs.len());
    for (i, t) in outputs.iter().enumerate() {
        println!("  out[{i}]: {t}");
    }

    if args.invoke {
        // We intentionally do not attempt to set inputs here; LiteRT-LM graphs often require
        // specific tensor values and state wiring. This is just to prove we can execute.
        interpreter.invoke().context("invoke")?;
        println!("invoke(): OK");
    }
    Ok(())
}

