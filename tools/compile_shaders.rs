use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn run(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !status.success() {
        return Err(format!("{program} exited with {status}").into());
    }
    Ok(())
}

fn parse_version(text: &str) -> Option<Vec<u32>> {
    for candidate in text.split(|character: char| !character.is_ascii_digit() && character != '.') {
        let candidate = candidate.trim_matches('.');
        if candidate.matches('.').count() >= 1 {
            if let Some(version) = candidate
                .split('.')
                .map(|component| component.parse().ok())
                .collect()
            {
                return Some(version);
            }
        }
    }
    None
}

fn require_tool_version(
    program: &str,
    argument: &str,
    name: &str,
    minimum: &[u32],
) -> Result<(), Box<dyn Error>> {
    let output = Command::new(program)
        .arg(argument)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    let version_text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let version = parse_version(&version_text).ok_or_else(|| {
        format!(
            "failed to read {name} version from '{}'",
            version_text.trim()
        )
    })?;
    if !output.status.success() || version.as_slice() < minimum {
        return Err(format!(
            "NoGraphicsAPI examples require {name} {} or newer; found '{}'",
            minimum
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("."),
            version_text.trim()
        )
        .into());
    }
    Ok(())
}

fn find_shaders(
    directory: &Path,
    compiled_directory: &Path,
    shaders: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path == compiled_directory {
            continue;
        }
        if path.is_dir() {
            find_shaders(&path, compiled_directory, shaders)?;
        } else if path.extension() == Some(OsStr::new("slang")) {
            shaders.push(path);
        }
    }
    Ok(())
}

fn compile_shader(source: &Path, output: &Path, manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output.parent().unwrap())?;
    println!("Compiling {}", source.display());

    let mut slang = Command::new("slangc");
    slang
        .arg(source)
        .args([
            "-target",
            "spirv",
            "-profile",
            "spirv_1_5",
            "-emit-spirv-directly",
            "-whole-program",
            "-fvk-use-entrypoint-name",
            "-fvk-use-c-layout",
            "-matrix-layout-row-major",
            "-capability",
            "spvDescriptorHeapEXT",
            "-capability",
            "spvMeshShadingEXT",
        ])
        .arg("-I")
        .arg(source.parent().unwrap())
        .arg("-I")
        .arg(manifest_dir.join("assets/shaders"))
        .arg("-I")
        .arg(manifest_dir.join("NoGraphicsAPI/include"))
        .arg("-I")
        .arg(manifest_dir.join("NoGraphicsAPI/utility/include"))
        .arg("-o")
        .arg(output);
    run(&mut slang)?;

    run(Command::new("spirv-val")
        .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
        .arg(output))
}

fn main() -> Result<(), Box<dyn Error>> {
    require_tool_version("slangc", "-version", "Slang", &[2026, 14, 1])?;
    require_tool_version("spirv-val", "--version", "SPIRV-Tools", &[2026, 3])?;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_directory = manifest_dir.join("assets/shaders");
    let compiled_directory = shader_directory.join("compiled");
    let mut shaders = Vec::new();
    find_shaders(&shader_directory, &compiled_directory, &mut shaders)?;
    shaders.sort();

    for source in shaders {
        let relative = source.strip_prefix(&shader_directory)?;
        let output = compiled_directory.join(relative).with_extension("spv");
        compile_shader(&source, &output, &manifest_dir)?;
    }
    Ok(())
}
