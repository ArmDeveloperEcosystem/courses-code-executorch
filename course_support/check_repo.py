#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
COURSE_MODULES = {
    0: ROOT / "module-0-course-executorch-intro",
    1: ROOT / "module-1-transformer-inference-with-xnnpack",
    2: ROOT / "module-2-cnn-inference-with-quantization",
    3: ROOT / "module-3-executorch-on-android-semantic-segmentation",
    4: ROOT / "module-4-ethos-npu-accelerated-inference",
}
COURSE_MODULE_DIRECTORY_NAMES = {path.name for path in COURSE_MODULES.values()}
COURSE_PATH_PREFIXES = tuple(
    f"{directory}/" for directory in sorted(COURSE_MODULE_DIRECTORY_NAMES)
) + (".agents/",)
MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / ".agents/course-context.md",
    ROOT / ".agents/skills/explain-mode/SKILL.md",
    ROOT / ".agents/skills/build-mode/SKILL.md",
    *(directory / f"module_{module}.md" for module, directory in COURSE_MODULES.items()),
    COURSE_MODULES[2] / "module_2_ext/module_2_ext.md",
]
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CONTEXT_PATH_PATTERN = re.compile(r"^- `([^`]+)`$")
GENERATED_SUFFIXES = {
    ".bin",
    ".etdp",
    ".etdump",
    ".etrecord",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pt2",
    ".pte",
    ".pth",
    ".tflite",
    ".tosa",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_python(errors: list[str]) -> int:
    files = sorted((ROOT / "course_support").glob("*.py"))
    script_directories = [
        *(COURSE_MODULES[module] / "scripts" for module in range(1, 5)),
        COURSE_MODULES[2] / "module_2_ext/scripts",
    ]
    for scripts in script_directories:
        if scripts.exists():
            files.extend(
                path
                for path in sorted(scripts.rglob("*.py"))
                if "__pycache__" not in path.parts
            )

    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Python parse failed: {relative(path)}: {exc}")
    return len(files)


def markdown_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def check_markdown(errors: list[str]) -> int:
    checked = 0
    for path in MARKDOWN_FILES:
        if not path.exists():
            errors.append(f"Missing Markdown file: {relative(path)}")
            continue

        checked += 1
        text = path.read_text(encoding="utf-8")
        fence_count = sum(line.startswith("```") for line in text.splitlines())
        if fence_count % 2:
            errors.append(f"Unbalanced code fences: {relative(path)}")

        notice_open = text.count("{{% notice")
        notice_close = text.count("{{% /notice %}}")
        if notice_open != notice_close:
            errors.append(
                f"Unbalanced notice blocks: {relative(path)} "
                f"({notice_open} opening, {notice_close} closing)"
            )

        for raw_target in LINK_PATTERN.findall(text):
            target = markdown_link_target(raw_target)
            if not target or target.startswith(
                ("#", "data:", "http://", "https://", "mailto:")
            ):
                continue
            local_path = unquote(target.split("#", 1)[0])
            if not local_path:
                continue
            resolved = (path.parent / local_path).resolve()
            if not resolved.exists():
                errors.append(
                    f"Broken local link: {relative(path)} -> {target}"
                )
    return checked


def parse_skill_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}

    values: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def check_agent_guidance(errors: list[str]) -> int:
    skills = sorted((ROOT / ".agents/skills").glob("*/SKILL.md"))
    for path in skills:
        frontmatter = parse_skill_frontmatter(path)
        expected_name = path.parent.name
        if frontmatter.get("name") != expected_name:
            errors.append(
                f"Invalid skill name in {relative(path)}: "
                f"expected {expected_name!r}"
            )
        if not frontmatter.get("description"):
            errors.append(f"Missing skill description: {relative(path)}")
        if "TODO" in path.read_text(encoding="utf-8"):
            errors.append(f"Skill contains a TODO placeholder: {relative(path)}")

    context = ROOT / ".agents/course-context.md"
    for line in context.read_text(encoding="utf-8").splitlines():
        match = CONTEXT_PATH_PATTERN.match(line)
        if not match:
            continue
        context_path = match.group(1)
        if not context_path.startswith(COURSE_PATH_PREFIXES):
            continue
        mapped = ROOT / context_path
        if not mapped.exists():
            errors.append(f"Missing course-context path: {context_path}")

    guidance_files = [ROOT / "AGENTS.md", *sorted((ROOT / ".agents").rglob("*.md"))]
    for path in guidance_files:
        if re.search(r"module[_ ]?5|module five", path.read_text(encoding="utf-8"), re.I):
            errors.append(f"Module 5 reference in agent guidance: {relative(path)}")
    return len(skills)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [entry.decode() for entry in result.stdout.split(b"\0") if entry]


def generated_reason(path: str) -> str | None:
    pure = PurePosixPath(path)
    parts = pure.parts
    if pure.name == ".DS_Store":
        return "OS metadata"
    if any(part in {".gradle", ".idea", "__pycache__", "build"} for part in parts):
        return "generated build or cache directory"
    if pure.suffix.lower() in GENERATED_SUFFIXES:
        return f"generated artifact ({pure.suffix})"
    if len(parts) >= 2 and parts[0] in COURSE_MODULE_DIRECTORY_NAMES:
        if "artifacts" in parts[1:]:
            return "generated module artifact"
        if "results" in parts[1:] and pure.name != ".gitkeep":
            return "generated module result"
    if "src" in parts and "main" in parts and "assets" in parts:
        if pure.name in {"model_spec.json", "voc_labels.json"}:
            return "generated Android model asset"
    return None


def check_tracked_files(errors: list[str]) -> int:
    files = tracked_files()
    for path in files:
        reason = generated_reason(path)
        if reason:
            errors.append(f"Tracked {reason}: {path}")
    return len(files)


def main() -> int:
    errors: list[str] = []
    python_count = check_python(errors)
    markdown_count = check_markdown(errors)
    skill_count = check_agent_guidance(errors)
    tracked_count = check_tracked_files(errors)

    if errors:
        print("Repository integrity checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Repository integrity checks passed.")
    print(f"- Parsed {python_count} Python files")
    print(f"- Checked {markdown_count} Markdown files")
    print(f"- Validated {skill_count} Agent Skills")
    print(f"- Inspected {tracked_count} tracked paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
