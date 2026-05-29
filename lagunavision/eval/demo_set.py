from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoEvalCase:
    id: str
    title: str
    visible_text: str
    question: str
    must_include: tuple[str, ...]
    accepted_fix_terms: tuple[str, ...]
    must_not_include: tuple[str, ...] = ()


DEMO_CASES = (
    DemoEvalCase(
        id="py_missing_requests_001",
        title="Python traceback",
        visible_text="Traceback\nModuleNotFoundError: No module named 'requests'\n$ python app.py",
        question="What is the bug shown in the screenshot and what is the minimal fix?",
        must_include=("ModuleNotFoundError", "requests"),
        accepted_fix_terms=("pip install requests", "python -m pip install requests", "install requests"),
        must_not_include=("syntax error",),
    ),
    DemoEvalCase(
        id="py_keyerror_config_002",
        title="Python config error",
        visible_text="KeyError: 'DATABASE_URL'\nsettings.py line 18\nos.environ['DATABASE_URL']",
        question="What configuration bug is visible and how should it be fixed?",
        must_include=("KeyError", "DATABASE_URL"),
        accepted_fix_terms=("set DATABASE_URL", "add DATABASE_URL", "environment variable"),
    ),
    DemoEvalCase(
        id="py_port_in_use_003",
        title="Python server error",
        visible_text="OSError: [Errno 48] Address already in use\nuvicorn main:app --port 8000",
        question="Why did the server fail to start and what is the minimal fix?",
        must_include=("Address already in use", "8000"),
        accepted_fix_terms=("free the port", "use another port", "kill the process"),
    ),
    DemoEvalCase(
        id="py_attribute_none_004",
        title="Python attribute error",
        visible_text="AttributeError: 'NoneType' object has no attribute 'strip'\nuser.email.strip()",
        question="What is the likely cause and fix?",
        must_include=("NoneType", "strip"),
        accepted_fix_terms=("check for None", "default email", "validate user.email"),
    ),
    DemoEvalCase(
        id="ts_missing_prop_005",
        title="TypeScript error",
        visible_text="TS2339: Property 'items' does not exist on type 'UserResponse'\nsrc/users.ts:42",
        question="What TypeScript issue is shown and how should it be fixed?",
        must_include=("TS2339", "items", "UserResponse"),
        accepted_fix_terms=("update UserResponse", "use the correct property", "add items"),
    ),
    DemoEvalCase(
        id="ts_undefined_symbol_006",
        title="TypeScript build error",
        visible_text="Cannot find name 'fetchUsers'. Did you mean 'getUsers'?\nsrc/page.tsx:19",
        question="What name error appears and what is the fix?",
        must_include=("Cannot find name", "fetchUsers"),
        accepted_fix_terms=("rename to getUsers", "import fetchUsers", "define fetchUsers"),
    ),
    DemoEvalCase(
        id="npm_missing_script_007",
        title="npm script error",
        visible_text="npm ERR! Missing script: \"dev\"\nTo see scripts, run npm run",
        question="What is wrong with the npm command and how can it be fixed?",
        must_include=("Missing script", "dev"),
        accepted_fix_terms=("add dev script", "run an existing script", "package.json"),
    ),
    DemoEvalCase(
        id="test_assertion_008",
        title="Failing test",
        visible_text="AssertionError: expected status 200 but got 500\ntest_api.py::test_create_user",
        question="What failed in the test and where should you look first?",
        must_include=("expected status 200", "500"),
        accepted_fix_terms=("server error", "create_user", "API handler"),
    ),
    DemoEvalCase(
        id="test_snapshot_009",
        title="Snapshot failure",
        visible_text="Snapshot failed: expected \"Submit\" received \"Save\"\nButton.test.tsx",
        question="What UI test mismatch is visible?",
        must_include=("Submit", "Save"),
        accepted_fix_terms=("update the snapshot", "fix the label", "Button"),
    ),
    DemoEvalCase(
        id="test_import_010",
        title="Import failure",
        visible_text="ImportError: cannot import name 'UserService' from 'services'\ntests/test_users.py",
        question="What import problem is shown and what is the likely fix?",
        must_include=("ImportError", "UserService"),
        accepted_fix_terms=("export UserService", "fix the import", "services"),
    ),
    DemoEvalCase(
        id="git_rejected_011",
        title="Git push error",
        visible_text="! [rejected] main -> main (fetch first)\nerror: failed to push some refs",
        question="Why was the git push rejected?",
        must_include=("rejected", "fetch first"),
        accepted_fix_terms=("pull", "fetch", "rebase"),
    ),
    DemoEvalCase(
        id="shell_permission_012",
        title="Shell permission error",
        visible_text="zsh: permission denied: ./deploy.sh\n-rw-r--r-- deploy.sh",
        question="What shell error is visible and how do you fix it?",
        must_include=("permission denied", "deploy.sh"),
        accepted_fix_terms=("chmod +x", "execute permission"),
    ),
    DemoEvalCase(
        id="docker_daemon_013",
        title="Docker error",
        visible_text="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        question="What dependency/service problem is shown?",
        must_include=("Docker daemon", "docker.sock"),
        accepted_fix_terms=("start Docker", "docker service", "daemon"),
    ),
    DemoEvalCase(
        id="env_missing_key_014",
        title="Env config error",
        visible_text="RuntimeError: OPENAI_API_KEY is not set\n.env.local missing required key",
        question="What config issue is visible and what should be added?",
        must_include=("OPENAI_API_KEY", "not set"),
        accepted_fix_terms=("set OPENAI_API_KEY", ".env", "environment variable"),
    ),
    DemoEvalCase(
        id="dependency_version_015",
        title="Dependency error",
        visible_text="ImportError: pydantic.v1 is not available\ninstalled pydantic==1.10.15",
        question="What dependency mismatch is shown?",
        must_include=("pydantic.v1", "1.10.15"),
        accepted_fix_terms=("upgrade pydantic", "pydantic 2", "fix dependency version"),
    ),
)


def generate_demo_eval(output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(DEMO_CASES, start=1):
            image_name = f"{index:03d}.png"
            _write_screenshot(images_dir / image_name, case)
            handle.write(
                json.dumps(
                    {
                        "id": case.id,
                        "image": f"images/{image_name}",
                        "question": case.question,
                        "ocr_text": case.visible_text,
                        "must_include": case.must_include,
                        "accepted_fix_terms": case.accepted_fix_terms,
                        "must_not_include": case.must_not_include,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest_path


def _write_screenshot(path: Path, case: DemoEvalCase) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install dev/data dependencies with `python -m pip install -e '.[dev]'`.") from exc

    image = Image.new("RGB", (1280, 720), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 1248, 96), fill="#1f2937")
    draw.text((56, 56), case.title, fill="#f9fafb")
    draw.rectangle((32, 120, 760, 688), outline="#374151", width=2)
    draw.text((56, 150), "editor", fill="#9ca3af")
    draw.rectangle((784, 120, 1248, 688), outline="#374151", width=2)
    draw.text((808, 150), "terminal", fill="#9ca3af")
    draw.multiline_text((808, 210), case.visible_text, fill="#f9fafb", spacing=8)
    image.save(path)
