#!/usr/bin/env python3
"""Detect objects in one image, generate isolated white-background images, and emit module-friendly JSON."""

from __future__ import annotations

import argparse
import ast
import base64
from io import BytesIO
import json
import mimetypes
import os
import re
import ssl
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image


API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
IMAGE_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DEFAULT_IMAGE_DIRS = (Path("input"), Path("imput_picture"))
DEFAULT_VISION_MODEL = "doubao-1-5-vision-pro-32k-250115"
DEFAULT_IMAGE_MODEL = "doubao-seedream-4-5-251128"
DEFAULT_GENERATION_SIZE = "1920x1920"
DEFAULT_TASK_ROOT = Path("tasks")
DEFAULT_MARGIN_RATIO = 0.03
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
EXCLUDED_LABELS = {"desk", "table", "desk_mat", "mouse_pad", "floor", "wall", "background"}
SSL_CONTEXT = ssl._create_unverified_context()

DEFAULT_DETECTION_PROMPT = """请分析这张图片，并只返回适合单独生成白底单物体图后传给后端的主要物体。

要求：
1. 只输出 JSON，不要输出 Markdown，不要补充解释。
2. JSON 格式固定为 {"scene_type":"desk_setup","objects":[{"label":"laptop","count":1,"evidence":"一句简短依据"}]}。
3. scene_type 使用简短英文 snake_case，例如 desk_setup、kitchen_counter、living_room、shelf_display、tabletop_scene。
4. label 使用英文 snake_case 单数名词，例如 laptop、cup、keyboard、watch。
5. 只返回具体、可单独生成单物体图的物体；忽略桌面、桌垫、墙面、地板等背景或承载物。
6. 不确定就不要猜测。
"""

BOX_PROMPT_TEMPLATE = """请根据这张图片，为这些物体标出边界框：{labels}

要求：
1. 只输出 JSON，不要输出 Markdown，不要补充解释。
2. JSON 格式固定为 {{"objects":[{{"label":"laptop","bbox":[x1,y1,x2,y2],"evidence":"一句简短依据"}}]}}。
3. bbox 使用 0 到 1000 的整数坐标，基于整张图，原点在左上角。
4. label 必须从给定列表中选择。
5. 如果同类物体有多个实例，请分别输出多条同 label 记录。
6. 只返回位置清楚、适合作为白底单物体图生成参考的物体。
"""

GENERATION_PROMPT_TEMPLATE = """Use the reference image to generate one isolated product-style image of the same {object_name}.

Requirements:
1. Keep the same object category, main shape, material, and color from the reference.
2. Output only one object on a pure white background.
3. Center the object and show the full object clearly.
4. Do not include any table, wall, hand, shadow clutter, text, logo, or watermark.
5. If the object is partially occluded in the reference, complete it into a plausible full shape.
Reference hint: {evidence}
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Detect objects in one image, generate isolated white-background images, and print a JSON manifest."
    )
    parser.add_argument(
        "--api-key",
        help="Volcengine Ark API Key. If omitted, the script will read ARK_API_KEY or ask you to input it.",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID for the output JSON. If omitted, one is derived from the image name.",
    )
    parser.add_argument(
        "--task-root",
        default=str(DEFAULT_TASK_ROOT),
        help=f"Root directory for task outputs (default: {DEFAULT_TASK_ROOT}).",
    )
    parser.add_argument(
        "--base-url",
        help="Optional URL prefix for crop_url, for example https://example.com/tasks",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VISION_MODEL,
        help=f"Vision model name for scene/object detection (default: {DEFAULT_VISION_MODEL}).",
    )
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Image generation model name (default: {DEFAULT_IMAGE_MODEL}).",
    )
    parser.add_argument(
        "--generation-size",
        default=DEFAULT_GENERATION_SIZE,
        help=f"Generated image size (default: {DEFAULT_GENERATION_SIZE}).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_DETECTION_PROMPT,
        help="Prompt sent to the vision model for scene/object detection.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=DEFAULT_MARGIN_RATIO,
        help=f"Extra margin around each crop (default: {DEFAULT_MARGIN_RATIO}).",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Keep at most N localized objects. 0 means all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs to stderr.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw detection model output and exit.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the image file. If omitted, the first image under input/ or imput_picture/ is used.",
    )
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    """Print a progress message to stderr when verbose mode is enabled."""
    if verbose:
        print(message, file=sys.stderr)


def prompt_text(prompt: str) -> str:
    """Read one line of interactive input from the terminal."""
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise RuntimeError("当前运行环境无法进行交互输入。") from exc


def find_default_image() -> Path:
    """Find the first supported image under the default input folders."""
    checked_dirs: list[Path] = []
    for image_dir in DEFAULT_IMAGE_DIRS:
        checked_dirs.append(image_dir)
        if not image_dir.exists():
            continue

        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if images:
            return images[0]

    joined = "、".join(str(path.resolve()) for path in checked_dirs)
    raise FileNotFoundError(f"默认目录里没有找到图片文件：{joined}")


def resolve_image_path(image_arg: str | None) -> Path:
    """Resolve and validate the input image path."""
    if image_arg:
        path = Path(image_arg).expanduser()
    else:
        try:
            path = find_default_image()
        except FileNotFoundError:
            entered = prompt_text("请输入图片路径：")
            if not entered:
                raise FileNotFoundError("没有输入图片路径。")
            path = Path(entered).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"图片路径不是文件：{path}")
    return path


def get_api_key(cli_api_key: str | None) -> str:
    """Resolve the Ark API key from argument, environment, or interactive input."""
    if cli_api_key:
        return cli_api_key.strip()

    env_api_key = os.environ.get("ARK_API_KEY", "").strip()
    if env_api_key:
        return env_api_key

    api_key = prompt_text("请输入 ARK_API_KEY：")
    if not api_key:
        raise RuntimeError("没有输入 ARK_API_KEY。")
    return api_key


def build_task_id(task_id: str | None, image_path: Path) -> str:
    """Build a stable task id from user input or the image filename."""
    if task_id:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", task_id).strip("_")
        if cleaned:
            return cleaned

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", image_path.stem).strip("_") or "image"
    return f"task_{stem}"


def normalize_identifier(text: str, fallback: str = "") -> str:
    """Normalize free-form text into a snake_case-style identifier."""
    normalized = re.sub(r"\s+", "_", text.strip().lower())
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or fallback


def image_to_data_url(image_path: Path) -> str:
    """Encode a local image file as a data URL for multimodal model input."""
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(model: str, prompt: str, image_data_url: str) -> dict[str, Any]:
    """Build the OpenAI-compatible multimodal chat payload for Ark."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }


def call_ark(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one HTTPS request to the Ark chat-completions endpoint."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120, context=SSL_CONTEXT) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"火山方舟接口返回 HTTP {exc.code}：{details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"调用火山方舟接口失败：{exc.reason}") from exc


def extract_text(response_json: dict[str, Any]) -> str:
    """Extract the first text response from an Ark chat-completions result."""
    choices = response_json.get("choices")
    if not choices:
        raise RuntimeError(f"接口返回里没有 choices：{response_json}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

    raise RuntimeError(f"无法解析模型输出：{response_json}")


def extract_json_text(text: str) -> str:
    """Extract the JSON-looking segment from model output text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start_candidates = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not start_candidates:
        raise ValueError("模型输出中没有找到 JSON。")

    start = min(start_candidates)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()

    raise ValueError("找到了 JSON 起始位置，但没有找到完整的结束位置。")


def load_json_like(text: str) -> Any:
    """Parse JSON or near-JSON text into Python data."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        normalized = re.sub(r",(\s*[}\]])", r"\1", text.strip())
        normalized = re.sub(r"\bnull\b", "None", normalized)
        normalized = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        return ast.literal_eval(normalized)


def parse_scene_result(model_text: str) -> dict[str, Any]:
    """Parse scene type and candidate object labels from model output."""
    data = load_json_like(extract_json_text(model_text))
    if not isinstance(data, dict):
        raise ValueError(f"场景识别 JSON 结构不符合预期：{data}")

    scene_type = normalize_identifier(str(data.get("scene_type", "")), "unknown_scene")
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError(f"场景识别结果里没有 objects：{data}")

    objects: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            continue

        label_source = item.get("label") or item.get("name") or ""
        label = normalize_identifier(str(label_source))
        if not label or label in seen or label in EXCLUDED_LABELS:
            continue

        seen.add(label)
        objects.append(
            {
                "label": label,
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )

    if not objects:
        raise ValueError("没有识别到可裁剪的具体物体。")

    return {"scene_type": scene_type, "objects": objects}


def build_box_prompt(objects: list[dict[str, str]]) -> str:
    """Build the localization prompt from the detected object labels."""
    labels = "、".join(item["label"] for item in objects)
    return BOX_PROMPT_TEMPLATE.format(labels=labels or "object")


def parse_bbox(value: Any) -> list[int] | None:
    """Parse one bounding box value into four integer coordinates."""
    if isinstance(value, list) and len(value) == 4:
        try:
            return [int(round(float(item))) for item in value]
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 4:
            return [int(round(float(item))) for item in numbers[:4]]
    return None


def parse_localized_objects(model_text: str, allowed_labels: set[str]) -> list[dict[str, Any]]:
    """Parse and filter localized object boxes returned by the model."""
    data = load_json_like(extract_json_text(model_text))
    if isinstance(data, dict):
        raw_objects = data.get("objects")
    elif isinstance(data, list):
        raw_objects = data
    else:
        raw_objects = None

    if not isinstance(raw_objects, list):
        raise ValueError(f"定位结果 JSON 结构不符合预期：{data}")

    localized: list[dict[str, Any]] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue

        label_source = item.get("label") or item.get("name") or ""
        label = normalize_identifier(str(label_source))
        if not label or label in EXCLUDED_LABELS or label not in allowed_labels:
            continue

        bbox = (
            parse_bbox(item.get("bbox"))
            or parse_bbox(item.get("box"))
            or parse_bbox(item.get("bbox_2d"))
        )
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        x1 = min(max(x1, 0), 1000)
        y1 = min(max(y1, 0), 1000)
        x2 = min(max(x2, 0), 1000)
        y2 = min(max(y2, 0), 1000)
        if x2 <= x1 or y2 <= y1:
            continue

        localized.append(
            {
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )

    if not localized:
        raise ValueError("模型没有返回可用的边界框。")

    label_order = {label: index for index, label in enumerate(sorted(allowed_labels))}
    localized.sort(key=lambda item: (label_order.get(item["label"], 9999), item["bbox"][1], item["bbox"][0]))
    return localized


def scale_bbox(
    bbox: list[int], image_width: int, image_height: int, margin_ratio: float
) -> list[int]:
    """Convert a normalized bbox into pixel coordinates with optional margin."""
    x1 = int(round(bbox[0] / 1000 * image_width))
    y1 = int(round(bbox[1] / 1000 * image_height))
    x2 = int(round(bbox[2] / 1000 * image_width))
    y2 = int(round(bbox[3] / 1000 * image_height))

    margin_x = max(4, int(round((x2 - x1) * max(margin_ratio, 0.0))))
    margin_y = max(4, int(round((y2 - y1) * max(margin_ratio, 0.0))))

    left = max(0, x1 - margin_x)
    top = max(0, y1 - margin_y)
    right = min(image_width, x2 + margin_x)
    bottom = min(image_height, y2 + margin_y)

    if right <= left or bottom <= top:
        raise ValueError(f"无效像素框：{bbox}")
    return [left, top, right, bottom]


def pil_image_to_data_url(image: Image.Image) -> str:
    """Encode an in-memory PIL image as a PNG data URL."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_image_bytes_as_png(image_bytes: bytes, output_path: Path) -> None:
    """Normalize generated image bytes into a PNG file on disk."""
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(output_path, format="PNG")


def build_generation_prompt(label: str, evidence: str) -> str:
    """Build the prompt used to generate one isolated object image."""
    object_name = label.replace("_", " ").strip() or "object"
    details = evidence.strip() or "Match the object appearance from the reference."
    return GENERATION_PROMPT_TEMPLATE.format(object_name=object_name, evidence=details)


def call_ark_image_generation(
    api_key: str,
    *,
    model: str,
    prompt: str,
    image_data_url: str,
    size: str,
) -> str:
    """Call the Ark image-generation endpoint and return the first base64 image."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "image": image_data_url,
            "response_format": "b64_json",
            "size": size,
            "watermark": False,
        }
    ).encode("utf-8")
    req = request.Request(
        IMAGE_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=300, context=SSL_CONTEXT) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"火山方舟生图接口返回 HTTP {exc.code}：{details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"调用火山方舟生图接口失败：{exc.reason}") from exc

    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"生图接口没有返回 data：{response_json}")

    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"生图接口返回格式异常：{response_json}")

    b64_json = first.get("b64_json")
    if not isinstance(b64_json, str) or not b64_json:
        raise RuntimeError(f"生图接口没有返回 b64_json：{response_json}")
    return b64_json


def build_crop_url(
    output_path: Path,
    task_id: str,
    object_id: str,
    base_url: str | None,
) -> str:
    """Build the generated-image URL or file URI stored in the output manifest."""
    relative = Path(task_id) / "crops" / f"{object_id}.png"
    if base_url:
        return f"{base_url.rstrip('/')}/{relative.as_posix()}"
    return output_path.resolve().as_uri()


def save_task_result(
    api_key: str,
    image_path: Path,
    scene_type: str,
    localized_objects: list[dict[str, Any]],
    task_root: Path,
    task_id: str,
    base_url: str | None,
    margin_ratio: float,
    image_model: str,
    generation_size: str,
    verbose: bool,
) -> tuple[Path, dict[str, Any]]:
    """Generate isolated object images, save them to disk, and write the final manifest."""
    task_dir = task_root / task_id
    crops_dir = task_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest_objects: list[dict[str, str]] = []
    with Image.open(image_path) as image:
        width, height = image.size
        for index, item in enumerate(localized_objects, start=1):
            pixel_bbox = scale_bbox(item["bbox"], width, height, margin_ratio)
            reference_image = image.crop(tuple(pixel_bbox)).convert("RGB")
            reference_data_url = pil_image_to_data_url(reference_image)

            log(f"正在生成：{item['label']}", verbose)
            b64_json = call_ark_image_generation(
                api_key,
                model=image_model,
                prompt=build_generation_prompt(item["label"], item.get("evidence", "")),
                image_data_url=reference_data_url,
                size=generation_size,
            )

            object_id = f"obj_{index:03d}"
            output_path = crops_dir / f"{object_id}.png"
            save_image_bytes_as_png(base64.b64decode(b64_json), output_path)

            manifest_objects.append(
                {
                    "object_id": object_id,
                    "label": item["label"],
                    "crop_url": build_crop_url(output_path, task_id, object_id, base_url),
                }
            )

    result = {
        "task_id": task_id,
        "scene_type": scene_type,
        "objects": manifest_objects,
    }

    result_path = task_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return task_dir, result


def process_image(
    image: str | Path,
    api_key: str | None = None,
    *,
    task_id: str | None = None,
    task_root: str | Path = DEFAULT_TASK_ROOT,
    base_url: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    generation_size: str = DEFAULT_GENERATION_SIZE,
    prompt: str = DEFAULT_DETECTION_PROMPT,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    max_objects: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full module pipeline and return the final task manifest."""
    image_path = resolve_image_path(str(image))
    resolved_api_key = get_api_key(api_key)
    resolved_task_id = build_task_id(task_id, image_path)

    log(f"开始分析图片：{image_path}", verbose)
    image_data_url = image_to_data_url(image_path)

    detection_payload = build_payload(
        model=model,
        prompt=prompt,
        image_data_url=image_data_url,
    )
    detection_text = extract_text(call_ark(api_key=resolved_api_key, payload=detection_payload))
    scene_result = parse_scene_result(detection_text)
    allowed_labels = {item["label"] for item in scene_result["objects"]}

    log(
        f"场景类型：{scene_result['scene_type']}，候选物体数：{len(scene_result['objects'])}",
        verbose,
    )

    box_payload = build_payload(
        model=model,
        prompt=build_box_prompt(scene_result["objects"]),
        image_data_url=image_data_url,
    )
    localized_objects = parse_localized_objects(
        extract_text(call_ark(api_key=resolved_api_key, payload=box_payload)),
        allowed_labels=allowed_labels,
    )

    limited_max_objects = max(max_objects, 0)
    if limited_max_objects:
        localized_objects = localized_objects[:limited_max_objects]

    resolved_task_root = Path(task_root).expanduser()
    task_dir, result = save_task_result(
        api_key=resolved_api_key,
        image_path=image_path,
        scene_type=scene_result["scene_type"],
        localized_objects=localized_objects,
        task_root=resolved_task_root,
        task_id=resolved_task_id,
        base_url=base_url,
        margin_ratio=margin_ratio,
        image_model=image_model,
        generation_size=generation_size,
        verbose=verbose,
    )

    log(f"任务目录：{task_dir}", verbose)
    return result


def smoke_test(
    image: str | Path,
    api_key: str | None = None,
    *,
    task_id: str = "task_smoke",
    task_root: str | Path = DEFAULT_TASK_ROOT,
    base_url: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    generation_size: str = DEFAULT_GENERATION_SIZE,
    prompt: str = DEFAULT_DETECTION_PROMPT,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    max_objects: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a lightweight end-to-end verification and assert expected outputs."""
    result = process_image(
        image=image,
        api_key=api_key,
        task_id=task_id,
        task_root=task_root,
        base_url=base_url,
        model=model,
        image_model=image_model,
        generation_size=generation_size,
        prompt=prompt,
        margin_ratio=margin_ratio,
        max_objects=max_objects,
        verbose=verbose,
    )

    task_dir = Path(task_root).expanduser() / result["task_id"]
    result_path = task_dir / "result.json"
    crops_dir = task_dir / "crops"

    if not result_path.exists():
        raise AssertionError(f"缺少结果文件：{result_path}")
    if not isinstance(result.get("scene_type"), str) or not result["scene_type"]:
        raise AssertionError("scene_type 为空。")

    objects = result.get("objects")
    if not isinstance(objects, list) or not objects:
        raise AssertionError("objects 为空。")

    for item in objects:
        if not isinstance(item.get("object_id"), str) or not item["object_id"]:
            raise AssertionError(f"object_id 非法：{item}")
        if not isinstance(item.get("label"), str) or not item["label"]:
            raise AssertionError(f"label 非法：{item}")

        crop_path = crops_dir / f"{item['object_id']}.png"
        if not crop_path.exists():
            raise AssertionError(f"缺少裁图文件：{crop_path}")

    return result


def main() -> int:
    """CLI entry point for processing one image and printing JSON to stdout."""
    args = parse_args()

    try:
        api_key = get_api_key(args.api_key)
        image_path = resolve_image_path(args.image)

        if args.raw:
            image_data_url = image_to_data_url(image_path)
            detection_payload = build_payload(
                model=args.model,
                prompt=args.prompt,
                image_data_url=image_data_url,
            )
            detection_text = extract_text(call_ark(api_key=api_key, payload=detection_payload))
            print(detection_text)
            return 0

        result = process_image(
            image=image_path,
            api_key=api_key,
            task_id=args.task_id,
            task_root=args.task_root,
            base_url=args.base_url,
            model=args.model,
            image_model=args.image_model,
            generation_size=args.generation_size,
            prompt=args.prompt,
            margin_ratio=args.margin_ratio,
            max_objects=args.max_objects,
            verbose=args.verbose,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
