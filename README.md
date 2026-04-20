# Image Object Generation Module

这个模块接收一张输入图片，调用火山方舟视觉模型识别场景和物体，再为每个物体生成单独的白底图片，并输出固定 JSON，方便后端继续处理。

## 功能

- 输入一张图片
- 识别 `scene_type`
- 识别可单独生成的主要物体
- 为每个物体生成单独白底 PNG
- 输出固定结构 JSON

默认会忽略桌面、桌垫、墙面、地板这类背景或承载物。

## 目录结构

运行后默认会生成：

```text
tasks/
  task_001/
    result.json
    crops/
      obj_001.png
      obj_002.png
      ...
```
test_module.py为调试使用的脚本，无需调用，可单独测试使用。
## 依赖

- Python 3
- Pillow
- 可访问火山方舟 Ark 的网络环境
- 有效的 `ARK_API_KEY`

## API Key 放哪里

最推荐放在环境变量里：

```bash
export ARK_API_KEY="your_api_key"
```

更适合生产环境的做法：

- Docker / Compose 环境变量
- Kubernetes Secret
- 云平台 Secret Manager
- 本地 `.env` 文件加载到环境变量，但不要提交到 Git

不建议把 API Key 直接写进代码或提交到仓库。

## 命令行用法

最简单：

```bash
python3 recognize_image.py /path/to/input.png
```

指定任务 ID：

```bash
python3 recognize_image.py --task-id task_001 /path/to/input.png
```

指定返回给后端的 URL 前缀：

```bash
python3 recognize_image.py \
  --task-id task_001 \
  --base-url https://example.com/tasks \
  /path/to/input.png
```

CLI 会把结果 JSON 直接打印到 `stdout`。

## Python 模块用法

### 正式调用

后端最推荐调用 `process_image()`：

```python
from recognize_image import process_image

result = process_image(
    image="/path/to/input.png",
    task_id="task_001",
    task_root="./tasks",
    base_url="https://example.com/tasks",
)
```

返回值示例：

```json
{
  "task_id": "task_001",
  "scene_type": "desk_setup",
  "objects": [
    {
      "object_id": "obj_001",
      "label": "laptop",
      "crop_url": "https://example.com/tasks/task_001/crops/obj_001.png"
    }
  ]
}
```

### 冒烟测试

联调时可以调用 `smoke_test()`：

```python
from recognize_image import smoke_test

result = smoke_test(
    image="/path/to/input.png",
    task_id="task_smoke",
)
```

它会额外检查：

- `result.json` 是否生成
- `scene_type` 是否有效
- `objects` 是否非空
- 对应生成图片文件是否真的存在

## 主要公开函数

- `process_image(...)`
  - 正式处理入口
  - 输入一张图，返回结果字典
  - 内部会先定位物体，再调用 AI 生成白底单物体图

- `smoke_test(...)`
  - 用于联调和打包前快速验证

- `main()`
  - 命令行入口

## 输出 JSON 结构

脚本输出结构固定为：

```json
{
  "task_id": "task_001",
  "scene_type": "desk_setup",
  "objects": [
    {
      "object_id": "obj_001",
      "label": "laptop",
      "crop_url": "https://example.com/tasks/task_001/crops/obj_001.png"
    }
  ]
}
```

字段说明：

- `task_id`: 当前任务 ID
- `scene_type`: 场景类型，英文 snake_case
- `objects`: 识别出的物体列表
- `object_id`: 物体唯一 ID
- `label`: 物体标签，英文 snake_case
- `crop_url`: 生成后的单物体图片地址；不传 `base_url` 时默认返回本地 `file://` URI

## 



