# YOLO 目标检测 Web 工具

一个基于 Flask + HTML 前端的 YOLO 目标检测推理工具，支持 YOLOv5、YOLOv8 和 YOLOv11 模型。

## 功能特点

- 🚀 **实时目标检测** - 基于 Ultralytics 框架的快速推理
- 🎨 **美观界面** - 现代化的响应式设计
- 📱 **移动端友好** - 完美适配各种屏幕尺寸
- 🔧 **多模型支持** - 支持 YOLOv5/v8/v11 系列模型
- ⚙️ **可配置参数** - 可调节置信度阈值
- 📊 **详细结果** - 显示检测统计和详细信息
- 🖱️ **拖拽上传** - 支持拖拽文件上传
- 🔄 **实时预览** - 上传前预览图片

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python run.py
```

或者直接运行：

```bash
python app.py
```

### 3. 访问应用

打开浏览器访问：`http://127.0.0.1:5000`

## 使用说明

### 基本使用

1. **上传图片**：点击"选择文件"或拖拽图片到上传区域
2. **选择模型**：从下拉菜单中选择 YOLO 模型
3. **调整置信度**：使用滑块调整检测置信度阈值
4. **开始检测**：点击"检测目标"按钮
5. **查看结果**：在结果页面查看原始图片和检测结果

### 支持的模型

- **YOLOv11 系列**：nano、small、medium、large、extra-large
- **YOLOv8 系列**：nano、small、medium、large、extra-large
- **YOLOv5 系列**：nano、small、medium、large、extra-large
- **自定义模型**：用户训练的YOLO模型（.pt格式）

### 置信度阈值

- **低阈值 (0.1-0.3)**：检测更多目标，可能包含误检
- **中阈值 (0.3-0.6)**：平衡检测数量和准确性
- **高阈值 (0.6-0.9)**：只检测高置信度目标，减少误检

### 使用自定义模型

1. **放置模型文件**：将您的YOLO模型文件（.pt格式）复制到 `models/` 目录
2. **重启应用**：重启Flask应用以加载新模型
3. **选择模型**：在Web界面中选择您的自定义模型
4. **开始检测**：上传图片进行目标检测

**注意事项**：
- 确保模型文件为有效的PyTorch格式
- 模型应与Ultralytics YOLO框架兼容
- 首次使用自定义模型时可能需要额外加载时间

## API 接口

### 检测接口

```bash
POST /api/detect
```

**参数：**
- `file`：图片文件
- `model`：模型名称（可选，默认：yolo11n.pt）
- `confidence`：置信度阈值（可选，默认：0.25）

**响应：**
```json
{
  "success": true,
  "result": {
    "summary": {
      "total_detections": 3,
      "detection_summary": {
        "person": 2,
        "car": 1
      },
      "model_used": "yolo11n.pt",
      "confidence_threshold": 0.25
    },
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [100, 200, 150, 300],
        "bbox_normalized": [0.1, 0.2, 0.15, 0.3],
        "area": 5000
      }
    ],
    "output_image_path": "static/outputs/unique_filename.jpg"
  },
  "original_image": "static/uploads/filename.jpg",
  "output_image": "static/outputs/unique_filename.jpg"
}
```

### 获取可用模型

```bash
GET /api/models
```

**响应：**
```json
{
  "models": [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt"
  ],
  "default_model": "yolo11n.pt"
}
```

## 项目结构

```
yolov5-web-demo/
├── app.py                 # Flask 主应用
├── model_inference.py     # YOLO 推理模块
├── detect.py              # 检测脚本
├── run.py                 # 启动脚本
├── requirements.txt       # Python 依赖
├── README_CN.md          # 中文说明文档
├── models/               # 自定义模型目录
│   └── README.md         # 自定义模型说明
├── static/               # 静态文件
│   ├── uploads/          # 上传文件目录
│   └── outputs/          # 输出文件目录
└── templates/            # HTML 模板
    ├── base.html         # 基础模板
    ├── index.html        # 首页
    ├── inference.html    # 结果页面
    └── about.html        # 关于页面
```

## 技术栈

- **后端**：Flask、Ultralytics YOLO、OpenCV、PyTorch
- **前端**：HTML5、CSS3、JavaScript、Bootstrap 5、Font Awesome
- **AI框架**：Ultralytics YOLO (支持 v5/v8/v11)

## 注意事项

1. **首次运行**：首次使用时会自动下载选定的 YOLO 模型
2. **模型大小**：大型模型需要更多内存和下载时间
3. **GPU 支持**：如果系统有 CUDA 支持的 GPU，会自动使用 GPU 加速
4. **文件格式**：支持 JPG、JPEG、PNG、WEBP 格式
5. **文件大小**：建议图片大小不超过 10MB

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 尝试手动下载模型文件

2. **内存不足**
   - 使用较小的模型（如 nano 或 small）
   - 减小图片尺寸

3. **检测速度慢**
   - 确保使用 GPU（如果可用）
   - 使用较小的模型

### 性能优化

- 使用 YOLOv11n 或 YOLOv8n 获得最快速度
- 使用 YOLOv11x 或 YOLOv8x 获得最佳精度
- 调整置信度阈值平衡速度和准确性
