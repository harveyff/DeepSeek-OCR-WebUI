# 🔍 DeepSeek-OCR-WebUI
[Visit Application →](https://deepseek-ocr.aws.xin/)

<div align="center">

**🌐 [English](./README.md) | [简体中文](./README_zh-CN.md) | [繁體中文](./README_zh-TW.md) | [日本語](./README_ja.md)**

[![Version](https://img.shields.io/badge/version-v3.2-blue.svg)](./CHANGELOG.md)
[![Docker](https://img.shields.io/badge/docker-supported-brightgreen.svg)](./docker-compose.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Language](https://img.shields.io/badge/languages-4-orange.svg)](#multilingual-support)

Intelligent OCR System · Batch Processing · Multi-Mode Support · Bounding Box Visualization

[Features](#features) • [Quick Start](#quick-start) • [Version History](#version-history) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 📖 Introduction

DeepSeek-OCR-WebUI is an intelligent image recognition web application based on the DeepSeek-OCR model, providing an intuitive user interface and powerful recognition capabilities.

### 🖼️ UI Preview

<div align="center">

![DeepSeek-OCR-WebUI Interface](./assets/ui_screenshot.png)

**Modern user interface with multilingual support, batch processing, and bounding box visualization**

</div>

### 📈 Star History

<div align="center">

![Star History Chart](https://api.star-history.com/svg?repos=neosun100/DeepSeek-OCR-WebUI&type=Date)

**Star growth over time - Help us grow! ⭐**

</div>

### ✨ Core Highlights

- 🎯 **7 Recognition Modes** - Document, OCR, Chart, Find, Freeform, etc.
- 🖼️ **Bounding Box Visualization** - Find mode automatically annotates positions
- 📦 **Batch Processing** - Support for multiple image sequential recognition
- 📄 **PDF Support** - Upload PDF files, automatically convert to images
- 🎨 **Modern UI** - Cool gradient backgrounds and animation effects
- 🌐 **Multilingual Support** - Simplified Chinese, Traditional Chinese, English, Japanese
- 🐳 **Docker Deployment** - One-click startup, ready to use
- ⚡ **GPU Acceleration** - High-performance inference based on NVIDIA GPU
- 🌏 **ModelScope Fallback** - Auto-switch to ModelScope when HuggingFace is unavailable

---

## 🚀 Features

### 7 Recognition Modes

| Mode | Icon | Description | Use Cases |
|------|------|-------------|-----------|
| **Doc to Markdown** | 📄 | Preserve format and layout | Contracts, papers, reports |
| **General OCR** | 📝 | Extract all visible text | Image text extraction |
| **Plain Text** | 📋 | Pure text without format | Simple text recognition |
| **Chart Parser** | 📊 | Recognize charts and formulas | Data charts, math formulas |
| **Image Description** | 🖼️ | Generate detailed descriptions | Image understanding, accessibility |
| **Find & Locate** ⭐ | 🔍 | Find and annotate positions | Invoice field locating |
| **Custom Prompt** ⭐ | ✨ | Customize recognition needs | Flexible recognition tasks |

### 📄 PDF Support (New in v3.2)

- **Multi-page PDF Conversion**: Automatically converts each page to a separate image
- **Real-time Progress**: Shows conversion progress page by page
- **Drag & Drop**: Support drag & drop PDF upload
- **Find Mode**: PDF support in Find mode (uses first page automatically)
- **Format Validation**: Automatic file type detection and error prompts

### 🌏 ModelScope Auto-Fallback (New in v3.2)

- **Auto-Switch**: Automatically switches to ModelScope when HuggingFace is unavailable
- **Smart Detection**: Intelligently detects network errors and timeouts
- **China-Friendly**: Seamless experience for users in mainland China
- **5-minute Timeout**: Configurable timeout for model loading

### 🎨 Find Mode Features

**Left-Right Split Layout**:
```
┌──────────────────────┬─────────────────────────────┐
│   Left: Control Panel │    Right: Result Display    │
├──────────────────────┼─────────────────────────────┤
│ 📤 Image Upload      │ 🖼️ Result Image (with boxes) │
│ 🎯 Search Input      │ 📊 Statistics               │
│ 🚀 Action Buttons    │ 📝 Recognition Text         │
│                      │ 📦 Match List                │
└──────────────────────┴─────────────────────────────┘
```

**Bounding Box Visualization**:
- 🟢 Colorful neon border auto-annotation
- 🎨 6 colors in rotation
- 📍 Precise coordinate positioning
- 🔄 Responsive auto-redraw

**Feature Demo**:

<div align="center">

![Find Mode Demo](./assets/find_mode_screenshot.png)

**Find & Locate mode in action: Upload on left, auto-annotated results on right**

</div>

---

## 🌐 Multilingual Support

### Supported Languages

- 🇨🇳 **Simplified Chinese** (zh-CN)
- 🇹🇼 **Traditional Chinese** (zh-TW)
- 🇺🇸 **English** (en-US) - Default
- 🇯🇵 **Japanese** (ja-JP)

### How to Switch Language

**Web UI**:
1. Click the language selector in the top-right corner
2. Select your desired language
3. Interface switches immediately, settings auto-save

---

## 📦 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + Drivers (recommended)
- 8GB+ RAM
- 20GB+ Disk Space

### One-Click Startup

```bash
# 1. Clone repository
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI

# 2. Start service
docker compose up -d

# 3. Wait for model loading (about 1-2 minutes)
docker logs -f deepseek-ocr-webui

# 4. Access Web UI
# http://localhost:8001
```

### Verify Installation

```bash
# Check container status
docker compose ps

# Check health status
curl http://localhost:8001/health

# View logs
docker logs deepseek-ocr-webui
```

---

## 📊 Version History

### v3.2 (2025-11-04) - PDF Support & ModelScope Fallback

**📄 New Features**:
- ✅ PDF upload support (auto-convert to images)
- ✅ Multi-page PDF conversion with real-time progress
- ✅ Drag & drop PDF upload
- ✅ ModelScope auto-fallback (when HuggingFace unavailable)
- ✅ Smart network error detection and retry

**🐛 Bug Fixes**:
- ✅ Fixed PDF conversion progress logging
- ✅ Fixed button text duplication in i18n
- ✅ Fixed system initialization log information

**🔧 Technical Improvements**:
- ✅ PyMuPDF integration for high-quality PDF conversion (144 DPI)
- ✅ Async PDF processing for real-time progress
- ✅ Enhanced error handling and logging

---

### v3.1 (2025-10-22) - Multilingual & Bug Fixes

**🌐 New Features**:
- ✅ Added multilingual support (Simplified Chinese, Traditional Chinese, English, Japanese)
- ✅ Language selector UI component
- ✅ Localization persistence storage
- ✅ Multilingual documentation (README)

**🐛 Bug Fixes**:
- ✅ Fixed mode switching issues
- ✅ Fixed bounding boxes exceeding image boundaries
- ✅ Optimized image container layout
- ✅ Added rendering delay for alignment

**🎨 UI Optimization**:
- ✅ Centered image display
- ✅ Responsive bounding box redraw
- ✅ Language switcher integration

---

### v3.0 (2025-10-22) - Find Mode & Split Layout

**✨ Major Updates**:
- ✅ New Find mode (find & locate)
- ✅ Dedicated left-right split layout
- ✅ Canvas bounding box visualization
- ✅ Colorful neon annotation effects

**🔧 Technical Improvements**:
- ✅ transformers engine (replacing vLLM)
- ✅ Precise coordinate conversion algorithm
- ✅ Responsive design optimization

---

## 📖 Documentation

### User Documentation

- 📘 [Quick Start Guide](./QUICK_START.md)
- 📗 [Find Mode Guide](./FIND_MODE_V2_GUIDE.md)
- 📙 [Enhanced Features](./ENHANCED_FEATURES.md)
- 📕 [Bug Fix Summary](./BUGFIX_SUMMARY.md)

### Technical Documentation

- 🔧 [Deployment Summary](./DEPLOYMENT_SUMMARY.md)
- 📝 [Changelog](./CHANGELOG.md)
- 🌐 [I18n Implementation](./I18N_IMPLEMENTATION.md)

---

## 🎯 Usage Examples

### Find Mode Example

```bash
Scenario: Find "Total" amount in invoice

Steps:
1. Select "🔍 Find & Locate" mode
2. Upload invoice image
3. Enter search term: Total
4. Click "🚀 Start Search"

Results:
✓ "Total" marked with green border on image
✓ Shows 1-2 matches found
✓ Provides precise coordinate information
```

### Batch Processing Example

```bash
Scenario: Batch recognize 20 contracts

Steps:
1. Select "📄 Doc to Markdown" mode
2. Drag and upload 20 images
3. Adjust order (optional)
4. Click "🚀 Start Recognition"

Results:
✓ Process each image sequentially
✓ Real-time progress display
✓ Auto-merge all results
✓ One-click copy or download
```

---

## 🔧 Configuration

### Environment Variables

```bash
# docker-compose.yml
API_HOST=0.0.0.0              # Listen address
MODEL_NAME=deepseek-ai/DeepSeek-OCR  # Model name
CUDA_VISIBLE_DEVICES=0        # GPU device
```

### Performance Tuning

```yaml
# Memory configuration
shm_size: "8g"                # Shared memory

# GPU configuration
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 🤝 Contributing

Contributions welcome! Please check the [Contributing Guide](./CONTRIBUTING.md).

### How to Contribute

1. Fork this repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📞 Support

### Having Issues?

1. Check [Troubleshooting](./TROUBLESHOOTING.md)
2. Check [Known Issues](./KNOWN_ISSUES.md)
3. Submit an [Issue](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)

### Feature Suggestions?

1. Check [Roadmap](./ROADMAP.md)
2. Submit a [Feature Request](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues/new?template=feature_request.md)

---

## 📱 Follow Us

<div align="center">

![Scan to Follow](./assets/qrcode_promo.png)

**Scan to get more information**

</div>

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgments

- [DeepSeek-AI](https://github.com/deepseek-ai) - DeepSeek-OCR model
- [deepseek_ocr_app](https://github.com/rdumasia303/deepseek_ocr_app) - Reference project
- All contributors and users

---

## 🔗 Related Links

- 🏠 [Project Home](https://github.com/neosun100/DeepSeek-OCR-WebUI)
- 📖 [Full Documentation](https://github.com/neosun100/DeepSeek-OCR-WebUI/wiki)
- 🐛 [Issue Tracker](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)
- 💬 [Discussions](https://github.com/neosun100/DeepSeek-OCR-WebUI/discussions)

---

<div align="center">

**⭐ If this project helps you, please give it a Star! ⭐**

Made with ❤️ by [neosun100](https://github.com/neosun100)

DeepSeek-OCR-WebUI v3.1 | © 2025

</div>
