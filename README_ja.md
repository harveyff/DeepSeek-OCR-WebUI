# 🔍 DeepSeek-OCR-WebUI

<div align="center">

**🌐 [English](./README.md) | [简体中文](./README_zh-CN.md) | [繁體中文](./README_zh-TW.md) | [日本語](./README_ja.md)**

[![Version](https://img.shields.io/badge/version-v3.2-blue.svg)](./CHANGELOG.md)
[![Docker](https://img.shields.io/badge/docker-supported-brightgreen.svg)](./docker-compose.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Language](https://img.shields.io/badge/languages-4-orange.svg)](#多言語サポート)

インテリジェントOCR認識システム · バッチ処理 · マルチモードサポート · バウンディングボックス可視化

[機能](#機能) • [クイックスタート](#クイックスタート) • [バージョン履歴](#バージョン履歴) • [ドキュメント](#ドキュメント) • [貢献](#貢献)

</div>

---

## 📖 はじめに

DeepSeek-OCR-WebUIは、DeepSeek-OCRモデルをベースにしたインテリジェント画像認識Webアプリケーションで、直感的なユーザーインターフェースと強力な認識機能を提供します。

### 🖼️ UIプレビュー

<div align="center">

![DeepSeek-OCR-WebUI インターフェース](./assets/ui_screenshot.png)

**多言語切り替え、バッチ処理、バウンディングボックス可視化をサポートする現代的なユーザーインターフェース**

</div>

### 📈 Star 成長曲線

<div align="center">

![Star History Chart](https://api.star-history.com/svg?repos=neosun100/DeepSeek-OCR-WebUI&type=Date)

**Star 成長曲線 - 私たちの成長を助けてください！⭐**

</div>

### ✨ コアハイライト

- 🎯 **7つの認識モード** - 文書、OCR、チャート、Find、Freeformなど
- 🖼️ **バウンディングボックス可視化** - Findモードで位置を自動注釈
- 📦 **バッチ処理** - 複数画像の順次認識をサポート
- 📄 **PDF サポート** - PDFファイルをアップロードし、自動的に画像に変換
- 🎨 **モダンUI** - クールなグラデーション背景とアニメーション効果
- 🌐 **多言語サポート** - 簡体字中国語、繁体字中国語、英語、日本語
- 🐳 **Docker デプロイ** - ワンクリック起動、すぐに使える
- ⚡ **GPU アクセラレーション** - NVIDIA GPUベースの高性能推論
- 🌏 **ModelScope 自動切り替え** - HuggingFaceが利用できない場合、自動的にModelScopeに切り替え

---

## 🚀 機能

### 7つの認識モード

| モード | アイコン | 説明 | 使用例 |
|------|------|------|--------|
| **文書→Markdown** | 📄 | フォーマットとレイアウトを保持 | 契約書、論文、レポート |
| **汎用OCR** | 📝 | すべての可視テキストを抽出 | 画像テキスト抽出 |
| **プレーンテキスト抽出** | 📋 | フォーマットなしの純粋なテキスト | シンプルなテキスト認識 |
| **チャート解析** | 📊 | チャートと数式を認識 | データチャート、数学式 |
| **画像説明** | 🖼️ | 詳細な説明を生成 | 画像理解、アクセシビリティ |
| **検索と位置特定** ⭐ | 🔍 | 検索して位置を注釈 | 請求書フィールド位置特定 |
| **カスタムプロンプト** ⭐ | ✨ | 認識ニーズをカスタマイズ | 柔軟な認識タスク |

### 📄 PDF サポート（v3.2 新機能）

- **複数ページPDF変換**：各ページを自動的に個別の画像に変換
- **リアルタイム進捗表示**：ページごとに変換進捗を表示
- **ドラッグ&ドロップ**：PDFファイルのドラッグ&ドロップアップロードをサポート
- **Findモードサポート**：FindモードでPDFをサポート（最初のページを自動使用）
- **フォーマット検証**：自動ファイルタイプ検出とエラーメッセージ

### 🌏 ModelScope 自動切り替え（v3.2 新機能）

- **自動切り替え**：HuggingFaceが利用できない場合、自動的にModelScopeに切り替え
- **インテリジェント検出**：ネットワークエラーとタイムアウトをインテリジェントに検出
- **中国対応**：中国本土のユーザーにシームレスな体験を提供
- **5分タイムアウト**：設定可能なモデル読み込みタイムアウト

### 🎨 Findモードの特徴

**左右分割レイアウト**：
```
┌──────────────────────┬─────────────────────────────┐
│   左：操作パネル      │    右：結果表示              │
├──────────────────────┼─────────────────────────────┤
│ 📤 画像アップロード   │ 🖼️ 結果画像（ボックス付き）  │
│ 🎯 検索語入力        │ 📊 統計情報                  │
│ 🚀 アクションボタン   │ 📝 認識テキスト              │
│                      │ 📦 マッチリスト               │
└──────────────────────┴─────────────────────────────┘
```

**バウンディングボックス可視化**：
- 🟢 カラフルなネオンボーダー自動注釈
- 🎨 6色のローテーション
- 📍 精密な座標位置特定
- 🔄 レスポンシブ自動再描画

**機能デモ**：

<div align="center">

![Findモードデモ](./assets/find_mode_screenshot.png)

**検索と位置特定モードの実際の効果：左側でアップロード操作、右側で自動注釈結果**

</div>

---

## 🌐 多言語サポート

### サポートされている言語

- 🇨🇳 **簡体字中国語** (zh-CN)
- 🇹🇼 **繁体字中国語** (zh-TW)
- 🇺🇸 **英語** (en-US)
- 🇯🇵 **日本語** (ja-JP) - デフォルト

### 言語の切り替え方法

**Web UI**：
1. アプリの右上隅にある言語セレクターをクリック
2. 必要な言語を選択
3. インターフェースが即座に切り替わり、設定が自動保存されます

---

## 📦 クイックスタート

### 前提条件

- Docker & Docker Compose
- NVIDIA GPU + ドライバー（推奨）
- 8GB以上のRAM
- 20GB以上のディスクスペース

### ワンクリック起動

```bash
# 1. リポジトリをクローン
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI

# 2. サービスを起動
docker compose up -d

# 3. モデルのロードを待つ（約1〜2分）
docker logs -f deepseek-ocr-webui

# 4. Web UIにアクセス
# http://localhost:8001
```

### インストールの確認

```bash
# コンテナのステータスを確認
docker compose ps

# ヘルスステータスを確認
curl http://localhost:8001/health

# ログを表示
docker logs deepseek-ocr-webui
```

---

## 📊 バージョン履歴

### v3.2 (2025-11-04) - PDF サポートと ModelScope 自動切り替え

**📄 新機能**：
- ✅ PDFアップロードサポート（自動的に画像に変換）
- ✅ 複数ページPDF変換、リアルタイム進捗表示
- ✅ ドラッグ&ドロップPDFアップロードサポート
- ✅ ModelScope自動切り替え（HuggingFaceが利用できない場合）
- ✅ インテリジェントネットワークエラー検出とリトライ

**🐛 バグ修正**：
- ✅ PDF変換進捗ログのリアルタイム表示問題を修正
- ✅ i18nでのボタンテキスト重複表示問題を修正
- ✅ システム初期化ログ情報を修正

**🔧 技術的改善**：
- ✅ 高品質PDF→画像変換のためのPyMuPDF統合（144 DPI）
- ✅ リアルタイム進捗表示のための非同期PDF処理
- ✅ エラーハンドリングとログ記録の強化

---

### v3.1 (2025-10-22) - 多言語とバグ修正

**🌐 新機能**：
- ✅ 多言語サポートを追加（簡体字中国語、繁体字中国語、英語、日本語）
- ✅ 言語セレクターUIコンポーネント
- ✅ ローカライゼーション永続化ストレージ
- ✅ 多言語ドキュメント（README）

**🐛 バグ修正**：
- ✅ モード切り替えの問題を修正
- ✅ バウンディングボックスが画像境界を超える問題を修正
- ✅ 画像コンテナレイアウトを最適化
- ✅ アライメントのためのレンダリング遅延を追加

**🎨 UI最適化**：
- ✅ 画像を中央に表示
- ✅ バウンディングボックスのレスポンシブ再描画
- ✅ 言語スイッチャーの統合

---

### v3.0 (2025-10-22) - Findモードと分割レイアウト

**✨ 主要な更新**：
- ✅ 新しいFindモード（検索と位置特定）
- ✅ 専用の左右分割レイアウト
- ✅ Canvasバウンディングボックス可視化
- ✅ カラフルなネオン注釈効果

**🔧 技術的改善**：
- ✅ transformersエンジン（vLLMを置き換え）
- ✅ 精密な座標変換アルゴリズム
- ✅ レスポンシブデザインの最適化

---

## 📖 ドキュメント

### ユーザードキュメント

- 📘 [クイックスタートガイド](./QUICK_START.md)
- 📗 [Findモード使用ガイド](./FIND_MODE_V2_GUIDE.md)
- 📙 [拡張機能説明](./ENHANCED_FEATURES.md)
- 📕 [バグ修正まとめ](./BUGFIX_SUMMARY.md)

### 技術ドキュメント

- 🔧 [デプロイメントまとめ](./DEPLOYMENT_SUMMARY.md)
- 📝 [変更ログ](./CHANGELOG.md)
- 🌐 [国際化実装](./I18N_IMPLEMENTATION.md)

---

## 🎯 使用例

### Findモードの例

```bash
シナリオ：請求書で「Total」金額を検索

手順：
1. "🔍 検索と位置特定"モードを選択
2. 請求書画像をアップロード
3. 検索語を入力：Total
4. "🚀 検索開始"をクリック

結果：
✓ 画像上の"Total"が緑色のボーダーでマーク
✓ 1〜2個のマッチが見つかったことを表示
✓ 正確な座標情報を提供
```

### バッチ処理の例

```bash
シナリオ：20枚の契約書をバッチ認識

手順：
1. "📄 文書→Markdown"モードを選択
2. 20枚の画像をドラッグしてアップロード
3. 順序を調整（オプション）
4. "🚀 認識開始"をクリック

結果：
✓ 各画像を順次処理
✓ リアルタイムで進捗を表示
✓ すべての結果を自動マージ
✓ ワンクリックでコピーまたはダウンロード
```

---

## 🔧 設定

### 環境変数

```bash
# docker-compose.yml
API_HOST=0.0.0.0              # リッスンアドレス
MODEL_NAME=deepseek-ai/DeepSeek-OCR  # モデル名
CUDA_VISIBLE_DEVICES=0        # GPUデバイス
```

### パフォーマンスチューニング

```yaml
# メモリ設定
shm_size: "8g"                # 共有メモリ

# GPU設定
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 🤝 貢献

貢献を歓迎します！[貢献ガイド](./CONTRIBUTING.md)をご確認ください。

### 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを開く

---

## 📞 サポート

### 問題がありますか？

1. [トラブルシューティング](./TROUBLESHOOTING.md)を確認
2. [既知の問題](./KNOWN_ISSUES.md)を確認
3. [Issue](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)を提出

### 機能の提案？

1. [ロードマップ](./ROADMAP.md)を確認
2. [機能リクエスト](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues/new?template=feature_request.md)を提出

---

## 📱 フォローしてください

<div align="center">

![スキャンしてフォロー](./assets/qrcode_promo.png)

**スキャンして詳細情報を入手**

</div>

---

## 📄 ライセンス

このプロジェクトは[MITライセンス](./LICENSE)の下でライセンスされています。

---

## 🙏 謝辞

- [DeepSeek-AI](https://github.com/deepseek-ai) - DeepSeek-OCRモデル
- [deepseek_ocr_app](https://github.com/rdumasia303/deepseek_ocr_app) - 参考プロジェクト
- すべての貢献者とユーザー

---

## 🔗 関連リンク

- 🏠 [プロジェクトホーム](https://github.com/neosun100/DeepSeek-OCR-WebUI)
- 📖 [完全なドキュメント](https://github.com/neosun100/DeepSeek-OCR-WebUI/wiki)
- 🐛 [問題追跡](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)
- 💬 [ディスカッション](https://github.com/neosun100/DeepSeek-OCR-WebUI/discussions)

---

<div align="center">

**⭐ このプロジェクトがお役に立ちましたら、Starをください！⭐**

Made with ❤️ by [neosun100](https://github.com/neosun100)

DeepSeek-OCR-WebUI v3.1 | © 2025

</div>
