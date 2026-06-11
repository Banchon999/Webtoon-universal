# Webtoon Translator

แอพเดสก์ท็อปแปล webtoon / มังงะ / คอมิกครบวงจร — ตรวจจับบับเบิล อ่านข้อความ ลบข้อความเดิม แปลภาษา จัดวางตัวอักษรใหม่ และ export ภาพ

A desktop app that translates webtoons/comics end-to-end: bubble detection, OCR, text removal (inpainting), machine translation with glossary, typesetting, and export.

## Features / ฟีเจอร์

| ขั้นตอน | เทคโนโลยี |
|---|---|
| 1. ตรวจจับบับเบิลและข้อความ | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) (RT-DETR-v2, **int8 ONNX** — เร็วระดับวินาทีต่อหน้าบน CPU) + tiled inference สำหรับภาพยาว |
| 2. อ่านข้อความ (OCR) | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) (VLM 1B รองรับหลายภาษา) |
| 3. ลบข้อความเดิม (inpainting) | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) + fast-fill สำหรับบับเบิลพื้นเรียบ |
| 4. แปลภาษา + Glossary | [OpenRouter](https://openrouter.ai) (เลือกโมเดล LLM ได้อิสระ) |
| 5. จัดวางตัวอักษร | Pillow + OpenCV, ตัดคำไทยด้วย PyThaiNLP, auto font-size |
| 6. Export | PNG / JPG / WebP + บันทึกโปรเจกต์ (.wtproj) |

- รองรับการแปล **ทุกภาษา → ทุกภาษา** (รวมไทย, มี glossary บังคับคำแปลศัพท์เฉพาะ)
- แก้ไขผล OCR / คำแปล / ฟอนต์ ได้ทีละบับเบิลใน GUI
- โมเดล AI ดาวน์โหลดอัตโนมัติครั้งแรกที่เปิดแอพ (~2.3 GB)
- รัน CPU ได้ทุกเครื่อง; ใช้ CUDA อัตโนมัติเมื่อรันจาก source พร้อม GPU
- ปรับจูนความเร็วได้ใน Settings: OCR batch size, จำนวน CPU threads; ขั้นแปลภาษา (รอ API) ทำงานซ้อนกับขั้นลบข้อความโดยอัตโนมัติ

## Download (Windows)

ดาวน์โหลด `WebtoonTranslator-win64-*.zip` จากหน้า [Releases](../../releases) (หรือ Actions artifacts สำหรับ build ล่าสุด) แตก zip แล้วเปิด `WebtoonTranslator/WebtoonTranslator.exe`

> เปิดครั้งแรกแอพจะดาวน์โหลดโมเดล AI ~2.3 GB และต้องใส่ OpenRouter API key ใน **Tools > Settings** ก่อนใช้งานแปล (สมัครฟรีที่ [openrouter.ai](https://openrouter.ai))

ความต้องการระบบ: Windows 10/11 64-bit, RAM 8 GB ขึ้นไป

## วิธีใช้

1. **Import** — ลากรูปหรือโฟลเดอร์ลงในรายการด้านซ้าย (รองรับภาพยาวแบบ webtoon)
2. **Settings** — ใส่ OpenRouter API key, เลือกภาษาต้นทาง/ปลายทาง และโมเดลแปล
3. **Glossary** (ตัวเลือก) — เพิ่มคำศัพท์เฉพาะ เช่น ชื่อตัวละคร ให้แปลตรงกันทุกหน้า
4. กด **▶ Translate all pages**
5. ตรวจ/แก้คำแปลทีละบับเบิล แล้วกด **Re-typeset**
6. **File > Export translated images…**

## Run from source

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
python -m webtoon_translator
```

GPU (NVIDIA): ติดตั้ง torch รุ่น CUDA แทน แล้วแอพจะใช้ GPU อัตโนมัติ

### Headless CLI

```bash
python scripts/run_pipeline_cli.py page1.png page2.png -o out/ \
    --target-lang th --api-key sk-or-... --model google/gemini-2.5-flash
```

### Tests

```bash
pytest          # unit tests (ไม่ต้องดาวน์โหลดโมเดล)
ruff check src tests scripts
```

## Build the exe

GitHub Actions (`.github/workflows/build-windows.yml`) build ให้อัตโนมัติเมื่อ push tag `v*` หรือกด **Run workflow** เอง — ได้ zip ของ PyInstaller onedir build

## Project structure

```
src/webtoon_translator/
  core/       data model, project IO, glossary, device selection
  pipeline/   detector, ocr, inpaint, translator, typeset, export (GUI-free)
  download/   model download manager (pinned HF revisions)
  gui/        PySide6 main window, canvas, editors, workers
assets/fonts/ bundled OFL fonts (Noto Sans, Sarabun)
packaging/    PyInstaller spec
scripts/      headless CLI + sample generator
```

## Licenses

- โค้ดโปรเจกต์: ตามไฟล์ [LICENSE](LICENSE)
- ฟอนต์ Noto Sans และ Sarabun: [SIL Open Font License 1.1](https://openfontlicense.org)
- โมเดล AI: ตาม license ของแต่ละ repo บน Hugging Face
