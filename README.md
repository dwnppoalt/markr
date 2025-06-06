# Markr — Proof of Concept

## Overview

This project is a quick proof-of-concept that uses printed ArUco fiducials to calibrate and align a whiteboard captured by a webcam. It then uses Google’s Gemini Flash API to analyze the whiteboard content and answer questions about whatever is written or drawn — formulas, illustrations, or text.

The goal was to experiment with combining computer vision (OpenCV + ArUco) and AI-powered content analysis (Google Gemini) in a minimal setup.


## Demo
[demo.webm](https://github.com/user-attachments/assets/512c58d4-9716-4fb8-b9b7-defb50658be5)


## Features

- Detects 4 ArUco markers placed at the corners of a whiteboard to correct for rotation and skew using homography.
- Captures a warped, top-down view of the whiteboard for consistent input.
- Sends the warped image to Google Gemini Flash API to generate answers based on the whiteboard content.
- Displays the AI-generated answer overlayed on a clean white background.
- Simple fiducial image generator included (`generate_fiducials.py`).

## Limitations & Notes

- This is **not production-ready**: no error handling, no modular architecture, and no extensive testing.
- The whole logic is contained in two main files plus the fiducial generator; no refactoring planned.
- Hardcoded marker IDs and whiteboard dimensions.
- Font paths are Linux-specific; fallback to default font if necessary.
- Requires a working webcam and an API key for Google Gemini (`GEMINI_KEY` in `.env`).
- Runs on Python with OpenCV, PIL/Pillow, and `google-generativeai` Python package.
- Only tested on personal setup; your mileage may vary.

## How to Use

1. Print the fiducial markers in `fiducials/` (generated by `generate_fiducials.py`) and stick them at the whiteboard corners.
2. Set your Google Gemini API key in a `.env` file as `GEMINI_KEY=your_api_key_here`.
3. Run `main.py`.
4. Show the entire whiteboard with all fiducials visible to the webcam.
5. Wait until the system aligns the board and processes the content.
6. See the AI-generated answer overlay.

Press `q` to quit the program.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- Pillow (`PIL`)
- `google-generativeai` Python package
- `python-dotenv`

Install dependencies with:

```bash
pip install -r requirements.txt
```
