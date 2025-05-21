from calibrate import calibrate_from_frame
import cv2
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_KEY"))

def ai_gen(image: Image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [image, "Return just the answer, do not add any additional text unless asked 'why' and/or explicitly stated."],
    )
    return response.text
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

result = None 
whiteboard_ready_frames = 0
whiteboard_required_frames = 10 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    warped_frame = calibrate_from_frame(frame)
    show_warped = False
    if result is None:
        if warped_frame is not None:
            whiteboard_ready_frames += 1
            cv2.putText(frame, f"Aligning...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if whiteboard_ready_frames >= whiteboard_required_frames:
                cv2.putText(frame, "Checking...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)
                time.sleep(0.5)
                pil_image = Image.fromarray(cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB))
                result = ai_gen(pil_image)
                print(result)
                show_warped = True
        else:
            whiteboard_ready_frames = 0
            cv2.putText(frame, "Show all 4 fiducials", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        show_warped = True

    if show_warped and warped_frame is not None:
        cv2.imshow("Warped Board", warped_frame)
        
        answer_width = 800
        answer_height = 400
        answer_overlay = np.ones((answer_height, answer_width, 3), dtype=np.uint8) * 255
        pil_img = Image.fromarray(answer_overlay)
        draw = ImageDraw.Draw(pil_img)

        font_path = None
        # because HERSHEY_SIMPLEX does not render math symbols well :/
        for f in ["/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
            if os.path.exists(f):
                font_path = f
                break

        if font_path is None:
            font = ImageFont.load_default()
            initial_font_size = 32
        else:
            initial_font_size = 72  
            font = ImageFont.truetype(font_path, initial_font_size)

        margin = 40  
        max_width = answer_width - (2 * margin)
        max_height = answer_height - (2 * margin)
        words = result.split() if result else []
        text = " ".join(words)

        min_size = 12
        max_size = initial_font_size
        optimal_size = min_size
        
        while min_size <= max_size:
            mid_size = (min_size + max_size) // 2
            test_font = ImageFont.truetype(font_path, mid_size) if font_path else ImageFont.load_default()
            
            lines = []
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                bbox = test_font.getbbox(test_line)
                if bbox[2] - bbox[0] > max_width:
                    if line:
                        lines.append(line)
                        line = word
                    else:
                        line = word
                else:
                    line = test_line
            if line:
                lines.append(line)
            

            total_height = len(lines) * (mid_size + 4) 
            
            if total_height <= max_height and all(test_font.getbbox(line)[2] - test_font.getbbox(line)[0] <= max_width for line in lines):
                optimal_size = mid_size
                min_size = mid_size + 1
            else:
                max_size = mid_size - 1

        font = ImageFont.truetype(font_path, optimal_size) if font_path else ImageFont.load_default()
        line_height = optimal_size + 4


        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] > max_width and line:
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)

        total_height = len(lines) * line_height
        start_y = (answer_height - total_height) // 2

        y = start_y
        for l in lines:
            bbox = font.getbbox(l)
            w = bbox[2] - bbox[0]
            x = (answer_width - w) // 2 
            draw.text((x, y), l, font=font, fill=(0, 0, 0))
            y += line_height

        answer_overlay = np.array(pil_img)
        cv2.imshow("Answer", answer_overlay)
        
        result = None
        whiteboard_ready_frames = 0
        cv2.waitKey(1000)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
