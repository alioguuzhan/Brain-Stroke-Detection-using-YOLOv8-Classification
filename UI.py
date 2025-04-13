from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Toplevel
from ultralytics import YOLO

# Modeli yükle
model = YOLO('/Users/alioguzhan/Desktop/Python/Brain Stroke Classification/yolov8cls-m/best.pt')

# Ana pencere
root = tk.Tk()
root.title("Brain Stroke Detection")
root.geometry("800x600")

# Arka plan
bg_image = Image.open('/Users/alioguzhan/Desktop/Python/Brain Stroke Classification/ChatGPT Image 11 Nis 2025 14_31_43.png').resize((800, 600))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Panel başlangıçta None
panel = None

# Tahmin fonksiyonu
def predict_image():
    global panel
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(image)

        if panel is None:
            panel = tk.Label(root, bg="white")
            panel.place(x=490, y=165, width=224, height=224)

        panel.config(image=img_tk)
        panel.image = img_tk

        results = model(file_path)
        probs = results[0].probs
        class_id = probs.top1
        class_name = results[0].names[class_id]
        confidence = probs.data[class_id].item()
        result_text = f"{class_name.upper()} ({confidence:.2%})"

        result_window = Toplevel(root)
        result_window.title("Tahmin Sonucu")
        result_window.geometry("300x100")

        color = "green" if "normal" in class_name.lower() else "red"
        label = tk.Label(result_window, text=result_text, font=("Helvetica", 16, "bold"), fg=color)
        label.pack(padx=20, pady=20)

# 👇 Görsel Butonu Yükle ve Yerleştir
btn_img = Image.open('/Users/alioguzhan/Desktop/Python/Brain Stroke Classification/Ekran Resmi 2025-04-11 17.44.21.png')
btn_img = btn_img.resize((200, 40))  # Uygun boyut
btn_img_tk = ImageTk.PhotoImage(btn_img)

btn_label = tk.Label(root, image=btn_img_tk, borderwidth=0, cursor="hand2")
btn_label.image = btn_img_tk
btn_label.place(x=505, y=430)
btn_label.bind("<Button-1>", lambda e: predict_image())  # Tıklama olayına fonksiyon bağla

root.mainloop()
