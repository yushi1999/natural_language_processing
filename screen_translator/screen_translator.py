from threading import Thread
from PIL import ImageGrab
import pytesseract
from pynput.keyboard import Key, Listener
import time
import tkinter as tk
from requests import post
from keyboard_monitor import KeyMonitor
from mouse_monitor import MouseMonitor
from dotenv import load_dotenv
import os

"""
スクリーンショットキーの押下を検出
左クリックを押した位置と離した位置を記録
クリップボード上のスクショ画像に対してOCR
DeepL APIに投げる
記録した領域にTkinterウィンドウ作成、原文を表示
返ってきた翻訳結果で上書き
"""

load_dotenv(".env")
AUTH_KEY = os.environ.get("AUTH_KEY")


class ImageToString:
    def __init__(self):
        self.text = ""

    def run(self, lang="jpn"):
        img = ImageGrab.grabclipboard()
        if img:
            self.text = pytesseract.image_to_string(img, lang=lang)
        return self.text


class Translator:
    def __init__(self, auth_key):
        self.data = {
            "source_lang": "KO",
            "target_lang": "JA",
            "split_sentences": "nonewlines",
            "auth_key": auth_key,
        }

    def translate(self, tk, text):
        task = Thread(
            target=self._translate,
            args=(
                tk,
                text,
            ),
            daemon=True,
        )
        task.start()

    def _translate(self, tk, text):
        tr_text = (
            post(
                "https://api-free.deepl.com/v2/translate",
                data=self.data | {"text": text},
            )
            .json()["translations"][0]
            .get("text")
        )
        print(tr_text)
        tk.label["text"] = tr_text
        x1, x2, y1, y2 = tk.coor
        tk.geometry(
            f"{x2-x1}x{max(tk.label.winfo_reqheight()+40, y2-y1)}+{x1}+{y1}")


class Application(tk.Tk):
    def __init__(self, x1, x2, y1, y2):
        super().__init__()
        self.coor = x1, x2, y1, y2
        self.title("Translation text")
        self.attributes("-alpha", 0.95)
        self.attributes("-topmost", True)
        self.bind("<Configure>", self.sized)
        self.bind("<Shift-ButtonPress-1>", self.toggleOverrideRedirect)
        time.sleep(0.5)
        c = ImageToString()
        self.text = c.run("kor")
        self.label = tk.Label(
            self,
            font=("ヒラギノ角ゴシック", "18"),
            anchor="e",
            justify="left",
            text=self.text,
        )
        tr = Translator(AUTH_KEY)
        tr.translate(self, self.text)
        self.label.pack(expand=True)
        self.geometry(
            f"{x2-x1+20}x{max(self.label.winfo_reqheight()+30, y2-y1)}+{x1}+{y1}"
        )

    def sized(self, *args):
        self.label["wraplength"] = self.winfo_width() - 40

    def toggleOverrideRedirect(ev):
        win = ev.widget.winfo_toplevel()
        win.overrideredirect(not win.overrideredirect())
        win.withdraw()
        win.deiconify()
        win.focus_force()
        return


class ScreenTranslator:
    def __init__(self, AUTH_KEY: str):
        self.AUTH_KEY = AUTH_KEY

    def start(self):
        key = KeyMonitor()  # キーイベントを検出
        key_th = Thread(target=key.start, args=(),
                        daemon=True)  # スレッド処理を行う
        key_th.start()

        while True:
            # スクリーンショットコマンドが押されている時
            if key.is_pressed("4") and key.is_pressed(Key.cmd) and key.is_pressed(Key.shift) and key.is_pressed(Key.ctrl):
                print("4 keys are pushed.")
                app = Application(*self.get_rectcoordinate())
                # app.overrideredirect(1)
                print("mainloop.")
                app.mainloop()

    # 左クリックをした位置と離した位置の座標を取得する
    def get_rectcoordinate(self):
        mou = MouseMonitor()  # マウスイベントを検出
        mou_th = Thread(target=mou.start, args=(),
                        daemon=True)
        mou_th.start()
        while True:
            if mou.released:
                print("clicked.")
                return mou.coordinate()


sct = ScreenTranslator(AUTH_KEY)
sct.start()
