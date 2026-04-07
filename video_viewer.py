from __future__ import annotations

from typing import Optional

import cv2

try:
    import tkinter as tk
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    tk = None
    Image = None
    ImageTk = None


class FrameViewer:
    def __init__(self, title: str) -> None:
        self.title = title
        self.mode = self._detect_mode()
        self.root: Optional["tk.Tk"] = None
        self.label = None
        self.photo = None
        self.closed = False

        if self.mode == "tk":
            self.root = tk.Tk()
            self.root.title(title)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.label = tk.Label(self.root)
            self.label.pack()
            helper = tk.Label(self.root, text="Press q in this window to stop")
            helper.pack()
            self.root.bind("<KeyPress-q>", lambda _event: self._on_close())

    def _detect_mode(self) -> str:
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(self.title)
            return "opencv"
        except cv2.error:
            if tk is not None and Image is not None and ImageTk is not None:
                return "tk"
            return "none"

    def show(self, frame) -> bool:
        if self.mode == "opencv":
            cv2.imshow(self.title, frame)
            return cv2.waitKey(30) & 0xFF != ord("q")

        if self.mode == "tk" and self.root is not None and self.label is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            self.photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self.photo)
            self.root.update_idletasks()
            self.root.update()
            return not self.closed

        return True

    def close(self) -> None:
        if self.mode == "opencv":
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        elif self.mode == "tk" and self.root is not None:
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def _on_close(self) -> None:
        self.closed = True
