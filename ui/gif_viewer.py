from __future__ import annotations

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


class GIFViewer(tk.Frame):
    """Embedded GIF animation surface used by EDITH's main Tk window."""

    def __init__(
        self,
        parent: tk.Misc,
        gif_path: str | Path,
        title: str = "EDITH AI",
    ) -> None:
        super().__init__(
            parent,
            bg="black",
            bd=0,
            highlightthickness=0,
        )

        self.gif_path = Path(gif_path).expanduser().resolve()
        self.title = title
        self.frames: list[ImageTk.PhotoImage] = []
        self.delay = 100
        self.current_frame = 0
        self.animation_id: str | None = None
        self.closed = False

        self.label = tk.Label(
            self,
            bg="black",
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        self.label.pack(fill="both", expand=True)

        self.load_gif()

        if not self.frames:
            raise RuntimeError(
                f"EDITH GIF could not be loaded: {self.gif_path}"
            )

        self.label.configure(image=self.frames[0])
        self.label.image = self.frames[0]

    def load_gif(self) -> None:
        if not self.gif_path.is_file():
            raise FileNotFoundError(
                f"EDITH animation GIF not found: {self.gif_path}"
            )

        try:
            with Image.open(self.gif_path) as image:
                frame_count = getattr(image, "n_frames", 1)
                self.delay = max(
                    20,
                    int(image.info.get("duration", 100) or 100),
                )

                for index in range(frame_count):
                    image.seek(index)
                    frame = image.convert("RGBA")
                    photo = ImageTk.PhotoImage(frame, master=self)
                    self.frames.append(photo)

            print(
                "EDITH UI: GIF loaded successfully -> "
                f"{self.gif_path} | frames={len(self.frames)} | "
                f"delay={self.delay}ms"
            )

        except Exception as exc:
            print(
                f"EDITH UI: Failed to load GIF '{self.gif_path}': {exc}"
            )
            raise

    def start_animation(self) -> None:
        if self.closed or not self.frames or self.animation_id is not None:
            return

        self.animate_gif()

    def animate_gif(self) -> None:
        if self.closed or not self.frames:
            return

        frame = self.frames[self.current_frame]
        self.label.configure(image=frame)
        self.label.image = frame
        self.current_frame = (self.current_frame + 1) % len(self.frames)

        self.animation_id = self.after(
            self.delay,
            self.animate_gif,
        )

    def set_status(self, text: str) -> None:
        """Reserved for state-specific visual effects."""
        _ = text

    def stop_animation(self) -> None:
        if self.closed:
            return

        self.closed = True

        if self.animation_id is not None:
            try:
                self.after_cancel(self.animation_id)
            except tk.TclError:
                pass
            self.animation_id = None

        try:
            self.destroy()
        except tk.TclError:
            pass

    def restart_animation(self) -> None:
        if self.closed:
            return

        self.current_frame = 0
        self.start_animation()


def show_startup_gif(
    root: tk.Misc,
    gif_path: str | Path,
) -> GIFViewer:
    """Compatibility helper for existing callers."""
    viewer = GIFViewer(root, gif_path)
    viewer.pack()
    viewer.start_animation()
    return viewer
