"""
Clone Hero Chart Generator — GUI App
Beautiful dark UI with drag & drop, progress tracking, and custom backgrounds.
"""

import sys
import os
import json
import threading
import subprocess
import queue
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, StringVar, BooleanVar
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#1DB954"
BG_DARK  = "#0f0f0f"
BG_CARD  = "#1a1a1a"
BG_INPUT = "#242424"
TEXT_DIM = "#888888"

CONFIG_PATH = Path(__file__).parent / "app_config.json"

THEMES = {
    "Default Dark":   {"accent": "#1DB954", "bg": "#0f0f0f", "card": "#1a1a1a"},
    "Blue Steel":     {"accent": "#4488ff", "bg": "#0a0e1a", "card": "#111827"},
    "Purple Haze":    {"accent": "#a855f7", "bg": "#0d0a1a", "card": "#1a1428"},
    "Fire Red":       {"accent": "#ef4444", "bg": "#120a0a", "card": "#1c1010"},
    "Gold Rush":      {"accent": "#f59e0b", "bg": "#111009", "card": "#1c1a10"},
}

# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

# ── Main App ──────────────────────────────────────────────────────────────────

class ChartGenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.cfg = load_config()
        self.songs: list[str] = []
        self.log_queue = queue.Queue()
        self.running = False
        self._bg_image_ref = None
        self._bg_label = None

        # Apply saved theme
        self._current_theme = self.cfg.get("theme", "Default Dark")
        self._apply_theme(self._current_theme, rebuild=False)

        self.title("Clone Hero Chart Generator")
        self.geometry("860x720")
        self.minsize(700, 600)
        self.configure(fg_color=BG_DARK)

        self._build_ui()
        self._apply_bg_image(self.cfg.get("bg_image", ""))
        self._poll_log()
        self.bind("<Configure>", self._on_resize)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self, name, rebuild=True):
        global ACCENT, BG_DARK, BG_CARD, BG_INPUT
        t = THEMES.get(name, THEMES["Default Dark"])
        ACCENT   = t["accent"]
        BG_DARK  = t["bg"]
        BG_CARD  = t["card"]
        self._current_theme = name
        self.cfg["theme"] = name
        save_config(self.cfg)
        if rebuild:
            self.configure(fg_color=BG_DARK)
            self._rebuild_ui()

    def _rebuild_ui(self):
        for w in self.winfo_children():
            w.destroy()
        self._bg_label = None
        self._build_ui()
        self._apply_bg_image(self.cfg.get("bg_image", ""))

    # ── Background Image ──────────────────────────────────────────────────────

    def _apply_bg_image(self, path):
        if not path or not os.path.exists(path):
            if self._bg_label:
                self._bg_label.destroy()
                self._bg_label = None
            return
        try:
            img = Image.open(path)
            w, h = self.winfo_width() or 860, self.winfo_height() or 720
            img = img.resize((w, h), Image.LANCZOS)
            img = img.filter(ImageFilter.GaussianBlur(radius=8))
            # Darken overlay
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 160))
            img = img.convert("RGBA")
            img = Image.alpha_composite(img, overlay).convert("RGB")
            self._bg_image_ref = ImageTk.PhotoImage(img)
            if not self._bg_label:
                self._bg_label = tk.Label(self, image=self._bg_image_ref)
                self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                self._bg_label.lower()
            else:
                self._bg_label.configure(image=self._bg_image_ref)
        except Exception as e:
            print(f"BG image error: {e}")

    def _on_resize(self, event):
        if self.cfg.get("bg_image") and event.widget == self:
            self.after(150, lambda: self._apply_bg_image(self.cfg.get("bg_image", "")))

    def _browse_bg(self):
        f = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if f:
            self.cfg["bg_image"] = f
            save_config(self.cfg)
            self._apply_bg_image(f)

    def _clear_bg(self):
        self.cfg["bg_image"] = ""
        save_config(self.cfg)
        if self._bg_label:
            self._bg_label.destroy()
            self._bg_label = None

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        ctk.CTkLabel(
            title_frame, text="🎸  Clone Hero Chart Generator",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=ACCENT
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(
            title_frame, text="v6 + ChartNet Neural",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        ).pack(side="left", pady=15)

        # Settings gear button
        ctk.CTkButton(
            title_frame, text="⚙", width=36, height=36,
            fg_color="transparent", hover_color="#333333",
            font=ctk.CTkFont(size=18),
            command=self._open_settings
        ).pack(side="right", padx=10)

        # Main content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=15)

        left = ctk.CTkFrame(content, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right = ctk.CTkFrame(content, fg_color="transparent", width=240)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_drop_zone(left)
        self._build_song_list(left)
        self._build_settings_panel(right)
        self._build_bottom()

    def _build_drop_zone(self, parent):
        self.drop_frame = ctk.CTkFrame(
            parent, fg_color=BG_CARD, corner_radius=12,
            border_width=2, border_color="#333333", height=110
        )
        self.drop_frame.pack(fill="x", pady=(0, 10))
        self.drop_frame.pack_propagate(False)

        self.drop_frame.bind("<Button-1>", lambda e: self._browse_songs())
        self.drop_frame.bind("<Enter>", lambda e: self.drop_frame.configure(border_color=ACCENT))
        self.drop_frame.bind("<Leave>", lambda e: self.drop_frame.configure(border_color="#333333"))

        inner = ctk.CTkFrame(self.drop_frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(inner, text="🎵", font=ctk.CTkFont(size=28)).pack()
        ctk.CTkLabel(inner, text="Click to add MP3 files",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="white").pack()
        ctk.CTkLabel(inner, text="Batch generation supported",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack()

        for w in inner.winfo_children():
            w.bind("<Button-1>", lambda e: self._browse_songs())

    def _build_song_list(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(header, text="Songs", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear all", width=65, height=24,
            font=ctk.CTkFont(size=11), fg_color="#333333", hover_color="#444444",
            command=self._clear_songs).pack(side="right")

        self.song_list_frame = ctk.CTkScrollableFrame(
            parent, fg_color=BG_CARD, corner_radius=10, height=160)
        self.song_list_frame.pack(fill="x", pady=(0, 10))

        self.empty_label = ctk.CTkLabel(self.song_list_frame,
            text="No songs added yet", text_color=TEXT_DIM, font=ctk.CTkFont(size=12))
        self.empty_label.pack(pady=20)

        ctk.CTkLabel(parent, text="Output Log",
            font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 5))

        self.log_box = ctk.CTkTextbox(
            parent, fg_color=BG_CARD, corner_radius=10,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#cccccc", height=180, wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def _build_settings_panel(self, parent):
        ctk.CTkLabel(parent, text="Settings",
            font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 10))

        # Output folder
        self._setting_label(parent, "Output Folder")
        self.out_var = StringVar(value=self.cfg.get("out_folder",
            str(Path.home() / "Desktop" / "Charts")))
        out_frame = ctk.CTkFrame(parent, fg_color="transparent")
        out_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkEntry(out_frame, textvariable=self.out_var,
            fg_color=BG_INPUT, border_color="#333333",
            font=ctk.CTkFont(size=10), height=30).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(out_frame, text="📁", width=30, height=30,
            fg_color=BG_INPUT, hover_color="#333333",
            command=self._browse_out).pack(side="right", padx=(4, 0))

        # Model
        self._setting_label(parent, "Neural Model (optional)")
        self.model_var = StringVar(value=self.cfg.get("model_path", ""))
        model_frame = ctk.CTkFrame(parent, fg_color="transparent")
        model_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkEntry(model_frame, textvariable=self.model_var,
            placeholder_text="Algorithmic only",
            fg_color=BG_INPUT, border_color="#333333",
            font=ctk.CTkFont(size=10), height=30).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(model_frame, text="📁", width=30, height=30,
            fg_color=BG_INPUT, hover_color="#333333",
            command=self._browse_model).pack(side="right", padx=(4, 0))

        # Frets
        self._setting_label(parent, "Fret Palette")
        self.frets_var = StringVar(value=self.cfg.get("frets", "0,1,2,3,4"))
        fret_presets = [("GRY", "0,1,2"), ("All 5", "0,1,2,3,4")]
        self._fret_btns = {}
        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 4))
        for label, val in fret_presets:
            active = self.frets_var.get() == val
            btn = ctk.CTkButton(btn_row, text=label, height=26,
                font=ctk.CTkFont(size=11),
                fg_color=ACCENT if active else "#333333",
                hover_color="#555555",
                command=lambda v=val, l=label: self._set_frets(v, l))
            btn.pack(side="left", padx=(0, 4))
            self._fret_btns[label] = btn
        ctk.CTkEntry(parent, textvariable=self.frets_var,
            fg_color=BG_INPUT, border_color="#333333",
            font=ctk.CTkFont(size=11), height=28).pack(fill="x", pady=(0, 12))

        # Options
        self._setting_label(parent, "Options")
        self.lyrics_var = BooleanVar(value=self.cfg.get("lyrics", False))
        ctk.CTkSwitch(parent, text="Whisper Lyrics (slow)",
            variable=self.lyrics_var,
            font=ctk.CTkFont(size=12),
            progress_color=ACCENT).pack(anchor="w", pady=(0, 4))

    def _build_bottom(self):
        bottom = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=85)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        inner = ctk.CTkFrame(bottom, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        self.progress = ctk.CTkProgressBar(inner, width=420, height=8,
            progress_color=ACCENT, fg_color="#333333")
        self.progress.pack(pady=(0, 8))
        self.progress.set(0)

        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack()

        self.gen_btn = ctk.CTkButton(btn_row, text="⚡  Generate Charts",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=42, width=220,
            fg_color=ACCENT, hover_color="#17a349",
            text_color="black",
            command=self._generate)
        self.gen_btn.pack(side="left", padx=(0, 10))

        ctk.CTkButton(btn_row, text="📂  Open Output",
            font=ctk.CTkFont(size=13), height=42, width=140,
            fg_color="#333333", hover_color="#444444",
            command=self._open_output).pack(side="left")

        self.status_label = ctk.CTkLabel(inner, text="Ready",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM)
        self.status_label.pack(pady=(6, 0))

    # ── Settings Window ───────────────────────────────────────────────────────

    def _open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("Appearance Settings")
        win.geometry("400x460")
        win.configure(fg_color=BG_DARK)
        win.grab_set()

        ctk.CTkLabel(win, text="🎨  Appearance",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT).pack(pady=(20, 15))

        # Theme selector
        ctk.CTkLabel(win, text="Theme", font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM).pack(anchor="w", padx=20)

        theme_frame = ctk.CTkFrame(win, fg_color="transparent")
        theme_frame.pack(fill="x", padx=20, pady=(4, 16))

        for name, t in THEMES.items():
            row = ctk.CTkFrame(theme_frame, fg_color=BG_CARD, corner_radius=8)
            row.pack(fill="x", pady=3)

            ctk.CTkLabel(row, text="●", text_color=t["accent"],
                font=ctk.CTkFont(size=16)).pack(side="left", padx=(10, 6), pady=8)
            ctk.CTkLabel(row, text=name,
                font=ctk.CTkFont(size=13, weight="bold" if name == self._current_theme else "normal")
            ).pack(side="left", pady=8)

            if name == self._current_theme:
                ctk.CTkLabel(row, text="✓ Active",
                    text_color=ACCENT, font=ctk.CTkFont(size=11)).pack(side="right", padx=10)
            else:
                ctk.CTkButton(row, text="Apply", width=60, height=26,
                    font=ctk.CTkFont(size=11),
                    fg_color="#333333", hover_color=t["accent"],
                    command=lambda n=name, w=win: (self._apply_theme(n), w.destroy())
                ).pack(side="right", padx=8)

        # Background image
        ctk.CTkLabel(win, text="Background Image",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM).pack(anchor="w", padx=20, pady=(0, 4))

        bg_frame = ctk.CTkFrame(win, fg_color=BG_CARD, corner_radius=8)
        bg_frame.pack(fill="x", padx=20, pady=(0, 10))

        current_bg = self.cfg.get("bg_image", "")
        bg_text = Path(current_bg).name if current_bg else "No image set"
        ctk.CTkLabel(bg_frame, text=f"📷  {bg_text}",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM if not current_bg else "white"
        ).pack(side="left", padx=12, pady=10)

        btn_f = ctk.CTkFrame(bg_frame, fg_color="transparent")
        btn_f.pack(side="right", padx=8)
        ctk.CTkButton(btn_f, text="Choose", width=65, height=28,
            fg_color=ACCENT, hover_color="#17a349", text_color="black",
            command=lambda w=win: (self._browse_bg(), w.destroy())).pack(side="left", padx=2)
        ctk.CTkButton(btn_f, text="Clear", width=50, height=28,
            fg_color="#333333", hover_color="#cc3333",
            command=lambda w=win: (self._clear_bg(), w.destroy())).pack(side="left", padx=2)

        ctk.CTkLabel(win, text="Images are auto-blurred and darkened for readability",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM).pack(pady=(0, 10))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _setting_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM).pack(anchor="w", pady=(0, 3))

    def _set_frets(self, val, label):
        self.frets_var.set(val)
        self.cfg["frets"] = val
        save_config(self.cfg)
        for l, btn in self._fret_btns.items():
            btn.configure(fg_color=ACCENT if l == label else "#333333")

    def _browse_songs(self):
        files = filedialog.askopenfilenames(title="Select audio files",
            filetypes=[("Audio", "*.mp3 *.ogg *.wav *.flac"), ("All", "*.*")])
        for f in files:
            if f not in self.songs:
                self.songs.append(f)
        self._refresh_song_list()

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)
            self.cfg["out_folder"] = d
            save_config(self.cfg)

    def _browse_model(self):
        f = filedialog.askopenfilename(title="Select ChartNet checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if f:
            self.model_var.set(f)
            self.cfg["model_path"] = f
            save_config(self.cfg)

    def _clear_songs(self):
        self.songs.clear()
        self._refresh_song_list()

    def _refresh_song_list(self):
        for w in self.song_list_frame.winfo_children():
            w.destroy()
        if not self.songs:
            ctk.CTkLabel(self.song_list_frame, text="No songs added yet",
                text_color=TEXT_DIM, font=ctk.CTkFont(size=12)).pack(pady=20)
            return
        for i, path in enumerate(self.songs):
            row = ctk.CTkFrame(self.song_list_frame, fg_color="#222222", corner_radius=6)
            row.pack(fill="x", pady=2, padx=4)
            ctk.CTkLabel(row, text=f"🎵  {Path(path).name}",
                font=ctk.CTkFont(size=12), anchor="w"
            ).pack(side="left", padx=10, pady=6, fill="x", expand=True)
            ctk.CTkButton(row, text="✕", width=26, height=26,
                fg_color="transparent", hover_color="#cc3333",
                font=ctk.CTkFont(size=12),
                command=lambda i=i: self._remove_song(i)).pack(side="right", padx=6)

    def _remove_song(self, idx):
        if 0 <= idx < len(self.songs):
            self.songs.pop(idx)
            self._refresh_song_list()

    def _open_output(self):
        out = self.out_var.get()
        if os.path.exists(out):
            os.startfile(out)

    def _log(self, msg):
        self.log_queue.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self):
        if self.running:
            return
        if not self.songs:
            self._log("⚠  No songs added. Click the drop zone to add MP3 files.")
            return
        self.running = True
        self.gen_btn.configure(state="disabled", text="⏳  Generating...")
        self.progress.set(0)
        self.status_label.configure(text="Starting...")
        threading.Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        songs  = list(self.songs)
        total  = len(songs)
        out    = self.out_var.get()
        model  = self.model_var.get()
        frets  = self.frets_var.get()
        lyrics = self.lyrics_var.get()

        os.makedirs(out, exist_ok=True)

        for i, song in enumerate(songs):
            name = Path(song).name
            self._log(f"\n{'─'*38}")
            self._log(f"[{i+1}/{total}]  {name}")
            self.after(0, lambda i=i, name=name: self.status_label.configure(
                text=f"Processing {i+1}/{total}: {name}"))

            cmd = [sys.executable, "chartgen.py", "--out", out, "--frets", frets]
            if not lyrics:
                cmd.append("--no-lyrics")
            if model:
                cmd += ["--model", model]
            cmd.append(song)

            try:
                proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(f"  {line}")
                proc.wait()
                self._log("✅  Done!" if proc.returncode == 0 else f"❌  Failed (exit {proc.returncode})")
            except Exception as e:
                self._log(f"❌  Error: {e}")

            self.after(0, lambda v=(i+1)/total: self.progress.set(v))

        self._log(f"\n🎸  All done! Output → {out}")
        self.after(0, self._on_done)

    def _on_done(self):
        self.running = False
        self.gen_btn.configure(state="normal", text="⚡  Generate Charts")
        self.status_label.configure(text=f"Done! {len(self.songs)} chart(s) generated")
        self.progress.set(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ChartGenApp()
    app.mainloop()
