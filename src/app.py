"""
Clone Hero Chart Generator Suite — Full GUI App
Tabs: Generate · Scrape · Preprocess · Train
"""

import sys, os, json, threading, subprocess, queue, shutil, tempfile, winreg
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, StringVar, BooleanVar, IntVar
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter

# ── Find Python executable (robust — works even when PATH is stripped) ─────────
def _find_python():
    # 1. Already known (running as .py)
    if not getattr(sys, 'frozen', False):
        return sys.executable
    # 2. PATH lookup
    found = shutil.which('python') or shutil.which('python3')
    if found:
        return found
    # 3. Windows registry — HKCU and HKLM Python installs
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for base in (r'SOFTWARE\Python\PythonCore', r'SOFTWARE\WOW6432Node\Python\PythonCore'):
            try:
                with winreg.OpenKey(hive, base) as core:
                    for i in range(winreg.QueryInfoKey(core)[0]):
                        ver = winreg.EnumKey(core, i)
                        try:
                            with winreg.OpenKey(core, rf'{ver}\InstallPath') as ip:
                                path = winreg.QueryValueEx(ip, 'ExecutablePath')[0]
                                if path and Path(path).exists():
                                    return path
                        except OSError:
                            pass
            except OSError:
                pass
    # 4. Common install locations
    for candidate in [
        r'C:\Python310\python.exe', r'C:\Python311\python.exe', r'C:\Python312\python.exe',
        Path.home() / 'AppData/Local/Programs/Python/Python310/python.exe',
        Path.home() / 'AppData/Local/Programs/Python/Python311/python.exe',
        Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe',
    ]:
        if Path(candidate).exists():
            return str(candidate)
    return 'python'  # last resort

# ── Path resolution (works as .py, PyInstaller .exe, and Nuitka .exe) ─────────
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

PYTHON_EXE = _find_python()

def script(name):
    return str(APP_DIR / name)

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#1DB954"
BG_DARK  = "#0d0d0d"
BG_CARD  = "#161616"
BG_INPUT = "#1f1f1f"
BG_SIDE  = "#111111"
TEXT_DIM = "#666666"
TEXT_MED = "#999999"

THEMES = {
    "Green (Default)": {"accent": "#1DB954", "bg": "#0d0d0d", "card": "#1a1a1a", "side": "#151515"},
    "Blue Steel":      {"accent": "#3b82f6", "bg": "#0a0e1a", "card": "#141c2e", "side": "#0f1520"},
    "Purple Haze":     {"accent": "#a855f7", "bg": "#0d0a1a", "card": "#1e1630", "side": "#181226"},
    "Fire Red":        {"accent": "#ef4444", "bg": "#0f0808", "card": "#1e1212", "side": "#170e0e"},
    "Gold Rush":       {"accent": "#f59e0b", "bg": "#0f0e08", "card": "#1e1c10", "side": "#17150a"},
    "Ice White":       {"accent": "#e2e8f0", "bg": "#0f0f0f", "card": "#1a1a1a", "side": "#161616"},
}

_appdata = Path(os.environ.get("APPDATA", Path.home()))
CONFIG_DIR  = _appdata / "ChartGen"
CONFIG_PATH = CONFIG_DIR / "config.json"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def load_cfg():
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}

def save_cfg(c):
    try: CONFIG_PATH.write_text(json.dumps(c, indent=2))
    except Exception: pass

# ── Reusable widgets ──────────────────────────────────────────────────────────

def label(parent, text, size=12, bold=False, color=None, **kw):
    return ctk.CTkLabel(parent, text=text,
        font=ctk.CTkFont(size=size, weight="bold" if bold else "normal"),
        text_color=color or "white", **kw)

def dim_label(parent, text, size=11, **kw):
    return label(parent, text, size=size, color=TEXT_MED, **kw)

def card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, **kw)

def btn(parent, text, command, width=120, height=34, accent=False, danger=False, **kw):
    fg = ACCENT if accent else ("#cc3333" if danger else "#2a2a2a")
    hv = "#17a349" if accent else ("#aa2222" if danger else "#3a3a3a")
    tc = "black" if accent else "white"
    return ctk.CTkButton(parent, text=text, command=command,
        width=width, height=height,
        fg_color=fg, hover_color=hv, text_color=tc,
        font=ctk.CTkFont(size=12), **kw)

def entry(parent, textvariable=None, placeholder="", width=None, **kw):
    e = ctk.CTkEntry(parent, textvariable=textvariable,
        placeholder_text=placeholder,
        fg_color=BG_INPUT, border_color="#2a2a2a",
        font=ctk.CTkFont(size=11), **kw)
    if width: e.configure(width=width)
    return e

class LogBox(ctk.CTkTextbox):
    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#cccccc", wrap="word", **kw)
        self.configure(state="disabled")
        self._q = queue.Queue()
        self._poll()

    def log(self, msg):
        self._q.put(msg)

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                self.configure(state="normal")
                self.insert("end", msg + "\n")
                self.see("end")
                self.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll)


def run_script(args, logbox, on_done=None, on_start=None):
    """Run a python script in a background thread, streaming output to logbox."""
    def _run():
        if on_start: on_start()
        cmd = [PYTHON_EXE] + args
        logbox.log("$ " + " ".join(str(a) for a in args))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line: logbox.log(line)
            proc.wait()
            logbox.log(f"\n{'✅ Done' if proc.returncode == 0 else '❌ Failed (exit ' + str(proc.returncode) + ')'}")
        except Exception as e:
            logbox.log(f"❌ Error: {e}")
        if on_done: on_done()
    threading.Thread(target=_run, daemon=True).start()

# ── Main App ──────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.cfg = load_cfg()
        self._apply_theme(self.cfg.get("theme", "Green (Default)"), init=True)
        self.title("Clone Hero Chart Generator")
        self.geometry("980x700")
        self.minsize(820, 580)
        self.configure(fg_color=BG_DARK)
        self._bg_ref = None
        self._bg_lbl = None
        self._active_tab = None
        self._build()
        self._load_bg()
        self.bind("<Configure>", lambda e: self._resize_bg() if e.widget == self else None)

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self, name, init=False):
        global ACCENT, BG_DARK, BG_CARD, BG_INPUT, BG_SIDE
        t = THEMES.get(name, THEMES["Green (Default)"])
        ACCENT = t["accent"]; BG_DARK = t["bg"]
        BG_CARD = t["card"]; BG_SIDE = t["side"]
        self.cfg["theme"] = name
        save_cfg(self.cfg)
        if not init:
            self.configure(fg_color=BG_DARK)
            for w in self.winfo_children(): w.destroy()
            self._bg_lbl = None
            self._build()
            self._load_bg()

    # ── Background ────────────────────────────────────────────────────────────
    def _load_bg(self):
        self._apply_bg(self.cfg.get("bg_image", ""))

    def _apply_bg(self, path):
        if not path or not os.path.exists(path):
            if self._bg_lbl: self._bg_lbl.destroy(); self._bg_lbl = None
            return
        try:
            w, h = max(self.winfo_width(), 980), max(self.winfo_height(), 700)
            img = Image.open(path).resize((w, h), Image.LANCZOS)
            img = img.filter(ImageFilter.GaussianBlur(10))
            ov  = Image.new("RGBA", img.size, (0,0,0,170))
            img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
            self._bg_ref = ImageTk.PhotoImage(img)
            if not self._bg_lbl:
                self._bg_lbl = tk.Label(self, image=self._bg_ref)
                self._bg_lbl.place(x=0, y=0, relwidth=1, relheight=1)
                self._bg_lbl.lower()
            else:
                self._bg_lbl.configure(image=self._bg_ref)
        except Exception: pass

    def _resize_bg(self):
        if self.cfg.get("bg_image"):
            self.after(200, lambda: self._apply_bg(self.cfg.get("bg_image","")))

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, fg_color=BG_SIDE, corner_radius=0, width=200,
            border_width=1, border_color="#2a2a2a")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=70)
        logo.pack(fill="x")
        logo.pack_propagate(False)
        label(logo, "🎸", size=24).pack(side="left", padx=(16,6), pady=18)
        f = ctk.CTkFrame(logo, fg_color="transparent")
        f.pack(side="left", pady=18)
        label(f, "ChartGen", size=15, bold=True, color=ACCENT).pack(anchor="w")
        dim_label(f, "Clone Hero Suite").pack(anchor="w")

        ctk.CTkFrame(self.sidebar, fg_color="#222222", height=1).pack(fill="x", padx=12)

        # Nav buttons
        self._nav_btns = {}
        nav_items = [
            ("🎸  Generate",    "generate"),
            ("📥  Scrape",      "scrape"),
            ("⚙️  Preprocess",  "preprocess"),
            ("🧠  Train",       "train"),
        ]
        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(fill="x", pady=12, padx=8)

        for text, key in nav_items:
            b = ctk.CTkButton(nav_frame, text=text,
                font=ctk.CTkFont(size=13),
                anchor="w", height=42, corner_radius=8,
                fg_color="transparent", hover_color="#222222",
                text_color=TEXT_MED,
                command=lambda k=key: self._switch_tab(k))
            b.pack(fill="x", pady=2)
            self._nav_btns[key] = b

        # Bottom settings button
        ctk.CTkFrame(self.sidebar, fg_color="#222222", height=1).pack(fill="x", padx=12, side="bottom", pady=(0,60))
        ctk.CTkButton(self.sidebar, text="⚙  Appearance",
            font=ctk.CTkFont(size=12), anchor="w", height=36,
            fg_color="transparent", hover_color="#222222", text_color=TEXT_MED,
            corner_radius=8, command=self._open_appearance
        ).pack(side="bottom", fill="x", padx=8, pady=(0,8))

        # ── Content area ──────────────────────────────────────────────────────
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True)

        # Build all tabs
        self._tabs = {
            "generate":   GenerateTab(self.content, self.cfg),
            "scrape":     ScrapeTab(self.content, self.cfg),
            "preprocess": PreprocessTab(self.content, self.cfg),
            "train":      TrainTab(self.content, self.cfg),
        }
        self._switch_tab("generate")

    def _switch_tab(self, key):
        if self._active_tab:
            self._tabs[self._active_tab].pack_forget()
            self._nav_btns[self._active_tab].configure(
                fg_color="transparent", text_color=TEXT_MED)
        self._active_tab = key
        self._tabs[key].pack(fill="both", expand=True)
        self._nav_btns[key].configure(fg_color="#222222", text_color=ACCENT)

    # ── Appearance window ─────────────────────────────────────────────────────
    def _open_appearance(self):
        win = ctk.CTkToplevel(self)
        win.title("Appearance")
        win.geometry("420x500")
        win.configure(fg_color=BG_DARK)
        win.grab_set()

        label(win, "🎨  Appearance", size=18, bold=True, color=ACCENT).pack(pady=(20,15), padx=20, anchor="w")

        # Themes
        dim_label(win, "COLOR THEME").pack(anchor="w", padx=20)
        tf = ctk.CTkScrollableFrame(win, fg_color="transparent", height=230)
        tf.pack(fill="x", padx=20, pady=(4,16))

        for name, t in THEMES.items():
            row = card(tf, height=42)
            row.pack(fill="x", pady=3)
            row.pack_propagate(False)
            ctk.CTkLabel(row, text="●", text_color=t["accent"],
                font=ctk.CTkFont(size=20)).pack(side="left", padx=(12,8))
            label(row, name, size=13,
                bold=(name==self.cfg.get("theme"))).pack(side="left")
            if name == self.cfg.get("theme","Green (Default)"):
                label(row, "✓", size=13, color=ACCENT).pack(side="right", padx=12)
            else:
                btn(row, "Apply", width=60, height=28, accent=False,
                    command=lambda n=name, w=win: (self._apply_theme(n), w.destroy())
                ).pack(side="right", padx=8)

        # Background image
        dim_label(win, "BACKGROUND IMAGE").pack(anchor="w", padx=20)
        bg_card = card(win, height=50)
        bg_card.pack(fill="x", padx=20, pady=(4,4))
        bg_card.pack_propagate(False)
        cur = self.cfg.get("bg_image","")
        label(bg_card, f"📷  {Path(cur).name if cur else 'No image'}",
            size=11, color=TEXT_MED if not cur else "white").pack(side="left", padx=12)
        bf = ctk.CTkFrame(bg_card, fg_color="transparent")
        bf.pack(side="right", padx=8)
        btn(bf, "Choose", width=65, height=28, accent=True,
            command=lambda w=win: self._pick_bg(w)).pack(side="left", padx=2)
        btn(bf, "Clear", width=50, height=28, danger=True,
            command=lambda w=win: self._clear_bg(w)).pack(side="left", padx=2)
        dim_label(win, "Auto-blurred and darkened for readability").pack(padx=20, pady=(2,0))

    def _pick_bg(self, win):
        f = filedialog.askopenfilename(title="Background image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp"),("All","*.*")])
        if f:
            self.cfg["bg_image"] = f
            save_cfg(self.cfg)
            self._apply_bg(f)
            win.destroy()

    def _clear_bg(self, win):
        self.cfg["bg_image"] = ""
        save_cfg(self.cfg)
        if self._bg_lbl: self._bg_lbl.destroy(); self._bg_lbl = None
        win.destroy()


# ── Base Tab ──────────────────────────────────────────────────────────────────

class BaseTab(ctk.CTkFrame):
    def __init__(self, parent, cfg, title, subtitle, icon):
        super().__init__(parent, fg_color="transparent")
        self.cfg = cfg
        self._running = False

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent", height=64)
        hdr.pack(fill="x", padx=24, pady=(18,0))
        hdr.pack_propagate(False)
        label(hdr, f"{icon}  {title}", size=22, bold=True, color=ACCENT).pack(side="left", anchor="s", pady=8)
        dim_label(hdr, subtitle).pack(side="left", anchor="s", padx=10, pady=10)

        ctk.CTkFrame(self, fg_color="#222222", height=1).pack(fill="x", padx=24, pady=(8,16))

    def _lock(self):   self._running = True
    def _unlock(self): self._running = False


# ── GENERATE TAB ──────────────────────────────────────────────────────────────

class GenerateTab(BaseTab):
    def __init__(self, parent, cfg):
        super().__init__(parent, cfg, "Generate Charts", "MP3 → Clone Hero .chart files", "🎸")

        # Bottom bar packed FIRST so expand=True on body doesn't eat its space
        self._build_bottom()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(0,8))

        # ── Left column ──────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0,14))

        # Drop zone
        drop = card(left, height=90, border_width=2, border_color="#2a2a2a")
        drop.pack(fill="x", pady=(0,10))
        drop.pack_propagate(False)
        drop.bind("<Button-1>", lambda e: self._browse())
        drop.bind("<Enter>",    lambda e: drop.configure(border_color=ACCENT))
        drop.bind("<Leave>",    lambda e: drop.configure(border_color="#2a2a2a"))
        di = ctk.CTkFrame(drop, fg_color="transparent")
        di.place(relx=.5, rely=.5, anchor="center")
        label(di, "🎵  Click to add MP3 / audio files", size=13, bold=True).pack()
        dim_label(di, "Supports batch — multiple files at once").pack(pady=(2,0))
        for w in di.winfo_children(): w.bind("<Button-1>", lambda e: self._browse())

        # Songs header
        sh = ctk.CTkFrame(left, fg_color="transparent")
        sh.pack(fill="x", pady=(0,4))
        label(sh, "Songs", size=12, bold=True).pack(side="left")
        btn(sh, "Clear all", self._clear, width=70, height=24).pack(side="right")

        self.song_list = ctk.CTkScrollableFrame(left, fg_color=BG_CARD, corner_radius=8, height=120)
        self.song_list.pack(fill="x", pady=(0,10))
        self._songs = []
        self._refresh_list()

        # Log
        label(left, "Output Log", size=12, bold=True).pack(anchor="w", pady=(0,4))
        self.log = LogBox(left)
        self.log.pack(fill="both", expand=True)

        # ── Right column — Settings ───────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color="transparent", width=230)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        label(right, "Settings", size=13, bold=True).pack(anchor="w", pady=(0,10))

        self._out_var   = self._folder_row(right, "Output Folder",
            cfg.get("gen_out", str(Path.home()/"Desktop"/"Charts")), self._browse_out)
        self._model_var = self._file_row(right, "Neural Model (optional)",
            cfg.get("model_path",""), "*.pt", self._browse_model)
        self._frets_var, self._fret_btns = self._fret_row(right, cfg.get("frets","0,1,2,3,4"))

        dim_label(right, "OPTIONS").pack(anchor="w", pady=(6,4))
        self._lyrics_var = BooleanVar(value=cfg.get("lyrics", False))
        ctk.CTkSwitch(right, text="Whisper Lyrics (slow)",
            variable=self._lyrics_var,
            font=ctk.CTkFont(size=12),
            progress_color=ACCENT).pack(anchor="w")

    def _build_bottom(self):
        bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=76)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.place(relx=.5, rely=.5, anchor="center")
        self.progress = ctk.CTkProgressBar(inner, width=460, height=6,
            progress_color=ACCENT, fg_color="#2a2a2a")
        self.progress.pack(pady=(0,8))
        self.progress.set(0)
        row = ctk.CTkFrame(inner, fg_color="transparent")
        row.pack()
        self.gen_btn = btn(row, "⚡  Generate Charts", self._generate,
            width=200, height=40, accent=True)
        self.gen_btn.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.gen_btn.pack(side="left", padx=(0,10))
        btn(row, "📂  Open Output", self._open_out, width=140, height=40).pack(side="left")
        self.status_lbl = dim_label(inner, "Ready")
        self.status_lbl.pack(pady=(6,0))

    def _folder_row(self, parent, title, default, browse_cmd):
        dim_label(parent, title.upper()).pack(anchor="w", pady=(0,3))
        var = StringVar(value=default)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0,10))
        entry(row, textvariable=var, height=30).pack(side="left", fill="x", expand=True)
        btn(row, "📁", browse_cmd, width=30, height=30).pack(side="right", padx=(4,0))
        return var

    def _file_row(self, parent, title, default, ext, browse_cmd):
        dim_label(parent, title.upper()).pack(anchor="w", pady=(0,3))
        var = StringVar(value=default)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0,10))
        entry(row, textvariable=var, placeholder="Not set", height=30).pack(side="left", fill="x", expand=True)
        btn(row, "📁", browse_cmd, width=30, height=30).pack(side="right", padx=(4,0))
        return var

    def _fret_row(self, parent, default):
        dim_label(parent, "FRET PALETTE").pack(anchor="w", pady=(0,3))
        var  = StringVar(value=default)
        btns = {}
        row  = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0,4))
        for lbl, val in [("GRY","0,1,2"),("All 5","0,1,2,3,4")]:
            active = var.get() == val
            b = ctk.CTkButton(row, text=lbl, height=28,
                font=ctk.CTkFont(size=11),
                fg_color=ACCENT if active else "#2a2a2a",
                hover_color="#3a3a3a",
                command=lambda v=val, l=lbl: self._set_frets(v, l, var, btns))
            b.pack(side="left", padx=(0,4))
            btns[lbl] = b
        entry(parent, textvariable=var, height=28).pack(fill="x", pady=(0,10))
        return var, btns

    def _set_frets(self, val, lbl, var, btns):
        var.set(val)
        self.cfg["frets"] = val
        save_cfg(self.cfg)
        for l, b in btns.items():
            b.configure(fg_color=ACCENT if l==lbl else "#2a2a2a")

    def _refresh_list(self):
        for w in self.song_list.winfo_children(): w.destroy()
        if not self._songs:
            dim_label(self.song_list, "No songs added yet").pack(pady=16)
            return
        for i, p in enumerate(self._songs):
            row = ctk.CTkFrame(self.song_list, fg_color="#1e1e1e", corner_radius=6)
            row.pack(fill="x", pady=2, padx=4)
            label(row, f"🎵  {Path(p).name}", size=12, anchor="w"
                ).pack(side="left", padx=10, pady=6, fill="x", expand=True)
            btn(row, "✕", lambda i=i: self._remove(i),
                width=26, height=26, danger=True).pack(side="right", padx=6)

    def _browse(self):
        files = filedialog.askopenfilenames(title="Select audio files",
            filetypes=[("Audio","*.mp3 *.ogg *.wav *.flac"),("All","*.*")])
        for f in files:
            if f not in self._songs: self._songs.append(f)
        self._refresh_list()

    def _remove(self, i):
        if 0 <= i < len(self._songs): self._songs.pop(i); self._refresh_list()

    def _clear(self): self._songs.clear(); self._refresh_list()

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d: self._out_var.set(d); self.cfg["gen_out"]=d; save_cfg(self.cfg)

    def _browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("Checkpoint","*.pt"),("All","*.*")])
        if f: self._model_var.set(f); self.cfg["model_path"]=f; save_cfg(self.cfg)

    def _open_out(self):
        p = self._out_var.get()
        if os.path.exists(p): os.startfile(p)

    def _generate(self):
        if self._running: return
        if not self._songs: self.log.log("⚠  No songs added."); return
        self._lock()
        self.gen_btn.configure(state="disabled", text="⏳  Generating...")
        self.progress.set(0)
        songs = list(self._songs)

        def run():
            for i, song in enumerate(songs):
                self.status_lbl.configure(text=f"Processing {i+1}/{len(songs)}: {Path(song).name}")
                cmd = [script("chartgen.py"), "--out", self._out_var.get(),
                       "--frets", self._frets_var.get()]
                if not self._lyrics_var.get(): cmd.append("--no-lyrics")
                if self._model_var.get(): cmd += ["--model", self._model_var.get()]
                cmd.append(song)
                self.log.log(f"\n[{i+1}/{len(songs)}] {Path(song).name}")
                try:
                    proc = subprocess.Popen([PYTHON_EXE]+[script("chartgen.py")]+cmd[1:],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in proc.stdout:
                        if line.strip(): self.log.log("  " + line.rstrip())
                    proc.wait()
                    self.log.log("✅ Done" if proc.returncode==0 else f"❌ Failed")
                except Exception as e:
                    self.log.log(f"❌ {e}")
                self.progress.set((i+1)/len(songs))
            self.gen_btn.configure(state="normal", text="⚡  Generate Charts")
            self.status_lbl.configure(text=f"Done! {len(songs)} chart(s)")
            self._unlock()

        threading.Thread(target=run, daemon=True).start()


# ── SCRAPE TAB ────────────────────────────────────────────────────────────────

class ScrapeTab(BaseTab):
    def __init__(self, parent, cfg):
        super().__init__(parent, cfg, "Scrape Dataset", "Download chart+audio pairs from Enchor", "📥")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24)

        left  = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0,16))
        right = ctk.CTkFrame(body, fg_color="transparent", width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        label(left, "Output Log", size=13, bold=True).pack(anchor="w", pady=(0,6))
        self.log = LogBox(left)
        self.log.pack(fill="both", expand=True)

        # Right panel settings
        label(right, "Scrape Settings", size=14, bold=True).pack(anchor="w", pady=(0,12))

        dim_label(right, "OUTPUT FOLDER").pack(anchor="w", pady=(0,3))
        self._out_var = StringVar(value=cfg.get("scrape_out", "./dataset"))
        row = ctk.CTkFrame(right, fg_color="transparent")
        row.pack(fill="x", pady=(0,12))
        entry(row, textvariable=self._out_var, height=30).pack(side="left", fill="x", expand=True)
        btn(row, "📁", self._browse_out, width=30, height=30).pack(side="right", padx=(4,0))

        dim_label(right, "GENRE QUERIES").pack(anchor="w", pady=(0,6))
        self._genres = {}
        genres = [("Metal", "metal", True), ("Rock", "rock", True),
                  ("Punk", "punk", True), ("Pop", "pop", True),
                  ("Alternative", "alternative", False),
                  ("Electronic", "electronic", False),
                  ("General", "", True)]
        gc = card(right)
        gc.pack(fill="x", pady=(0,12))
        for name, val, default in genres:
            var = BooleanVar(value=cfg.get(f"genre_{val}", default))
            self._genres[val] = var
            ctk.CTkCheckBox(gc, text=name, variable=var,
                font=ctk.CTkFont(size=12), checkmark_color="black",
                fg_color=ACCENT).pack(anchor="w", padx=12, pady=4)

        dim_label(right, "SONGS PER GENRE").pack(anchor="w", pady=(0,3))
        self._limit_var = StringVar(value=str(cfg.get("scrape_limit", 2000)))
        entry(right, textvariable=self._limit_var, height=30).pack(fill="x", pady=(0,12))

        dim_label(right, "WORKERS (parallel downloads)").pack(anchor="w", pady=(0,3))
        self._workers_var = StringVar(value=str(cfg.get("scrape_workers", 16)))
        entry(right, textvariable=self._workers_var, height=30).pack(fill="x", pady=(0,16))

        self._scrape_btn = btn(right, "📥  Start Scraping", self._scrape,
            width=240, height=40, accent=True)
        self._scrape_btn.configure(font=ctk.CTkFont(size=13, weight="bold"))
        self._scrape_btn.pack(fill="x")

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d: self._out_var.set(d); self.cfg["scrape_out"]=d; save_cfg(self.cfg)

    def _scrape(self):
        if self._running: return
        genres = [v for v, var in self._genres.items() if var.get()]
        if not genres: self.log.log("⚠  Select at least one genre."); return
        out     = self._out_var.get()
        limit   = self._limit_var.get()
        workers = self._workers_var.get()
        self.cfg.update({"scrape_out":out,"scrape_limit":int(limit),"scrape_workers":int(workers)})
        save_cfg(self.cfg)
        self.log.clear()
        self.log.log(f"Starting scrape: {len(genres)} genre(s) × {limit} songs = up to {len(genres)*int(limit)} total\n")

        def run():
            self._lock()
            self._scrape_btn.configure(state="disabled", text="⏳  Scraping...")
            for g in genres:
                q = g if g else ""
                label_str = g if g else "general"
                self.log.log(f"\n{'─'*36}")
                self.log.log(f"▶  Genre: {label_str} (limit {limit})")
                args = [script("scraper.py"), "--out", out, "--limit", limit, "--workers", workers]
                if q: args += ["--query", q]
                try:
                    proc = subprocess.Popen([PYTHON_EXE]+args,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in proc.stdout:
                        if line.strip(): self.log.log(line.rstrip())
                    proc.wait()
                except Exception as e:
                    self.log.log(f"❌ {e}")
            self.log.log("\n🎉  All genres scraped!")
            self._scrape_btn.configure(state="normal", text="📥  Start Scraping")
            self._unlock()

        threading.Thread(target=run, daemon=True).start()


# ── PREPROCESS TAB ────────────────────────────────────────────────────────────

class PreprocessTab(BaseTab):
    def __init__(self, parent, cfg):
        super().__init__(parent, cfg, "Preprocess", "Convert songs to mel spectrograms for training", "⚙️")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24)

        left  = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0,16))
        right = ctk.CTkFrame(body, fg_color="transparent", width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        label(left, "Output Log", size=13, bold=True).pack(anchor="w", pady=(0,6))
        self.log = LogBox(left)
        self.log.pack(fill="both", expand=True)

        label(right, "Preprocess Settings", size=14, bold=True).pack(anchor="w", pady=(0,12))

        dim_label(right, "DATASET FOLDER (from scraper)").pack(anchor="w", pady=(0,3))
        self._data_var = StringVar(value=cfg.get("scrape_out","./dataset"))
        self._folder_row(right, self._data_var, self._browse_data)

        dim_label(right, "OUTPUT FOLDER (processed .pt files)").pack(anchor="w", pady=(0,3))
        self._out_var = StringVar(value=cfg.get("preprocess_out","./processed"))
        self._folder_row(right, self._out_var, self._browse_out)

        # Info card
        ic = card(right)
        ic.pack(fill="x", pady=(0,16))
        for line in ["• Skips already-processed songs","• ~1-2 hrs for 5k songs","• Creates .pt tensor files"]:
            dim_label(ic, line).pack(anchor="w", padx=12, pady=3)

        self._btn = btn(right, "⚙️  Start Preprocessing", self._run,
            width=240, height=40, accent=True)
        self._btn.configure(font=ctk.CTkFont(size=13, weight="bold"))
        self._btn.pack(fill="x")

    def _folder_row(self, parent, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0,12))
        entry(row, textvariable=var, height=30).pack(side="left", fill="x", expand=True)
        btn(row, "📁", cmd, width=30, height=30).pack(side="right", padx=(4,0))

    def _browse_data(self):
        d = filedialog.askdirectory()
        if d: self._data_var.set(d)

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d: self._out_var.set(d); self.cfg["preprocess_out"]=d; save_cfg(self.cfg)

    def _run(self):
        if self._running: return
        self.log.clear()
        self._btn.configure(state="disabled", text="⏳  Processing...")
        run_script([script("preprocess_dataset.py"),
                    "--data", self._data_var.get(),
                    "--out",  self._out_var.get()],
            self.log,
            on_done=lambda: self._btn.configure(state="normal", text="⚙️  Start Preprocessing"))
        self._lock()


# ── TRAIN TAB ─────────────────────────────────────────────────────────────────

class TrainTab(BaseTab):
    def __init__(self, parent, cfg):
        super().__init__(parent, cfg, "Train Model", "Train ChartNet v3 on your dataset", "🧠")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24)

        left  = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0,16))
        right = ctk.CTkFrame(body, fg_color="transparent", width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        label(left, "Training Log", size=13, bold=True).pack(anchor="w", pady=(0,6))
        self.log = LogBox(left)
        self.log.pack(fill="both", expand=True)

        label(right, "Train Settings", size=14, bold=True).pack(anchor="w", pady=(0,12))

        # Paths
        dim_label(right, "PROCESSED DATA FOLDER").pack(anchor="w", pady=(0,3))
        self._data_var = StringVar(value=cfg.get("preprocess_out","./processed"))
        self._folder_row(right, self._data_var, lambda: self._browse(self._data_var))

        dim_label(right, "CHECKPOINT OUTPUT FOLDER").pack(anchor="w", pady=(0,3))
        self._ckpt_var = StringVar(value=cfg.get("train_ckpt","./checkpoints"))
        self._folder_row(right, self._ckpt_var, lambda: self._browse(self._ckpt_var))

        dim_label(right, "RESUME FROM CHECKPOINT (optional)").pack(anchor="w", pady=(0,3))
        self._resume_var = StringVar(value="")
        rrow = ctk.CTkFrame(right, fg_color="transparent")
        rrow.pack(fill="x", pady=(0,12))
        entry(rrow, textvariable=self._resume_var, placeholder="Leave empty for fresh start", height=30).pack(side="left", fill="x", expand=True)
        btn(rrow, "📁", lambda: self._browse_file(self._resume_var), width=30, height=30).pack(side="right", padx=(4,0))

        # Hyperparams grid
        hc = card(right)
        hc.pack(fill="x", pady=(0,12))
        params = [
            ("Epochs",       "train_epochs",  "160"),
            ("Batch Size",   "train_batch",   "600"),
            ("Workers",      "train_workers", "16"),
            ("Learning Rate","train_lr",      "2e-4"),
        ]
        self._param_vars = {}
        for pname, key, default in params:
            pr = ctk.CTkFrame(hc, fg_color="transparent")
            pr.pack(fill="x", padx=10, pady=4)
            dim_label(pr, pname).pack(side="left")
            var = StringVar(value=str(cfg.get(key, default)))
            self._param_vars[key] = var
            entry(pr, textvariable=var, width=80, height=26).pack(side="right")

        # Flags
        self._reset_best = BooleanVar(value=False)
        ctk.CTkCheckBox(right, text="Reset best F1 (new dataset)", variable=self._reset_best,
            font=ctk.CTkFont(size=12), checkmark_color="black",
            fg_color=ACCENT).pack(anchor="w", pady=(0,12))

        # Buttons
        brow = ctk.CTkFrame(right, fg_color="transparent")
        brow.pack(fill="x")
        self._train_btn = btn(brow, "🧠  Start Training", self._train,
            width=165, height=40, accent=True)
        self._train_btn.configure(font=ctk.CTkFont(size=13, weight="bold"))
        self._train_btn.pack(side="left", padx=(0,6))
        btn(brow, "⏹ Stop", self._stop, width=65, height=40, danger=True).pack(side="left")

        self._proc = None

    def _folder_row(self, parent, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0,10))
        entry(row, textvariable=var, height=30).pack(side="left", fill="x", expand=True)
        btn(row, "📁", cmd, width=30, height=30).pack(side="right", padx=(4,0))

    def _browse(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)

    def _browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Checkpoint","*.pt"),("All","*.*")])
        if f: var.set(f)

    def _train(self):
        if self._running: return
        self.log.clear()
        args = [script("train_chartnet.py"),
                "--data",    self._data_var.get(),
                "--out",     self._ckpt_var.get(),
                "--epochs",  self._param_vars["train_epochs"].get(),
                "--batch",   self._param_vars["train_batch"].get(),
                "--workers", self._param_vars["train_workers"].get(),
                "--lr",      self._param_vars["train_lr"].get()]
        if self._resume_var.get(): args += ["--resume", self._resume_var.get()]
        if self._reset_best.get(): args.append("--reset-best")

        for k in ["train_epochs","train_batch","train_workers","train_lr"]:
            self.cfg[k] = self._param_vars[k].get()
        save_cfg(self.cfg)

        self._train_btn.configure(state="disabled", text="⏳  Training...")
        self._lock()

        def run():
            self.log.log("$ " + " ".join(str(a) for a in args[1:]))
            try:
                self._proc = subprocess.Popen([PYTHON_EXE]+args,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in self._proc.stdout:
                    if line.strip(): self.log.log(line.rstrip())
                self._proc.wait()
                self.log.log("\n✅ Training complete!" if self._proc.returncode==0 else "❌ Training stopped")
            except Exception as e:
                self.log.log(f"❌ {e}")
            self._train_btn.configure(state="normal", text="🧠  Start Training")
            self._unlock()

        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        if self._proc:
            self._proc.terminate()
            self.log.log("\n⏹  Training stopped by user.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
