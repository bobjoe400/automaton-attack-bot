"""The control panel window.

Layout: menubar (tools live there), status banner, stat cards, big
Start/Stop, and a one-line last-event strip. The full activity feed is a
View-menu toggle -- most sessions never need it, and everything important
already surfaces in the banner and cards. All actual work happens in
:class:`tasks.TaskRunner`.
"""

from __future__ import annotations

import queue

from .. import __version__
from .feed import STATES, display_text, interpret
from .tasks import TaskRunner

_POLL_MS = 100


def run_gui(args) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog, ttk
        from tkinter.scrolledtext import ScrolledText
    except ImportError as exc:
        print(f"error: tkinter unavailable ({exc}); run with --cli instead")
        return 1

    from ..cli import COMMANDS, build_parser, cmd_run

    runner = TaskRunner()
    root = tk.Tk()
    root.title(f"Automaton Attack Bot v{__version__}")
    root.minsize(560, 360)

    # -- status banner -----------------------------------------------------
    banner = tk.Label(root, text="IDLE", fg="white", bg=STATES["idle"][1],
                      font=("Segoe UI", 22, "bold"), pady=10)
    banner.pack(fill="x")

    # -- stat cards --------------------------------------------------------
    cards = ttk.Frame(root, padding=(12, 10))
    cards.pack(fill="x")
    card_vars = {}
    for column, (key, caption) in enumerate(
            (("score", "SCORE"), ("combo", "COMBO"), ("typed", "WORDS TYPED"))):
        card = ttk.Frame(cards, relief="groove", borderwidth=1, padding=10)
        card.grid(row=0, column=column, sticky="nsew",
                  padx=(0 if column == 0 else 8, 0))
        cards.columnconfigure(column, weight=1)
        value = tk.Label(card, text="--", font=("Segoe UI", 18, "bold"))
        value.pack()
        ttk.Label(card, text=caption, font=("Segoe UI", 8)).pack()
        card_vars[key] = value

    # -- play controls: the two buttons ARE the interface ------------------
    play = ttk.Frame(root, padding=(12, 4))
    play.pack(fill="x")
    start_btn = tk.Button(play, text="▶  START", fg="white",
                          bg="#1f8a3b", activebackground="#26a147",
                          activeforeground="white", relief="flat",
                          font=("Segoe UI", 15, "bold"), pady=8)
    start_btn.pack(side="left", fill="x", expand=True)
    stop_btn = tk.Button(play, text="■  STOP", fg="white",
                         bg="#8a8a8a", activebackground="#c62828",
                         activeforeground="white", relief="flat",
                         font=("Segoe UI", 15, "bold"), pady=8,
                         state="disabled")
    stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

    options = ttk.Frame(root, padding=(12, 6))
    options.pack(fill="x")
    dry_var = tk.BooleanVar(value=getattr(args, "dry_run", False))
    dry_check = ttk.Checkbutton(options, text="Dry run (type nothing)",
                                variable=dry_var)
    dry_check.pack(side="left")
    ttk.Label(options, text="Rounds:").pack(side="left", padx=(18, 4))
    rounds_var = tk.StringVar(value=str(getattr(args, "rounds", 1)))
    rounds_box = ttk.Spinbox(options, from_=1, to=99, width=4,
                             textvariable=rounds_var)
    rounds_box.pack(side="left")

    # -- last event + optional activity feed -------------------------------
    last_event = tk.Label(root, text="Ready.", anchor="w", fg="#777777",
                          font=("Consolas", 9), padx=12)
    last_event.pack(fill="x", pady=(0, 6))

    feed_frame = ttk.LabelFrame(root, text="Activity", padding=(4, 2))
    feed_view = ScrolledText(feed_frame, height=12, state="disabled",
                             font=("Consolas", 9), borderwidth=0)
    feed_view.pack(fill="both", expand=True)
    feed_visible = tk.BooleanVar(value=False)

    def toggle_feed():
        if feed_visible.get():
            feed_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        else:
            feed_frame.pack_forget()

    # -- behaviour ---------------------------------------------------------
    def set_state(key):
        label, colour = STATES[key]
        banner.config(text=label, bg=colour)

    def append(text):
        shown = display_text(text)
        last_event.config(text=shown)
        feed_view["state"] = "normal"
        feed_view.insert("end", shown + "\n")
        feed_view.see("end")
        feed_view["state"] = "disabled"

    def set_running(running, stoppable=False):
        start_btn["state"] = "disabled" if running else "normal"
        start_btn["bg"] = "#8a8a8a" if running else "#1f8a3b"
        stop_btn["state"] = "normal" if running and stoppable else "disabled"
        stop_btn["bg"] = "#c62828" if running and stoppable else "#8a8a8a"
        dry_check.state(["disabled"] if running else ["!disabled"])
        rounds_box["state"] = "disabled" if running else "normal"

    def launch(argv, banner_state="working", stoppable=False):
        run_args = build_parser().parse_args(argv)
        command = run_args.command or "run"
        if command == "run":
            started = runner.launch(cmd_run, run_args,
                                    stop_event=runner.stop)
        else:
            started = runner.launch(COMMANDS[command], run_args)
        if started:
            set_running(True, stoppable=stoppable)
            set_state(banner_state)

    def start_play():
        for key in card_vars:
            card_vars[key].config(text="--", fg="black")
        launch((["--dry-run"] if dry_var.get() else [])
               + ["--rounds", rounds_var.get() or "1"],
               banner_state="unknown", stoppable=True)

    def pick_and_run(subcommand):
        clip = filedialog.askopenfilename(
            title=f"Choose a recording to {subcommand}",
            filetypes=[("Videos", "*.mp4 *.mkv *.avi"), ("All files", "*")])
        if clip:
            launch([subcommand, clip])

    def run_match():
        text = simpledialog.askstring(
            "Match", "OCR read to resolve against the lexicon:", parent=root)
        if text and text.strip():
            launch(["match", text.strip()])

    start_btn["command"] = start_play
    stop_btn["command"] = lambda: (stop_btn.configure(state="disabled"),
                                   runner.request_stop())

    # -- menubar -----------------------------------------------------------
    menubar = tk.Menu(root)
    tools = tk.Menu(menubar, tearoff=0)
    tools.add_command(label="Doctor", command=lambda: launch(["doctor"]))
    tools.add_command(label="Update word data",
                      command=lambda: launch(["update-data"]))
    tools.add_separator()
    tools.add_command(label="Analyze clip...",
                      command=lambda: pick_and_run("analyze"))
    tools.add_command(label="Replay clip...",
                      command=lambda: pick_and_run("replay"))
    tools.add_separator()
    tools.add_command(label="Match text...", command=run_match)
    menubar.add_cascade(label="Tools", menu=tools)
    view = tk.Menu(menubar, tearoff=0)
    view.add_checkbutton(label="Show activity", variable=feed_visible,
                         command=toggle_feed)
    menubar.add_cascade(label="View", menu=view)
    root.config(menu=menubar)

    def poll():
        try:
            while True:
                kind, payload = runner.events.get_nowait()
                if kind == "line":
                    append(payload)
                    updates = interpret(payload)
                    if "state" in updates:
                        set_state(updates["state"])
                    if "score" in updates:
                        card_vars["score"].config(text=updates["score"])
                    if "combo" in updates:
                        card_vars["combo"].config(
                            text=f"x{updates['combo']}",
                            fg="#c62828" if updates["combo_lost"]
                            else "#1f8a3b")
                    if "typed" in updates:
                        card_vars["typed"].config(text=updates["typed"])
                else:   # done
                    set_running(False)
                    set_state("idle")
        except queue.Empty:
            pass
        root.after(_POLL_MS, poll)

    def close():
        runner.request_stop()
        runner.join(timeout=3.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    append(f"automaton {__version__} -- Start plays; full-detail logs land "
           f"in logs/run-<stamp>.log.")
    poll()
    if getattr(args, "smoke", False):
        root.after(700, close)
    root.mainloop()
    return 0
