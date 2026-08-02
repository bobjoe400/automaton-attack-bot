"""The control panel window.

Layout: a status banner, stat cards (score / combo / words typed), play
controls, a tools row exposing the other subcommands, and an activity
feed. All actual work happens in :class:`tasks.TaskRunner`.
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
        from tkinter import filedialog, ttk
        from tkinter.scrolledtext import ScrolledText
    except ImportError as exc:
        print(f"error: tkinter unavailable ({exc}); run with --cli instead")
        return 1

    from ..cli import COMMANDS, build_parser, cmd_run

    runner = TaskRunner()
    root = tk.Tk()
    root.title(f"Automaton Attack Bot v{__version__}")
    root.minsize(600, 480)

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

    # -- play controls -----------------------------------------------------
    play = ttk.LabelFrame(root, text="Play", padding=(10, 6))
    play.pack(fill="x", padx=12)
    start_btn = ttk.Button(play, text="Start")
    start_btn.pack(side="left")
    stop_btn = ttk.Button(play, text="Stop", state="disabled")
    stop_btn.pack(side="left", padx=(6, 0))
    dry_var = tk.BooleanVar(value=getattr(args, "dry_run", False))
    ttk.Checkbutton(play, text="Dry run",
                    variable=dry_var).pack(side="left", padx=(18, 0))
    ttk.Label(play, text="Rounds:").pack(side="left", padx=(18, 4))
    rounds_var = tk.StringVar(value=str(getattr(args, "rounds", 1)))
    ttk.Spinbox(play, from_=1, to=99, width=4,
                textvariable=rounds_var).pack(side="left")

    # -- tools -------------------------------------------------------------
    tools = ttk.LabelFrame(root, text="Tools", padding=(10, 6))
    tools.pack(fill="x", padx=12, pady=(8, 0))
    tool_btns = []

    def tool_button(label, command):
        btn = ttk.Button(tools, text=label, command=command)
        btn.pack(side="left", padx=(0, 6))
        tool_btns.append(btn)
        return btn

    match_var = tk.StringVar()
    match_entry = ttk.Entry(tools, textvariable=match_var, width=18)

    # -- activity feed -----------------------------------------------------
    feed_frame = ttk.LabelFrame(root, text="Activity", padding=(4, 2))
    feed_frame.pack(fill="both", expand=True, padx=12, pady=(8, 12))
    feed_view = ScrolledText(feed_frame, height=12, state="disabled",
                             font=("Consolas", 9), borderwidth=0)
    feed_view.pack(fill="both", expand=True)

    # -- behaviour ---------------------------------------------------------
    def set_state(key):
        label, colour = STATES[key]
        banner.config(text=label, bg=colour)

    def append(text):
        feed_view["state"] = "normal"
        feed_view.insert("end", display_text(text) + "\n")
        feed_view.see("end")
        feed_view["state"] = "disabled"

    def set_running(running, stoppable=False):
        start_btn["state"] = "disabled" if running else "normal"
        stop_btn["state"] = "normal" if running and stoppable else "disabled"
        for btn in tool_btns:
            btn["state"] = "disabled" if running else "normal"

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
        text = match_var.get().strip()
        if text:
            launch(["match", text])

    start_btn["command"] = start_play
    stop_btn["command"] = lambda: (stop_btn.configure(state="disabled"),
                                   runner.request_stop())
    tool_button("Doctor", lambda: launch(["doctor"]))
    tool_button("Update data", lambda: launch(["update-data"]))
    tool_button("Analyze clip...", lambda: pick_and_run("analyze"))
    tool_button("Replay clip...", lambda: pick_and_run("replay"))
    match_entry.pack(side="left", padx=(12, 4))
    match_entry.bind("<Return>", lambda _e: run_match())
    tool_button("Match", run_match)

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
