# PyInstaller build recipe: `uv run pyinstaller automaton.spec --noconfirm`
#
# Onefile: the release asset is a single automaton.exe. It self-extracts to
# temp on each launch (~2s measured) -- paid once per session, and worth it
# for a one-file download. tkinter rides along for the control panel, which
# is what a bare double-click opens.

from PyInstaller.utils.hooks import collect_all, collect_data_files

# rapidocr ships its ONNX models and yaml config as package data; none of it
# is found by import analysis.
datas, binaries, hiddenimports = collect_all("rapidocr_onnxruntime")
# our own package data (custom_vocab.txt), read via importlib.resources
datas += collect_data_files("automaton_attack_bot")

a = Analysis(
    ["packaging/entry.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports + ["mss", "pydirectinput"],
    excludes=["matplotlib", "PIL.ImageTk"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="automaton",
    console=True,
)
