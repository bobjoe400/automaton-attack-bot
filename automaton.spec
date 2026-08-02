# PyInstaller build recipe: `uv run pyinstaller automaton.spec --noconfirm`
#
# Onedir on purpose: with onnxruntime + OpenCV aboard, a onefile exe would
# unpack hundreds of MB to temp on every launch. The release workflow zips
# the folder instead.

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
    excludes=["tkinter", "matplotlib", "PIL.ImageTk"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="automaton",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="automaton",
)
