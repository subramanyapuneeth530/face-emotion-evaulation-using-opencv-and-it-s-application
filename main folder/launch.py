# launch.py
import tkinter as tk
from functools import partial
import threading

import main  # imports run() from your main.py

MODELS = [
    ("EffNet-B0 (7)", "enet_b0_7"),
    ("EffNet-B2 (7)", "enet_b2_7"),
    # ("EffNet-B4 (8)", "enet_b4_8"),
]

def start_model(model_name, root):
    # close the launcher window and run the camera UI in the same process/thread
    root.destroy()
    main.run(model_name)

def run_launcher():
    root = tk.Tk()
    root.title("Choose Emotion Model")
    root.geometry("360x240")

    tk.Label(root, text="Select a model to launch:", font=("Segoe UI", 12, "bold")).pack(pady=16)

    for label, name in MODELS:
        tk.Button(root, text=label, width=26, height=2,
                  command=partial(start_model, name, root)).pack(pady=6)

    # optional: Quit button
    tk.Button(root, text="Quit", width=26, command=root.destroy).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_launcher()
