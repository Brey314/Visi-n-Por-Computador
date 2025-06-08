It detect objects using Yolo8l, exported to .engine using Ultralytics.
You can use other models like, Yolo11, but you must have export to .engine.

After install Ultralytics you must reinstall torch

Example to cuda 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

You must have NVidia to run it.

You must export the models from Ultralytics using exporttoengine.py
