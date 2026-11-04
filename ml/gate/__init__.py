"""Trainable plant-vs-other image gate.

The production gate (``backend/app/utils/plant_gate.py``) uses a small random
forest trained by ``ml/gate/train_plant_gate.py`` on hundreds of thousands of
extracted features: HSV colour statistics, texture, and ImageNet-pretrained
semantics (plant synset scores, top-1/top-5 behaviour). It answers one
question the 2-class YOLO detector cannot: *"is this a crop-field photo at
all?"* before any detection/classification result is shown.

This package is deliberately small and re-trainable: point the trainer at a
folder of your own non-plant photos (people, objects, animals, roads) and run
it again to improve the gate with real negatives.
"""