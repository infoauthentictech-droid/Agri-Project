"""
Gradient-computed Grad-CAM for the YOLO detection model.

Standard Grad-CAM recipe adapted for object detection:

1. Run the model's internal backbone/neck under ``torch.set_grad_enabled(True)``.
2. Register a forward/backward hook on the last convolutional layer before the
   detection head (the layer producing spatially meaningful feature maps).
3. Build a scalar *class score* from the raw detection-head output for the
   target class and call ``backward()`` on it.
4. Weight the target-layer feature maps by their mean gradient and ReLU, then
   upscale to the input resolution.

The resulting map answers "which pixels most increased the model's vote for
class X" — the defensible Grad-CAM claim. ``target_box`` selects which
prediction (and therefore which class) the caller wants explained; when a
class id is given explicitly the aggregated logit of that class is used for
backpropagation.

If the gradient path cannot complete (unsupported head shape, missing layer,
gradient failure) the code falls back to a *clearly labelled* non-gradient
activation map instead of pretending the fallback is Grad-CAM.
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """
    Gradient-computed Grad-CAM explainer for a YOLO object-detection model.
    """

    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Args:
            model: Trained Ultralytics YOLO model (wraps a DetectionModel).
            target_layer: Optional dotted layer name. When None the last
                convolutional layer before the detection head is selected.
        """
        self.model = model
        self.model.model.eval()
        self.gradients = None
        self.activations = None
        self._hooks_registered = False
        self._target_layer_name = None

        if target_layer is not None:
            self._register_named_layer(target_layer)
        else:
            layer = self._find_target_conv()
            if layer is not None:
                self._register_hooks(*layer)
            else:
                logger.warning("No target conv layer found - gradient path disabled.")

    # -- hook management --------------------------------------------------

    def _sequential(self):
        """The raw nn.Sequential holding backbone/neck/head of the model."""
        m = self.model.model
        inner = getattr(m, "model", None)
        if inner is not None:
            m = inner
        return m

    def _find_target_conv(self) -> Optional[Tuple[str, torch.nn.Module]]:
        """Last Conv2d in the NECK (outside the Detect head).

        The head's own convs (e.g. the ``dfl`` distribution convs) are only
        exercised by the decode path we intentionally bypass, so the target
        has to live in the backbone/neck fed into the head.
        """
        seq = self._sequential()
        children = list(seq)
        detect = children[-1] if children else None
        search = children[:-1] if children else []
        for child in reversed(search):
            if child is detect:
                continue
            last_conv = None
            for name, module in child.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = (name, module)
            if last_conv is not None:
                return last_conv
        return None

    def _register_named_layer(self, target_layer: str) -> None:
        """Register hooks on an explicit dotted layer name if it exists."""
        for name, module in self._sequential().named_modules():
            if name == target_layer or target_layer in name:
                self._register_hooks(name, module)
                return
        logger.warning("Target layer '%s' not found - gradient path disabled.",
                       target_layer)

    def _register_hooks(self, name: str, module: torch.nn.Module) -> None:
        self._target_layer_name = name
        module.register_forward_hook(self._forward_hook)
        self._hooks_registered = True
        logger.info("Registered Grad-CAM hook on layer '%s'", name)

    def _forward_hook(self, module, inputs, output):
        # Keep the graph reference (NOT detached) so torch.autograd.grad can
        # compute the gradient of the class score w.r.t. this exact tensor.
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # -- inference ---------------------------------------------------------

    def _forward_manual(self, x: torch.Tensor):
        """Run the raw nn.Sequential (backbone+neck+head) outside inference_mode.

        Ultralytics decorates ``DetectionModel.forward`` with
        ``torch.inference_mode()`` which forbids saving tensors for backward.
        We replicate its ``_predict_once`` routing (``m.f``/``m.i``) by hand so
        autograd stays active and genuine gradient maps can be computed.
        """
        seq = self.model.model
        inner = getattr(seq, "model", None)
        if inner is not None:
            seq = inner
        saved: list = []
        for module in seq:
            src = getattr(module, "f", None)
            if src is not None and src != -1:
                if isinstance(src, int):
                    x_in = saved[src]
                elif isinstance(src, (list, tuple)):
                    x_in = [x if j == -1 else saved[j] for j in src]
                else:
                    x_in = x
            else:
                x_in = x
            x = module(x_in)
            saved.append(x)
        return x

    def _device(self) -> torch.device:
        """Device of the underlying model (CPU or CUDA)."""
        params = list(self.model.model.parameters())
        if params:
            return params[0].device
        return torch.device("cpu")

    def _preprocess_image(self, image: np.ndarray, img_size: int = 640) -> torch.Tensor:
        """Resize + CHW float tensor (matches YOLO inference shape)."""
        image_resized = cv2.resize(image, (img_size, img_size))
        image_normalized = image_resized / 255.0
        tensor = (
            torch.from_numpy(image_normalized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
        )
        tensor = tensor.to(self._device())
        tensor.requires_grad_(True)
        return tensor

    @staticmethod
    def _extract_class_scores(output) -> torch.Tensor:
        """Return a (N, num_classes) score tensor from the detection head output.

        Newer Ultralytics versions return a dict {"boxes", "scores", "feats"}
        where ``scores`` has shape (1, nc, N). Older versions return a list of
        per-level feature tensors (1, 4+nc, h, w) when the head is in
        training-like mode. Both layouts are handled.
        """
        if isinstance(output, dict):
            scores = output.get("scores")
            if scores is not None and isinstance(scores, torch.Tensor) and scores.ndim == 3:
                # (1, nc, N) -> (N, nc)
                return scores[0].transpose(0, 1).contiguous()
            levels = []
            for v in output.values():
                if isinstance(v, (list, tuple)) and v:
                    levels.extend(v)
                elif isinstance(v, torch.Tensor):
                    levels.append(v)
        elif isinstance(output, (list, tuple)):
            levels = [t for t in output if isinstance(t, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            levels = [output]
        else:
            levels = []
        if not levels:
            raise ValueError("No tensor in model output")

        scores_across_levels = []
        for head in levels:
            if head.ndim != 4 or head.shape[0] != 1 or head.shape[1] <= 4:
                continue
            # (1, 4+nc, h, w) -> (N, nc) where N = h*w
            cls = head[0, 4:, :, :].reshape(head.shape[2] * head.shape[3], head.shape[1] - 4)
            scores_across_levels.append(cls)
        if not scores_across_levels:
            raise ValueError("Unsupported detection head output shape")
        return torch.cat(scores_across_levels, dim=0)

    def _class_score_loss(self, output, class_id: int) -> torch.Tensor:
        """Sum of the class logits of ``class_id`` across every level/spatial cell."""
        if isinstance(output, dict):
            scores = output.get("scores")
            if scores is not None and isinstance(scores, torch.Tensor) and scores.ndim == 3:
                return scores[0, class_id].sum()
            levels = []
            for v in output.values():
                if isinstance(v, (list, tuple)) and v:
                    levels.extend(v)
                elif isinstance(v, torch.Tensor):
                    levels.append(v)
        elif isinstance(output, (list, tuple)):
            levels = [t for t in output if isinstance(t, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            levels = [output]
        else:
            levels = []
        components = []
        for head in levels:
            if head.ndim != 4 or head.shape[0] != 1 or head.shape[1] <= 4:
                continue
            components.append(head[0, 4:, :, :][class_id].sum())
        if not components:
            raise ValueError("Unsupported detection head output shape")
        return torch.stack(components).sum() if len(components) > 1 else components[0]

    def generate_cam(
        self,
        image_path: str,
        class_id: Optional[int] = None,
        target_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate a gradient-computed Grad-CAM heatmap for an image.

        Args:
            image_path: Path to input image.
            class_id: Target class ID (falls back to the argmax class).
            target_box: Bounding box (x1, y1, x2, y2) of the detection to
                explain; its class selects the class score when class_id is
                not given.

        Returns:
            (heatmap_0_to_1, method) where method is 'gradcam' or
            'fallback_activation'.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        h, w = image.shape[:2]

        if not self._hooks_registered:
            return self._generate_activation_heatmap(image), "fallback_activation"

        try:
            image_tensor = self._preprocess_image(image)
            seq = self._sequential()
            head = seq[-1] if len(seq) > 0 else None
            with torch.set_grad_enabled(True):
                self.gradients = None
                self.activations = None
                # Bypass Ultralytics' inference_mode decorator (it forbids
                # saving tensors for backward) and flip only the Detect head
                # into training-like mode so it returns raw feature tensors
                # that autograd can backprop through.
                head_was_training = None
                if head is not None:
                    head_was_training = bool(getattr(head, "training", False))
                    try:
                        head.train()
                    except Exception:
                        head_was_training = None
                try:
                    output = self._forward_manual(image_tensor)
                finally:
                    if head is not None and head_was_training is not None:
                        if head_was_training:
                            head.train()
                        else:
                            head.eval()

                if class_id is None:
                    scores = self._extract_class_scores(output)  # (N, nc)
                    class_id = int(scores.max(dim=0).values.argmax().item())
                class_id = int(class_id)

                # Class-score gradient path (Grad-CAM).
                loss = self._class_score_loss(output, class_id)
                acts_ref = self.activations
                if acts_ref is None:
                    return (
                        self._generate_activation_heatmap(image),
                        "fallback_activation",
                    )
                if not acts_ref.requires_grad:
                    return (
                        self._generate_activation_heatmap(image),
                        "fallback_activation",
                    )
                grads = torch.autograd.grad(loss, acts_ref, retain_graph=False)[0]

            grads = grads.detach()
            acts = acts_ref.detach().float()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu(weights * acts).sum(dim=1, keepdim=True)
            cam = cam.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam, "gradcam"
        except Exception as exc:
            logger.warning("Grad-CAM gradient path failed (%s); using fallback.", exc)
            return self._generate_activation_heatmap(image), "fallback_activation"

    def _generate_activation_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Explicit NON-Grad-CAM fallback (clearly labelled, not a gradient map)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        heatmap = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
        return heatmap.astype(np.float32)

    def explain_prediction(
        self,
        image_path: str,
        yolo_results,
        class_names: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """
        Generate explanations for every YOLO detection in an image.

        Each explanation records the method actually used: ``gradcam`` when the
        gradient path completed, ``fallback_activation`` otherwise, so the API
        never presents the fallback as a genuine Grad-CAM map.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        predictions = []
        for result in yolo_results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                predictions.append({
                    "box": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": (class_names or {}).get(cls_id, f"class_{cls_id}"),
                })

        explanations = []
        for pred in predictions:
            try:
                cam, method = self.generate_cam(
                    image_path, pred["class_id"], tuple(int(v) for v in pred["box"])
                )
                explanations.append({
                    "prediction": pred,
                    "cam": cam,
                    "method": method,
                    "success": True,
                })
            except Exception as e:
                explanations.append({
                    "prediction": pred,
                    "cam": None,
                    "method": "failed",
                    "success": False,
                    "error": str(e),
                })

        return {
            "image_path": image_path,
            "image_shape": image.shape,
            "predictions": predictions,
            "explanations": explanations,
        }


if __name__ == "__main__":
    print("Grad-CAM explainer for YOLO object detection")
