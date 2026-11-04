import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../models/detection_result.dart';
import '../utils/app_media.dart';

/// Widget that displays an image with semantic segmentation masks overlaid.
/// Uses pixel-level segmentation masks from the backend (green=crop, red=weed, black=background).
/// Falls back to bounding box visualization if segmentation mask is not available.
class MaskedImage extends StatefulWidget {
  final String imagePath;
  final List<Prediction> predictions;
  final BoxFit fit;
  final double? width;
  final double? height;
  final String? segmentationMask; // Base64 encoded PNG mask from backend

  const MaskedImage({
    super.key,
    required this.imagePath,
    required this.predictions,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
    this.segmentationMask,
  });

  @override
  State<MaskedImage> createState() => _MaskedImageState();
}

class _MaskedImageState extends State<MaskedImage> {
  Uint8List? _maskedBytes;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadAndMaskImage();
  }

  Future<void> _loadAndMaskImage() async {
    try {
      // If semantic segmentation mask is available, use it
      if (widget.segmentationMask != null && widget.segmentationMask!.isNotEmpty) {
        final maskBytes = base64Decode(widget.segmentationMask!);
        final imageBytes = await readMediaBytes(widget.imagePath);

        // Combine original image with segmentation mask
        final combinedBytes = await _combineImageWithMask(imageBytes, maskBytes);

        if (mounted) {
          setState(() {
            _maskedBytes = combinedBytes;
            _isLoading = false;
          });
        }
      } else {
        // Fallback to bounding box visualization
        final bytes = await readMediaBytes(widget.imagePath);
        final codec = await ui.instantiateImageCodec(bytes);
        final frame = await codec.getNextFrame();

        // Create masked version with bounding boxes
        final maskedBytes = await _createBoundingBoxMask(frame.image);

        if (mounted) {
          setState(() {
            _maskedBytes = maskedBytes;
            _isLoading = false;
          });
        }
        frame.image.dispose();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<Uint8List> _combineImageWithMask(Uint8List imageBytes, Uint8List maskBytes) async {
    final codec = await ui.instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;

    final maskCodec = await ui.instantiateImageCodec(maskBytes);
    final maskFrame = await maskCodec.getNextFrame();
    final mask = maskFrame.image;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Draw original image
    canvas.drawImage(image, Offset.zero, Paint());

    // Draw semi-transparent segmentation mask on top
    final maskPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.5)
      ..blendMode = BlendMode.multiply;

    canvas.drawImage(mask, Offset.zero, maskPaint);

    final picture = recorder.endRecording();
    final img = await picture.toImage(image.width, image.height);
    final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
    img.dispose();
    image.dispose();
    mask.dispose();

    return byteData!.buffer.asUint8List();
  }

  Future<Uint8List> _createBoundingBoxMask(ui.Image image) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Draw original image
    canvas.drawImage(image, Offset.zero, Paint());

    // Apply colored masks for each prediction (bounding box fallback)
    for (final prediction in widget.predictions) {
      final box = prediction.boundingBox;
      final color = _getMaskColor(prediction.className);

      final rect = Rect.fromLTRB(
        box.x1,
        box.y1,
        box.x2,
        box.y2,
      );

      // Draw semi-transparent colored overlay
      final paint = Paint()
        ..color = color.withValues(alpha: 0.6)
        ..style = PaintingStyle.fill;

      canvas.drawRect(rect, paint);

      // Draw border
      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3;

      canvas.drawRect(rect, borderPaint);
    }

    final picture = recorder.endRecording();
    final img = await picture.toImage(image.width, image.height);
    final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
    img.dispose();

    return byteData!.buffer.asUint8List();
  }

  Color _getMaskColor(String className) {
    final lowerName = className.toLowerCase();

    // Green for groundnut plants
    if (lowerName.contains('groundnut') ||
        lowerName.contains('plant') ||
        lowerName.contains('crop')) {
      return const Color(0xFF2E7D32); // Green
    }

    // Red for weeds
    if (lowerName.contains('weed')) {
      return const Color(0xFFD32F2F); // Red
    }

    // Black for other objects
    return const Color(0xFF000000); // Black
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Container(
        width: widget.width,
        height: widget.height,
        color: Colors.grey[200],
        child: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    if (_maskedBytes != null) {
      return Image.memory(
        _maskedBytes!,
        fit: widget.fit,
        width: widget.width,
        height: widget.height,
      );
    }

    // Fallback to original image if masking fails
    return SizedBox(
      width: widget.width,
      height: widget.height,
      child: localImage(
        widget.imagePath,
        fit: widget.fit,
      ),
    );
  }
}
