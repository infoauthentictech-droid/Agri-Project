import 'dart:convert';
import 'package:flutter/material.dart';
import '../utils/app_media.dart';

/// Web-safe rendering of a detection-history image.
///
/// Priority:
/// 1. [imageData] — a base64 thumbnail embedded in the history record (durable
///    copy that survives temp-file cleanup and works on web).
/// 2. [imagePath] — the original local file path (native) or blob URL (web),
///    only rendered when [mediaExists] reports it is still available.
///
/// Any failure falls back to a broken-image placeholder instead of an error.
class HistoryImage extends StatelessWidget {
  const HistoryImage({
    super.key,
    this.imageData,
    required this.imagePath,
    this.fit = BoxFit.cover,
  });

  final String? imageData;
  final String imagePath;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) {
    final data = imageData;
    if (data != null && data.isNotEmpty) {
      try {
        return Image.memory(
          base64Decode(data),
          fit: fit,
          errorBuilder: _placeholderBuilder,
        );
      } catch (_) {
        // fall through to path-based rendering
      }
    }
    return FutureBuilder<bool>(
      future: mediaExists(imagePath),
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const SizedBox.shrink();
        }
        if (snapshot.data != true) {
          return const Center(
            child: Icon(Icons.broken_image, size: 48, color: Colors.grey),
          );
        }
        return localImage(imagePath, fit: fit);
      },
    );
  }

  static Widget _placeholderBuilder(
      BuildContext context, Object error, StackTrace? stackTrace) {
    return const Center(
      child: Icon(Icons.broken_image, size: 48, color: Colors.grey),
    );
  }
}