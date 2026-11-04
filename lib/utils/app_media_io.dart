/// Cross-platform media/file helpers — native implementation.
///
/// The web build swaps this file via conditional import in `app_media.dart`
/// (see `app_media_io.dart` / `app_media_web.dart`). Kept as pure `dart:io`.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

/// Read the full bytes of a file (or a blob URL on web).
Future<Uint8List> readMediaBytes(String path) async => File(path).readAsBytes();

/// Whether the media at [path] exists.
Future<bool> mediaExists(String path) async => File(path).exists();

/// Byte length of the media at [path].
Future<int> mediaLength(String path) async => File(path).length();

/// Build a widget that renders the media at [path].
Widget localImage(String path, {BoxFit fit = BoxFit.contain}) {
  return Image.file(File(path), fit: fit);
}