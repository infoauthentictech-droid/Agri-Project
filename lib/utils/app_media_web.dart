// Cross-platform media platform helpers — web implementation.
//
// On the web there is no local file system: image_picker returns `XFile`s whose
// `.path` is a `blob:` URL. Bytes are fetched over an (in-page) XHR and images
// are rendered with `Image.network`, which Flutter web resolves against the
// same-origin blob URL.
library;

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

Future<Uint8List> readMediaBytes(String path) async {
  final response = await http.get(Uri.parse(path));
  return response.bodyBytes;
}

Future<bool> mediaExists(String path) async => path.isNotEmpty;

Future<int> mediaLength(String path) async =>
    (await readMediaBytes(path)).length;

Widget localImage(String path, {BoxFit fit = BoxFit.contain}) {
  return Image.network(path, fit: fit);
}