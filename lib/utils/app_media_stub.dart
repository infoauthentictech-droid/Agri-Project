// Cross-platform media platform helpers — fallback (never selected in practice).
library;

import 'dart:typed_data';

import 'package:flutter/material.dart';

Future<Uint8List> readMediaBytes(String path) =>
    throw UnsupportedError('No media backend selected');

Future<bool> mediaExists(String path) => throw UnsupportedError('No media backend');

Future<int> mediaLength(String path) => throw UnsupportedError('No media backend');

Widget localImage(String path, {BoxFit fit = BoxFit.contain}) =>
    throw UnsupportedError('No media backend');