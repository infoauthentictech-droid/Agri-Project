/// Cross-platform media helpers for Groundnut AI (flutter web + native).
///
/// The native build uses `dart:io` (`app_media_io.dart`); the web build uses
/// fetch-on-blob-URL (`app_media_web.dart`). Both expose the same 4 functions.
library;

export 'app_media_stub.dart'
    if (dart.library.io) 'app_media_io.dart'
    if (dart.library.html) 'app_media_web.dart';