import 'package:flutter/foundation.dart';

enum Environment {
  local,
  emulator,
  device,
  production,
}

class ApiConfig {
  // Environment configuration
  static Environment _currentEnvironment = Environment.local;

  // Optional user-supplied custom base URL (dev mode only). When set, it takes
  // precedence over the environment default (see setCustomUrl).
  static String? _customUrl;

  // Base URLs for different environments
  static const Map<Environment, String> _baseUrls = {
    Environment.local: 'http://localhost:8000/api/v1',
    Environment.emulator: 'http://10.0.2.2:8000/api/v1',
    // Physical phone → set this to the LAN IP of the PC that runs the backend
    // (find it with `ipconfig` on Windows). Only used in debug builds.
    Environment.device: 'http://192.168.10.36:8000/api/v1',
    Environment.production: 'https://api.groundnutai.com/api/v1',
  };

  // API key — provided at build/run time via --dart-define (never hard-coded).
  // Not set => requests will be rejected by the backend (401), which is the
  // safe default. Set with:
  //   flutter run --dart-define=GROUNDNUT_API_KEY=your_key_here
  static const String apiKey = String.fromEnvironment('GROUNDNUT_API_KEY');

  // Build-time base-URL override baked into *debug* APKs by run_project.ps1
  // (--dart-define=API_BASE_URL=http://<PC_LAN_IP>:8000/api/v1). It lets the
  // phone reach this PC's backend over Wi-Fi without a USB cable. It only
  // takes effect in debug mode; release builds always use the HTTPS URL.
  static const String _definedBaseUrl = String.fromEnvironment('API_BASE_URL');

  // Auto-detect the environment (debug mode only).
  static void init() {
    if (!kReleaseMode) {
      _currentEnvironment = autoDetectEnvironment();
    }
  }

  // Get current base URL
  static String get baseUrl {
    // In release mode, always use the release URL. Override it at build time
    // when you have a real deployment:
    //   flutter build apk --release --dart-define=API_BASE_URL=https://api.your-domain.com/api/v1
    if (kReleaseMode) {
      if (_releaseUrl.isNotEmpty) return _releaseUrl;
      return _baseUrls[Environment.production]!;
    }
    // Build-time override baked into debug APKs by run_project.ps1
    // (--dart-define=API_BASE_URL=...). Takes precedence so the phone talks to
    // THIS PC's backend over Wi-Fi (no USB cable needed).
    if (_definedBaseUrl.isNotEmpty) {
      return _definedBaseUrl;
    }
    if (_customUrl != null) {
      return _customUrl!;
    }
    return _baseUrls[_currentEnvironment] ?? _baseUrls[Environment.local]!;
  }

  /// Optional release-mode base-URL override (see [baseUrl]).
  static const String _releaseUrl = String.fromEnvironment('API_BASE_URL');

  // Set environment (only works in debug mode)
  static void setEnvironment(Environment env) {
    if (!kReleaseMode) {
      _currentEnvironment = env;
      // A manual environment selection overrides any prior custom URL.
      _customUrl = null;
    }
  }

  // Validate URL format
  static bool isValidUrl(String url) {
    try {
      final uri = Uri.parse(url);
      return uri.hasScheme && (uri.scheme == 'http' || uri.scheme == 'https') && uri.hasAuthority;
    } catch (e) {
      return false;
    }
  }

  // Set custom URL with validation (only works in debug mode). On success the
  // validated URL becomes the active base URL.
  static bool setCustomUrl(String url) {
    if (kReleaseMode) {
      return false; // Cannot set custom URL in release mode
    }
    if (!isValidUrl(url)) {
      return false;
    }
    _customUrl = url;
    return true;
  }

  // Auto-detect environment based on platform.
  //
  // Android debug builds default to `device` (the LAN URL) because a manually
  // installed physical phone build must reach the PC that runs the backend.
  // Emulator-only debugging should pass --dart-define=ENVIRONMENT=emulator (or
  // --dart-define=API_BASE_URL=http://10.0.2.2:8000/api/v1) explicitly.
  static Environment autoDetectEnvironment() {
    if (kReleaseMode) {
      return Environment.production;
    }
    if (kIsWeb) {
      // Flutter web talks to the local backend from the same machine.
      return Environment.local;
    }
    if (defaultTargetPlatform == TargetPlatform.android) {
      return Environment.device;
    } else if (defaultTargetPlatform == TargetPlatform.iOS) {
      return Environment.device;
    } else {
      return Environment.local;
    }
  }

  // Timeouts
  static const int connectTimeout = 30000;
  static const int receiveTimeout = 30000;
  static const int sendTimeout = 30000;

  // Heavy /analyze (+ /explain) timeout in milliseconds. /analyze runs the full
  // pipeline (plant gate + YOLO detection + density + patch classification) and
  // can lazily load the gate's ImageNet featurizer and the classifier on first
  // use, so it needs a much longer budget than the quick endpoints above. The
  // standard connectTimeout (30 s) is kept for health/models/recommend calls.
  static const int analyzeTimeout = 120000;

  // Endpoints
  static const String health = '/health';
  static const String predict = '/predict';
  static const String detect = '/detect';
  static const String models = '/models';
  static const String classify = '/classify';
  static const String analyze = '/analyze';
  static const String recommend = '/recommend';
  static const String explain = '/explain';
  static const String segment = '/segment';
  static const String plantGate = '/plant_gate';

  // Headers (with API key when provided)
  static Map<String, String> get defaultHeaders => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    if (apiKey.isNotEmpty) 'X-API-Key': apiKey,
  };

  // Multipart upload headers
  static Map<String, String> get multipartHeaders => {
    'Accept': 'application/json',
    if (apiKey.isNotEmpty) 'X-API-Key': apiKey,
  };
}
