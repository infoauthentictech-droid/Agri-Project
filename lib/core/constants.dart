class AppConstants {
  // App Info
  static const String appName = 'Groundnut AI';
  static const String appVersion = '1.0.0';
  
  // API Configuration
  static const String apiBaseUrl = 'http://localhost:8000/api/v1';
  static const int apiTimeout = 30000; // 30 seconds
  
  // Image Constraints
  static const int maxImageSizeMB = 10;
  static const int maxImageDimension = 4096;
  static const List<String> allowedImageFormats = ['jpeg', 'jpg', 'png'];
  
  // Confidence Thresholds
  static const double defaultConfidenceThreshold = 0.5;
  static const double lowConfidenceThreshold = 0.3;
  
  // Storage Keys
  static const String keyDetectionHistory = 'detection_history';
  static const String keySettings = 'app_settings';
  
  // Screen Names
  static const String splashScreen = '/splash';
  static const String homeScreen = '/home';
  static const String cameraScreen = '/camera';
  static const String galleryScreen = '/gallery';
  static const String imagePreviewScreen = '/image_preview';
  static const String modelSelectionScreen = '/model_selection';
  static const String detectionResultScreen = '/detection_result';
  static const String herbicideScreen = '/herbicide';
  static const String herbicideDecisionScreen = '/herbicide_decision';
  static const String xaiScreen = '/xai';
  static const String historyScreen = '/history';
  static const String historyDetailScreen = '/history_detail';
  static const String settingsScreen = '/settings';
}
