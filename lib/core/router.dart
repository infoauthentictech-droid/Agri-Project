import 'package:flutter/material.dart';
import '../core/constants.dart';
import '../screens/splash_screen.dart';
import '../screens/home_screen.dart';
import '../screens/camera_screen.dart';
import '../screens/gallery_screen.dart';
import '../screens/image_preview_screen.dart';
import '../screens/model_selection_screen.dart';
import '../screens/detection_result_screen.dart';
import '../screens/herbicide_screen.dart';
import '../screens/herbicide_decision_screen.dart';
import '../screens/xai_screen.dart';
import '../screens/history_screen.dart';
import '../screens/history_detail_screen.dart';
import '../screens/settings_screen.dart';

class AppRouter {
  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case AppConstants.splashScreen:
        return MaterialPageRoute(
          builder: (_) => const SplashScreen(),
          settings: settings,
        );
      
      case AppConstants.homeScreen:
        return MaterialPageRoute(
          builder: (_) => const HomeScreen(),
          settings: settings,
        );
      
      case AppConstants.cameraScreen:
        return MaterialPageRoute(
          builder: (_) => const CameraScreen(),
          settings: settings,
        );
      
      case AppConstants.galleryScreen:
        return MaterialPageRoute(
          builder: (_) => const GalleryScreen(),
          settings: settings,
        );
      
      case AppConstants.imagePreviewScreen:
        // Removed unused local variable 'args'
        return MaterialPageRoute(
          builder: (_) => const ImagePreviewScreen(),
          settings: settings,
        );
      
      case AppConstants.modelSelectionScreen:
        return MaterialPageRoute(
          builder: (_) => const ModelSelectionScreen(),
          settings: settings,
        );
      
      case AppConstants.detectionResultScreen:
        // Removed unused local variable 'args'
        return MaterialPageRoute(
          builder: (_) => const DetectionResultScreen(),
          settings: settings,
        );
      
      case AppConstants.herbicideScreen:
        return MaterialPageRoute(
          builder: (_) => const HerbicideScreen(),
          settings: settings,
        );
      
      case AppConstants.herbicideDecisionScreen:
        return MaterialPageRoute(
          builder: (_) => const HerbicideDecisionScreen(),
          settings: settings,
        );
      
      case AppConstants.xaiScreen:
        return MaterialPageRoute(
          builder: (_) => const XAIScreen(),
          settings: settings,
        );
      
      case AppConstants.historyScreen:
        return MaterialPageRoute(
          builder: (_) => const HistoryScreen(),
          settings: settings,
        );
      
      case AppConstants.historyDetailScreen:
        return MaterialPageRoute(
          builder: (_) => const HistoryDetailScreen(),
          settings: settings,
        );
      
      case AppConstants.settingsScreen:
        return MaterialPageRoute(
          builder: (_) => const SettingsScreen(),
          settings: settings,
        );
      
      default:
        return MaterialPageRoute(
          builder: (_) => const SplashScreen(),
          settings: settings,
        );
    }
  }
}
