import 'package:flutter/material.dart';
import 'core/api_config.dart';
import 'core/app_theme.dart';
import 'core/constants.dart';
import 'core/router.dart';

void main() {
  ApiConfig.init();

  // 1) Optional explicit API base URL for development, e.g.:
  //    flutter run --dart-define=API_BASE_URL=http://192.168.10.36:8000/api/v1
  //    Only honoured in debug builds (release always uses the production URL).
  const String apiBaseUrl = String.fromEnvironment('API_BASE_URL');
  if (apiBaseUrl.isNotEmpty) {
    ApiConfig.setCustomUrl(apiBaseUrl);
  }

  // 2) Optional environment selector, e.g.: --dart-define=ENVIRONMENT=device
  //    (local | emulator | device | production). No-op in release builds.
  const String environment = String.fromEnvironment('ENVIRONMENT');
  if (environment.isNotEmpty) {
    final Environment? target = switch (environment) {
      'local' => Environment.local,
      'emulator' => Environment.emulator,
      'device' => Environment.device,
      'production' => Environment.production,
      _ => null,
    };
    if (target != null) {
      ApiConfig.setEnvironment(target);
    }
  }

  runApp(const GroundnutAIApp());
}

class GroundnutAIApp extends StatelessWidget {
  const GroundnutAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppConstants.appName,
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.system,
      initialRoute: AppConstants.splashScreen,
      onGenerateRoute: AppRouter.generateRoute,
    );
  }
}
