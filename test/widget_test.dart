import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:groundnut_ai/core/app_theme.dart';
import 'package:groundnut_ai/core/constants.dart';
import 'package:groundnut_ai/core/router.dart';
import 'package:groundnut_ai/models/analysis.dart';
import 'package:groundnut_ai/models/detection_result.dart';

/// Test-only entry route that pushes the REAL detection-result screen with an
/// *empty* analysis response, so the "No Detection" view (which contains the
/// "Retake Image" button) appears without any network I/O.
class _DetectionLauncher extends StatelessWidget {
  const _DetectionLauncher();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            Navigator.pushNamed(
              context,
              AppConstants.detectionResultScreen,
              arguments: {
                'imagePath': 'no_such_file.jpg',
                'analysis': AnalysisResponse(
                  success: true,
                  detectionModel: 'yolov8n',
                  classificationModel: 'mobilenet',
                  detections: <Prediction>[],
                  detectionTimeMs: 1,
                  classificationTimeMs: 1,
                  labelsConfirmed: false,
                ),
              },
            );
          },
          child: const Text('Open Detection'),
        ),
      ),
    );
  }
}

/// Launches the detection-result screen with a non-plant rejection, so the
/// "Not a Plant Photo" exception view (Retake Photo / Upload Image) appears.
class _NotPlantLauncher extends StatelessWidget {
  const _NotPlantLauncher();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            Navigator.pushNamed(
              context,
              AppConstants.detectionResultScreen,
              arguments: {
                'imagePath': 'no_such_file.jpg',
                'analysis': AnalysisResponse(
                  success: false,
                  detectionModel: 'yolov8n',
                  classificationModel: 'mobilenet',
                  detections: <Prediction>[],
                  detectionTimeMs: 1,
                  classificationTimeMs: 1,
                  labelsConfirmed: false,
                  errorCode: 'not_plant_image',
                  message: 'This image does not look like a crop-field photo.',
                ),
              },
            );
          },
          child: const Text('Open NotPlant'),
        ),
      ),
    );
  }
}

void main() {
  testWidgets('Splash screen auto-navigates to Home (no stuck loading)', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.lightTheme,
        initialRoute: AppConstants.splashScreen,
        onGenerateRoute: AppRouter.generateRoute,
      ),
    );
    await tester.pumpAndSettle();

    // Splash (the "loading page") is shown first...
    expect(find.text(AppConstants.appName), findsOneWidget);

    // ...and after its delayed auto-navigation we must reach the Home screen.
    await tester.pump(const Duration(seconds: 3));
    await tester.pumpAndSettle();

    expect(find.text('Welcome to Groundnut AI'), findsOneWidget);
    expect(find.byType(CircularProgressIndicator), findsNothing);
  });

  testWidgets('Retake Image returns to Home without a stuck loading screen', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.lightTheme,
        initialRoute: '/launcher',
        onGenerateRoute: (RouteSettings settings) {
          // Test-internal routes (including MaterialApp's implicit root page)
          // must NOT fall through to AppRouter's default case, which returns
          // the SplashScreen (with its 2s auto-nav timer).
          if (settings.name == '/launcher' || settings.name == '/') {
            return MaterialPageRoute(
              builder: (_) => const _DetectionLauncher(),
              settings: settings,
            );
          }
          // All real app routes are resolved by the production router.
          return AppRouter.generateRoute(settings);
        },
      ),
    );
    await tester.pumpAndSettle();

    // Reach the detection result "No Detection" view.
    await tester.tap(find.text('Open Detection'));
    await tester.pumpAndSettle();
    expect(find.text('No Detection'), findsOneWidget);
    expect(find.text('Retake Image'), findsOneWidget);

    // Tap Retake Image -> must land back on Home instantly, deterministically.
    await tester.ensureVisible(find.text('Retake Image'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Retake Image'));
    await tester.pumpAndSettle();

    expect(find.text('Welcome to Groundnut AI'), findsOneWidget);
    // The launcher route and every screen below it were removed, and no loading
    // spinner may remain on stage.
    expect(find.text('Open Detection'), findsNothing);
    expect(find.byType(CircularProgressIndicator), findsNothing);
  });

  testWidgets('Not-a-plant rejection shows exception with Retake/Upload', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.lightTheme,
        initialRoute: '/launcher2',
        onGenerateRoute: (RouteSettings settings) {
          if (settings.name == '/launcher2' || settings.name == '/') {
            return MaterialPageRoute(
              builder: (_) => const _NotPlantLauncher(),
              settings: settings,
            );
          }
          return AppRouter.generateRoute(settings);
        },
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('Open NotPlant'));
    await tester.pumpAndSettle();

    // The exception view explains the rejection and offers retake/upload.
    expect(find.text('Not a Plant Photo'), findsOneWidget);
    expect(find.text('Retake Photo'), findsOneWidget);
    expect(find.text('Upload Image'), findsOneWidget);
    expect(find.text('Retake Image'), findsNothing);
  });
}
