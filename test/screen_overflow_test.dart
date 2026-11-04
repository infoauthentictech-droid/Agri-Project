 import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:groundnut_ai/core/constants.dart';
import 'package:groundnut_ai/models/analysis.dart';
import 'package:groundnut_ai/models/detection_result.dart';
import 'package:groundnut_ai/screens/detection_result_screen.dart';
import 'package:groundnut_ai/screens/herbicide_decision_screen.dart';

/// Renders the busiest screens at a small-phone width (320 logical px) with
/// realistic sample data and asserts no RenderFlex overflow ("right overflowed
/// by N pixels") exceptions are thrown.
void main() {
  Future<void> pumpRoute(
    WidgetTester tester, {
    required String route,
    required Map<String, dynamic> arguments,
    required Widget screen,
  }) async {
    tester.view.physicalSize = const Size(320, 640);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        onGenerateRoute: (settings) => MaterialPageRoute(
          builder: (_) => screen,
          settings: settings,
        ),
        home: Builder(
          builder: (context) => Scaffold(
            body: Center(
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, route, arguments: arguments);
                },
                child: const Text('go'),
              ),
            ),
          ),
        ),
      ),
    );

    await tester.tap(find.text('go'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pumpAndSettle();
    // NOTE: we intentionally DO NOT call tester.takeException() here. Any
    // RenderFlex overflow is reported by the test framework with full
    // diagnostics (including the offending widget's creator chain) and fails
    // the test automatically.
  }

  final analysis = AnalysisResponse(
    success: true,
    detectionModel: 'yolov8n',
    classificationModel: 'mobilenet',
    detections: [
      Prediction(
        classId: 1,
        className: 'class_1',
        confidence: 0.92,
        boundingBox: BoundingBox(x1: 10, y1: 20, x2: 110, y2: 150),
      ),
      Prediction(
        classId: 1,
        className: 'class_1',
        confidence: 0.74,
        boundingBox: BoundingBox(x1: 30, y1: 40, x2: 130, y2: 180),
      ),
    ],
    detectionTimeMs: 150.0,
    classification: ClassificationResult(
      model: 'mobilenet',
      architecture: 'MobileNetV2',
      className: 'weed',
      classId: 1,
      confidence: 0.96,
      probabilities: [
        ProbabilityItem(classId: 1, className: 'weed', probability: 0.96),
        ProbabilityItem(classId: 0, className: 'groundnut', probability: 0.04),
      ],
    ),
    classificationTimeMs: 4.5,
  );

  final detection = DetectionResponse(
    success: true,
    predictions: [
      Prediction(
        classId: 1,
        className: 'class_1',
        confidence: 0.9,
        boundingBox: BoundingBox(x1: 0, y1: 0, x2: 10, y2: 10),
      ),
    ],
    numPredictions: 1,
    inferenceTimeMs: 5.0,
  );

  testWidgets('Detection result screen has no overflow at 320px width',
      (tester) async {
    await pumpRoute(
      tester,
      route: AppConstants.detectionResultScreen,
      arguments: {
        'imagePath': null,
        'analysis': analysis,
      },
      screen: const DetectionResultScreen(),
    );
  });

  testWidgets('Herbicide decision screen (species confirmation) has no overflow',
      (tester) async {
    await pumpRoute(
      tester,
      route: AppConstants.herbicideDecisionScreen,
      arguments: {'detectionResponse': detection},
      screen: const HerbicideDecisionScreen(),
    );
  });
}