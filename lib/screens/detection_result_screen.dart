import 'dart:convert';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_button.dart';
import '../widgets/app_card.dart';
import '../utils/app_media.dart';
import '../utils/loading_state.dart';
import '../utils/error_state.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';
import '../services/api_service.dart';
import '../services/local_history_storage_service.dart';
import '../models/detection_result.dart';
import '../models/analysis.dart';
import '../widgets/masked_image.dart';

enum ConfidenceLevel {
  high,
  moderate,
  low,
}

enum DensityCategory {
  low,
  medium,
  high,
}

class DetectionResultScreen extends StatefulWidget {
  const DetectionResultScreen({super.key});

  @override
  State<DetectionResultScreen> createState() => _DetectionResultScreenState();
}

class _DetectionResultScreenState extends State<DetectionResultScreen> {
  String? _imagePath;
  bool _isLoading = false;
  bool _hasError = false;
  String? _errorMessage;
  DetectionResponse? _detectionResponse;
  final ApiService _apiService = ApiService();
  final LocalHistoryStorageService _storageService = LocalHistoryStorageService();
  Size? _imageSize;

  /// Pre-computed combined analysis (from the ModelSelectionScreen flow).
  AnalysisResponse? _analysis;

  /// Guards against saving the same detection result to history multiple times
  /// (e.g. when didChangeDependencies re-fires with the same route arguments).
  bool _historySaved = false;

  /// Guards the automatic detection run from didChangeDependencies so it only
  /// fires once per route entry. The retry button still calls _runDetection()
  /// directly.
  bool _autoDetectStarted = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (args != null) {
      if (args.containsKey('imagePath')) {
        setState(() {
          _imagePath = args['imagePath'] as String?;
        });
      }
      if (args.containsKey('analysis')) {
        setState(() {
          _analysis = args['analysis'] as AnalysisResponse;
        });
      }
    }
    _loadImageSize();
    // When an analysis object is passed in, surface it straight away; otherwise
    // auto-run detection.
    final analysis = _analysis;
    if (analysis != null) {
      _applyAnalysis(analysis);
    } else if (_imagePath != null && !_autoDetectStarted) {
      _autoDetectStarted = true;
      _runDetection();
    }
  }

  void _applyAnalysis(AnalysisResponse analysis) {
    if (!mounted) return;
    setState(() {
      _analysis = analysis;
      _detectionResponse = DetectionResponse(
        success: analysis.success,
        predictions: analysis.detections,
        numPredictions: analysis.detections.length,
        inferenceTimeMs: analysis.detectionTimeMs,
      );
    });
    if (!_historySaved && _detectionResponse != null &&
        _detectionResponse!.predictions.isNotEmpty) {
      _historySaved = true;
      _saveToHistory(_detectionResponse!, analysis: analysis);
    }
  }

  /// Decode the picked image to get its real dimensions (used for coverage).
  Future<void> _loadImageSize() async {
    if (_imagePath == null) return;
    try {
      final bytes = await readMediaBytes(_imagePath!);
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      if (mounted) {
        setState(() {
          _imageSize = Size(
            frame.image.width.toDouble(),
            frame.image.height.toDouble(),
          );
        });
      }
      frame.image.dispose();
    } catch (_) {
      // Keep null -> fall back to an estimated area in coverage calculation.
    }
  }

  ConfidenceLevel _getConfidenceLevel(double confidence) {
    if (confidence >= 0.7) return ConfidenceLevel.high;
    if (confidence >= 0.5) return ConfidenceLevel.moderate;
    return ConfidenceLevel.low;
  }

  double _calculateAverageConfidence() {
    if (_detectionResponse == null || _detectionResponse!.predictions.isEmpty) {
      return 0.0;
    }
    final totalConfidence = _detectionResponse!.predictions
        .fold<double>(0.0, (sum, p) => sum + p.confidence);
    return totalConfidence / _detectionResponse!.predictions.length;
  }

  double _calculateWeedCoverage() {
    if (_detectionResponse == null || _detectionResponse!.predictions.isEmpty) {
      return 0.0;
    }

    // Use the real decoded image area when available; fall back to a nominal
    // 4 MP reference only if decoding failed.
    final area = _imageSize != null
        ? _imageSize!.width * _imageSize!.height
        : (2000.0 * 2000.0);
    double totalBoxArea = 0.0;

    for (final prediction in _detectionResponse!.predictions) {
      final box = prediction.boundingBox;
      final width = (box.x2 - box.x1).abs();
      final height = (box.y2 - box.y1).abs();
      totalBoxArea += width * height;
    }

    return (totalBoxArea / area * 100).clamp(0.0, 100.0);
  }

  // Density thresholds are kept consistent with the backend density estimator
  // (ml/density/density_estimator.py): low < 5%, medium 5-15%, high > 15%.
  DensityCategory _getDensityCategory(double coverage) {
    if (coverage < 5) return DensityCategory.low;
    if (coverage < 15) return DensityCategory.medium;
    return DensityCategory.high;
  }

  String _getDensityCategoryText(DensityCategory category) {
    switch (category) {
      case DensityCategory.low:
        return 'Low Density';
      case DensityCategory.medium:
        return 'Medium Density';
      case DensityCategory.high:
        return 'High Density';
    }
  }

  Color _getDensityColor(DensityCategory category) {
    switch (category) {
      case DensityCategory.low:
        return AppTheme.primaryGreen;
      case DensityCategory.medium:
        return AppTheme.warning;
      case DensityCategory.high:
        return AppTheme.error;
    }
  }

  /// Return to the Home screen deterministically.
  ///
  /// Pushes a fresh Home route and discards every route below it. This is safer
  /// than `Navigator.popUntil((route) => route.isFirst)`: the navigator's truly
  /// first route is the app's `/splash` boot screen, and when `popUntil` revives
  /// it the splash's one-shot auto-navigation never re-fires, leaving the app
  /// stuck on its loading view forever. Pushing Home by name (and removing the
  /// routes below it) always lands the user on a screen with no async work.
  void _retakeImage() {
    Navigator.pushNamedAndRemoveUntil(
      context,
      AppConstants.homeScreen,
      (route) => false,
    );
  }

  Future<void> _runDetection() async {
    if (_imagePath == null) return;

    setState(() {
      _isLoading = true;
      _hasError = false;
      _errorMessage = null;
      _detectionResponse = null;
      _historySaved = false;
    });

    try {
      // Check backend availability
      final isAvailable = await _apiService.isBackendAvailable();
      if (!isAvailable) {
        throw ServerException('Backend is not available. Please check your connection.');
      }

      // Run detection
      final response = await _apiService.detectWithImage(
        imagePath: _imagePath!,
        confidenceThreshold: 0.25,
        iouThreshold: 0.45,
      );

      // Run semantic segmentation for accurate masking
      String? segmentationMask;
      Map<String, double>? coverage;
      try {
        final segResult = await _apiService.segmentWithImage(
          imagePath: _imagePath!,
          targetSize: 224,
        );
        if (segResult['success'] == true) {
          segmentationMask = segResult['mask'] as String?;
          coverage = (segResult['coverage'] as Map<String, dynamic>?)
              ?.map((k, v) => MapEntry(k, (v as num).toDouble()));
        }
      } catch (e) {
        // Segmentation failure is not fatal - continue with detection only
        debugPrint('Segmentation failed: $e');
      }

      if (mounted) {
        setState(() {
          _isLoading = false;
          _detectionResponse = DetectionResponse(
            success: response.success,
            predictions: response.predictions,
            numPredictions: response.numPredictions,
            inferenceTimeMs: response.inferenceTimeMs,
            message: response.message,
            segmentationMask: segmentationMask,
            coverage: coverage,
          );
        });

        // Save to history (once per result).
        if (!_historySaved) {
          _historySaved = true;
          _saveToHistory(_detectionResponse!);
        }
      }
    } on NetworkException {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.networkUnavailable;
        });
      }
    } on ServerException {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.serverError;
        });
      }
    } on ValidationException catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.unexpectedError;
        });
      }
    }
  }

  Future<void> _saveToHistory(DetectionResponse response,
      {AnalysisResponse? analysis}) async {
    if (_imagePath == null || response.predictions.isEmpty) return;

    // Highest-confidence prediction, NOT the first listed one.
    final best = response.predictions.reduce(
      (a, b) => a.confidence >= b.confidence ? a : b,
    );

    // Density: prefer the backend's (uncalibrated) estimate so the stored
    // value matches what the app actually displayed.
    final String weedDensity;
    if (analysis?.densityEstimate != null) {
      weedDensity = analysis!.densityEstimate!.densityCategory == 'high'
          ? 'High Density'
          : analysis.densityEstimate!.densityCategory == 'medium'
              ? 'Medium Density'
              : 'Low Density';
    } else {
      final weedCoverage = _calculateWeedCoverage();
      weedDensity = _getDensityCategoryText(_getDensityCategory(weedCoverage));
    }

    // Durable thumbnail copy (base64) so history survives OS temp-file cleanup
    // and renders on web builds.
    final String? thumbnailBase64 = await _buildThumbnailBase64(_imagePath!);

    final historyRecord = _storageService.createDetectionRecord(
      imagePath: _imagePath!,
      imageData: thumbnailBase64,
      detectedClass: best.className,
      weedSpecies: best.className,
      confidence: best.confidence,
      weedDensity: weedDensity,
      hasRecommendation: false,
    );

    await _storageService.saveDetection(historyRecord);
  }

  /// Re-encode the picked image as a small PNG thumbnail (256px wide) and
  /// return base64. Returns null when decoding fails (history still saves
  /// with just the path).
  Future<String?> _buildThumbnailBase64(String imagePath) async {
    try {
      final bytes = await readMediaBytes(imagePath);
      final codec = await ui.instantiateImageCodec(bytes, targetWidth: 256);
      final frame = await codec.getNextFrame();
      final byteData = await frame.image.toByteData(
        format: ui.ImageByteFormat.png,
      );
      frame.image.dispose();
      if (byteData == null) return null;
      return base64Encode(byteData.buffer.asUint8List());
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () {
              Navigator.pushNamed(
                context,
                AppConstants.xaiScreen,
                arguments: {
                  'imagePath': _imagePath,
                  'detectionResponse': _detectionResponse,
                },
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Masked Image Preview
            if (_imagePath != null)
              Container(
                height: 280,
                width: double.infinity,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppTheme.radiusLg),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.1),
                      blurRadius: 8,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                clipBehavior: Clip.antiAlias,
                child: _detectionResponse != null && _detectionResponse!.predictions.isNotEmpty
                    ? MaskedImage(
                        imagePath: _imagePath!,
                        predictions: _detectionResponse!.predictions,
                        fit: BoxFit.cover,
                        segmentationMask: _detectionResponse!.segmentationMask,
                      )
                    : localImage(_imagePath!, fit: BoxFit.cover),
              ),
            
            const SizedBox(height: AppTheme.spacingLg),
            
            if (_isLoading)
              const LoadingWidget(message: 'Analyzing image...')
            else if (_hasError)
              ErrorStateWidget(
                message: 'Detection Failed',
                subMessage: _errorMessage,
                onRetry: _runDetection,
              )
            else if (_detectionResponse == null)
              _buildInitialView()
            else
              _buildResultsView(),
          ],
        ),
      ),
    );
  }

  Widget _buildInitialView() {
    return Column(
      children: [
        Icon(
          Icons.camera_alt_outlined,
          size: 64,
          color: AppTheme.primaryGreen.withValues(alpha: 0.5),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        Text(
          'Ready to Analyze',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: AppTheme.spacingXs),
        Text(
          'Tap the button below to start AI detection',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            color: AppTheme.textSecondary,
          ),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        AppButton(
          text: 'Run Detection',
          icon: Icons.analytics,
          onPressed: _runDetection,
        ),
      ],
    );
  }

  Widget _buildResultsView() {
    final predictions = _detectionResponse!.predictions;
    final averageConfidence = _calculateAverageConfidence();
    final confidenceLevel = _getConfidenceLevel(averageConfidence);

    // Prefer the backend's (overlap-aware, uncalibrated) density estimate.
    final density = _analysis?.densityEstimate;
    final weedCount = density?.weedCount ?? predictions.length;
    final weedCoverage = density?.coveragePercent ?? _calculateWeedCoverage();
    final densityCategory = density != null
        ? _densityCategoryFromString(density.densityCategory)
        : _getDensityCategory(weedCoverage);

    // Handle no detection case
    if (predictions.isEmpty) {
      return _buildNoDetectionView(
          message: _analysis?.message, errorCode: _analysis?.errorCode);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Detector-label honesty note when semantics are still unconfirmed.
        if (_analysis != null && !_analysis!.labelsConfirmed) ...[
          _buildLabelNoteCard(_analysis!.labelNote),
          const SizedBox(height: AppTheme.spacingMd),
        ],

        // Confidence Level Indicator (only show for high or low, not moderate)
        if (confidenceLevel != ConfidenceLevel.moderate) ...[
          _buildConfidenceCard(confidenceLevel, averageConfidence),
          const SizedBox(height: AppTheme.spacingLg),
        ],

        // Classification summary (from the /analyze flow)
        if (_analysis?.classification != null) ...[
          _buildClassificationCard(context),
          const SizedBox(height: AppTheme.spacingLg),
        ],

        // Weed density estimate (backend /analyze when present, else local).
        _buildDensityCard(densityCategory, weedCount, weedCoverage,
            calibrated: density?.calibrated ?? false),
        const SizedBox(height: AppTheme.spacingLg),

        // Semantic segmentation coverage (if available)
        if (_detectionResponse?.coverage != null) ...[
          _buildCoverageCard(_detectionResponse!.coverage!),
          const SizedBox(height: AppTheme.spacingLg),
        ],

        const SizedBox(height: AppTheme.spacingLg),
        
        // Detection Results
        Text(
          'Detected Objects',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: AppTheme.spacingSm),
        
        ...predictions.map((prediction) => _buildPredictionCard(prediction, confidenceLevel)),
        
        const SizedBox(height: AppTheme.spacingLg),
        
        // Action Buttons
        Row(
          children: [
            Expanded(
              child: AppButton(
                text: 'Get Recommendation',
                icon: Icons.medication,
                onPressed: () {
                  Navigator.pushNamed(
                    context,
                    AppConstants.herbicideDecisionScreen,
                    arguments: {'detectionResponse': _detectionResponse},
                  );
                },
              ),
            ),
            const SizedBox(width: AppTheme.spacingMd),
            Expanded(
              child: AppButton(
                text: 'New Image',
                isSecondary: true,
                onPressed: _retakeImage,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildDensityCard(
    DensityCategory category,
    int weedCount,
    double coverage, {
    required bool calibrated,
  }) {
    final color = _getDensityColor(category);
    return AppCard(
      color: color.withValues(alpha: 0.06),
      borderColor: color,
      child: Row(
        children: [
          Icon(Icons.grass, color: color),
          const SizedBox(width: AppTheme.spacingSm),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _getDensityCategoryText(category),
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: color,
                        fontWeight: FontWeight.w600,
                      ),
                ),
                const SizedBox(height: AppTheme.spacingXxs),
                Text(
                  '$weedCount weed box(es) · ${coverage.toStringAsFixed(1)}% of the image'
                  '${calibrated ? '' : ' (uncalibrated estimate)'}',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.textSecondary,
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  DensityCategory _densityCategoryFromString(String value) {
    switch (value) {
      case 'high':
        return DensityCategory.high;
      case 'medium':
        return DensityCategory.medium;
      default:
        return DensityCategory.low;
    }
  }

  Widget _buildLabelNoteCard(String? note) {
    final text = note ?? 'Detector class labels are unconfirmed; results are '
        'shown with their raw class names.';
    return AppCard(
      color: AppTheme.warning.withValues(alpha: 0.1),
      borderColor: AppTheme.warning,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.science_outlined, color: AppTheme.warning, size: 20),
          const SizedBox(width: AppTheme.spacingXs),
          Expanded(
            child: Text(
              text,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildClassificationCard(BuildContext context) {
    final classification = _analysis?.classification;
    if (classification == null) {
      return const SizedBox.shrink();
    }
    final isWeed = classification.className.toLowerCase().contains('weed');
    final color = isWeed ? AppTheme.warning : AppTheme.success;

    // The detection box the classifier was run on (top-confidence box). The
    // classification label belongs to THIS object only — not to every box in
    // the "Detected Objects" list, which is why a mixed image can show a
    // different label here than in the list.
    final box = _analysis?.classificationBox;
    final boxLabel = box?.className;
    final disagrees = boxLabel != null &&
        boxLabel.toLowerCase() != classification.className.toLowerCase();

    final secondary = [
      if (classification.probabilities.length > 1)
        for (final prob in classification.probabilities.skip(1))
          '${prob.className} ${(prob.probability * 100).toStringAsFixed(1)}%',
    ].join(', ');

    return AppCard(
      color: color.withValues(alpha: 0.12),
      borderColor: color,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isWeed ? Icons.grass : Icons.eco,
                color: color,
                size: 28,
              ),
              const SizedBox(width: AppTheme.spacingXs),
              Expanded(
                child: Text(
                  'Classification',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTheme.spacingXs,
                  vertical: AppTheme.spacingXxs,
                ),
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                ),
                child: Text(
                  '${(classification.confidence * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: AppTheme.fontSizeSm,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingXs),
          Text(
            classification.className,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          if (box != null) ...[
            const SizedBox(height: AppTheme.spacingXxs),
            Text(
              'for the top detected object: ${box.className} '
              '(${(box.confidence * 100).toStringAsFixed(0)}% box score)',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
          ],
          if (secondary.isNotEmpty) ...[
            const SizedBox(height: AppTheme.spacingXxs),
            Text(
              secondary,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
          ],
          const SizedBox(height: AppTheme.spacingXxs),
          Text(
            'via ${classification.architecture}',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textHint,
            ),
          ),
          if (disagrees) ...[
            const SizedBox(height: AppTheme.spacingXs),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(AppTheme.spacingXs),
              decoration: BoxDecoration(
                color: AppTheme.warning.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                border: Border.all(color: AppTheme.warning),
              ),
              child: Text(
                'The detector labels this region as $boxLabel, but the '
                'classifier reads it as ${classification.className}. These are '
                'two separate models, so their labels can differ on a single '
                'patch.',
                style: const TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: AppTheme.fontSizeXs,
                ),
              ),
            ),
          ],
          const SizedBox(height: AppTheme.spacingXxs),
          Text(
            'Softmax confidence, 2 classes (groundnut/weed) \u2014 '
            'model confidence, not field accuracy.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textHint,
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfidenceCard(ConfidenceLevel level, double confidence) {
    Color color;
    String label;
    IconData icon;

    switch (level) {
      case ConfidenceLevel.high:
        color = AppTheme.success;
        label = 'High Confidence';
        icon = Icons.check_circle;
        break;
      case ConfidenceLevel.moderate:
        color = AppTheme.warning;
        label = 'Moderate Confidence';
        icon = Icons.info;
        break;
      case ConfidenceLevel.low:
        color = AppTheme.error;
        label = 'Low Confidence';
        icon = Icons.warning;
        break;
    }

    return AppCard(
      color: color.withValues(alpha: 0.1),
      borderColor: color,
      child: Row(
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(width: AppTheme.spacingXs),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: AppTheme.spacingXxs),
                Text(
                  '${(confidence * 100).toStringAsFixed(1)}% average confidence',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppTheme.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCoverageCard(Map<String, double> coverage) {
    // Build coverage bars for each class
    final coverageEntries = coverage.entries.toList();

    return AppCard(
      color: AppTheme.primaryGreen.withValues(alpha: 0.1),
      borderColor: AppTheme.primaryGreen,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.crop_square,
                color: AppTheme.primaryGreen,
                size: 24,
              ),
              const SizedBox(width: AppTheme.spacingXs),
              Expanded(
                child: Text(
                  'Segmentation Coverage',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: AppTheme.primaryGreen,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingSm),
          ...coverageEntries.map((entry) {
            final className = entry.key;
            final percentage = entry.value;
            final color = _getCoverageColor(className);

            return Padding(
              padding: const EdgeInsets.only(bottom: AppTheme.spacingXs),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        className,
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      Text(
                        '${percentage.toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.textSecondary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: percentage / 100,
                      backgroundColor: Colors.grey[300],
                      valueColor: AlwaysStoppedAnimation<Color>(color),
                      minHeight: 6,
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Color _getCoverageColor(String className) {
    final lowerName = className.toLowerCase();
    if (lowerName.contains('crop') || lowerName.contains('groundnut')) {
      return const Color(0xFF2E7D32); // Green
    }
    if (lowerName.contains('weed')) {
      return const Color(0xFFD32F2F); // Red
    }
    return Colors.grey;
  }

  Widget _buildPredictionCard(Prediction prediction, ConfidenceLevel overallLevel) {
    final predictionLevel = _getConfidenceLevel(prediction.confidence);
    final confidenceColor = _getConfidenceColor(predictionLevel);
    
    return Padding(
      padding: const EdgeInsets.only(bottom: AppTheme.spacingSm),
      child: AppCard(
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: confidenceColor.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
              ),
              child: Icon(
                _getConfidenceIcon(predictionLevel),
                color: confidenceColor,
                size: 24,
              ),
            ),
            const SizedBox(width: AppTheme.spacingSm),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    prediction.className,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: AppTheme.spacingXxs),
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppTheme.spacingXs,
                          vertical: AppTheme.spacingXxs,
                        ),
                        decoration: BoxDecoration(
                          color: confidenceColor,
                          borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _getConfidenceIcon(predictionLevel),
                              color: Colors.white,
                              size: 12,
                            ),
                            const SizedBox(width: AppTheme.spacingXxs),
                            Text(
                              '${(prediction.confidence * 100).toStringAsFixed(0)}%',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: AppTheme.fontSizeXs,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getConfidenceColor(ConfidenceLevel level) {
    switch (level) {
      case ConfidenceLevel.high:
        return AppTheme.primaryGreen;
      case ConfidenceLevel.moderate:
        return AppTheme.warning;
      case ConfidenceLevel.low:
        return AppTheme.error;
    }
  }

  IconData _getConfidenceIcon(ConfidenceLevel level) {
    switch (level) {
      case ConfidenceLevel.high:
        return Icons.check;
      case ConfidenceLevel.moderate:
        return Icons.warning;
      case ConfidenceLevel.low:
        return Icons.error;
    }
  }

  Widget _buildNoDetectionView({String? message, String? errorCode}) {
    final hasMessage = message != null && message.isNotEmpty;
    final isNotPlant = errorCode == 'not_plant_image';
    return Column(
      children: [
        Icon(
          isNotPlant ? Icons.warning_amber : Icons.search_off,
          size: 64,
          color: isNotPlant ? AppTheme.warning : AppTheme.textHint,
        ),
        const SizedBox(height: 16),
        Text(
          isNotPlant ? 'Not a Plant Photo' : 'No Detection',
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          isNotPlant
              ? 'This image does not appear to contain groundnut plants or '
                  'weeds, so it cannot be analysed.'
              : 'No object was confidently detected in this image.',
          style: const TextStyle(
            color: AppTheme.textSecondary,
          ),
          textAlign: TextAlign.center,
        ),
        if (hasMessage) ...[
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(AppTheme.spacingSm),
            decoration: BoxDecoration(
              color: AppTheme.warning.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
              border: Border.all(color: AppTheme.warning),
            ),
            child: Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 13,
              ),
            ),
          ),
        ],
        const SizedBox(height: 8),
        Text(
          isNotPlant
              ? 'Take or upload a clear photo of the groundnut field / weeds.'
              : 'Try capturing an image with better lighting or closer to the '
                  'crop.',
          style: const TextStyle(
            color: AppTheme.textHint,
            fontSize: 12,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        if (isNotPlant)
          Row(
            children: [
              Expanded(
                child: AppButton(
                  text: 'Retake Photo',
                  icon: Icons.camera_alt,
                  onPressed: () {
                    Navigator.pushNamedAndRemoveUntil(
                      context,
                      AppConstants.cameraScreen,
                      (route) => false,
                    );
                  },
                ),
              ),
              const SizedBox(width: AppTheme.spacingMd),
              Expanded(
                child: AppButton(
                  text: 'Upload Image',
                  isSecondary: true,
                  icon: Icons.photo_library,
                  onPressed: () {
                    Navigator.pushNamedAndRemoveUntil(
                      context,
                      AppConstants.galleryScreen,
                      (route) => false,
                    );
                  },
                ),
              ),
            ],
          )
        else
          AppButton(
            text: 'Retake Image',
            icon: Icons.camera_alt,
            onPressed: _retakeImage,
          ),
      ],
    );
  }
}
