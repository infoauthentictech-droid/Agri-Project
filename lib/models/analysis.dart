import 'detection_result.dart';

/// Models returned by the backend `GET /models` registry endpoint.
class ModelOption {
  final String name;
  final bool available;
  final bool loaded;
  final String kind;

  /// Held-out TEST-SET accuracy for this model (classifiers), or detection
  /// mAP@50. This is the honest basis for any "accuracy" claim: it is a
  /// model-level statistic from evaluation, never a per-image confidence.
  final double? accuracy;
  final String accuracyBasis;

  ModelOption({
    required this.name,
    required this.available,
    required this.loaded,
    required this.kind,
    this.accuracy,
    this.accuracyBasis = '',
  });

  factory ModelOption.fromJson(Map<String, dynamic> json) {
    return ModelOption(
      name: json['name'] as String,
      available: (json['available'] ?? false) as bool,
      loaded: (json['loaded'] ?? false) as bool,
      kind: (json['kind'] ?? '') as String,
      accuracy: (json['accuracy'] as num?)?.toDouble(),
      accuracyBasis: (json['accuracy_basis'] ?? '') as String,
    );
  }
}

class ModelCatalog {
  final List<ModelOption> detectionModels;
  final List<ModelOption> classificationModels;
  final String defaultDetectionModel;
  final String defaultClassificationModel;

  ModelCatalog({
    required this.detectionModels,
    required this.classificationModels,
    required this.defaultDetectionModel,
    required this.defaultClassificationModel,
  });

  factory ModelCatalog.fromJson(Map<String, dynamic> json) {
    List<ModelOption> parseList(dynamic raw) {
      if (raw is! List) return [];
      return raw
          .map((e) => ModelOption.fromJson(e as Map<String, dynamic>))
          .toList();
    }

    return ModelCatalog(
      detectionModels: parseList(json['detection_models']),
      classificationModels: parseList(json['classification_models']),
      defaultDetectionModel: (json['default_detection_model'] ?? 'yolov8n') as String,
      defaultClassificationModel: (json['default_classification_model'] ?? 'mobilenet') as String,
    );
  }
}

/// Result of `POST /plant_gate`: whether the photo plausibly shows a crop field.
class PlantGateResult {
  final bool isPlant;
  final double probability;
  final List<String> reasons;

  PlantGateResult({
    required this.isPlant,
    required this.probability,
    required this.reasons,
  });

  factory PlantGateResult.fromJson(Map<String, dynamic> json) {
    return PlantGateResult(
      isPlant: (json['is_plant'] ?? false) as bool,
      probability: (json['probability'] as num? ?? 0.0).toDouble(),
      reasons: ((json['reasons'] ?? []) as List).map((e) => e.toString()).toList(),
    );
  }
}

class ProbabilityItem {
  final int classId;
  final String className;
  final double probability;

  ProbabilityItem({
    required this.classId,
    required this.className,
    required this.probability,
  });

  factory ProbabilityItem.fromJson(Map<String, dynamic> json) {
    return ProbabilityItem(
      classId: json['class_id'] as int,
      className: json['class_name'] as String,
      probability: (json['probability'] as num).toDouble(),
    );
  }
}

class ClassificationResult {
  final String model;
  final String architecture;
  final String className;
  final int classId;
  final double confidence;
  final List<ProbabilityItem> probabilities;

  ClassificationResult({
    required this.model,
    required this.architecture,
    required this.className,
    required this.classId,
    required this.confidence,
    required this.probabilities,
  });

  factory ClassificationResult.fromJson(Map<String, dynamic> json) {
    final rawProbs = (json['probabilities'] ?? []) as List;
    return ClassificationResult(
      model: (json['model'] ?? '') as String,
      architecture: (json['architecture'] ?? '') as String,
      className: (json['class_name'] ?? 'unknown') as String,
      classId: (json['class_id'] ?? 0) as int,
      confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
      probabilities: rawProbs
          .map((e) => ProbabilityItem.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// Single-image classification response (POST /classify).
class ClassifyResponse {
  final bool success;
  final ClassificationResult? result;
  final double inferenceTimeMs;
  final String? message;

  ClassifyResponse({
    required this.success,
    this.result,
    required this.inferenceTimeMs,
    this.message,
  });

  factory ClassifyResponse.fromJson(Map<String, dynamic> json) {
    final rawResult = json['result'] as Map<String, dynamic>?;
    return ClassifyResponse(
      success: (json['success'] ?? false) as bool,
      result: rawResult != null
          ? ClassificationResult.fromJson(rawResult)
          : null,
      inferenceTimeMs: (json['inference_time_ms'] as num? ?? 0.0).toDouble(),
      message: json['message'] as String?,
    );
  }
}

/// Uncalibrated area-based density estimate returned by the backend /analyze.
///
/// ``calibrated`` is false and stays false until the detector label mapping is
/// confirmed and the 5%/15% thresholds are calibrated against field data. The
/// UI must present it as an *estimate*, never as a measurement.
class DensityEstimate {
  final List<int> weedClassIds;
  final int weedCount;
  final double coveragePercent;
  final String densityCategory; // low | medium | high
  final bool calibrated;
  final String note;

  DensityEstimate({
    required this.weedClassIds,
    required this.weedCount,
    required this.coveragePercent,
    required this.densityCategory,
    required this.calibrated,
    required this.note,
  });

  factory DensityEstimate.fromJson(Map<String, dynamic> json) {
    final rawIds = (json['weed_class_ids'] ?? []) as List;
    return DensityEstimate(
      weedClassIds: rawIds.map((e) => e as int).toList(),
      weedCount: (json['weed_count'] as num? ?? 0).toInt(),
      coveragePercent: (json['coverage_percent'] as num? ?? 0.0).toDouble(),
      densityCategory: (json['density_category'] ?? 'unknown') as String,
      calibrated: (json['calibrated'] ?? false) as bool,
      note: (json['note'] ?? '') as String,
    );
  }
}

/// The detection box (top-confidence) whose crop the classifier was run on.
class ClassificationBox {
  final int classId;
  final String className;
  final double confidence;

  ClassificationBox({
    required this.classId,
    required this.className,
    required this.confidence,
  });

  factory ClassificationBox.fromJson(Map<String, dynamic> json) {
    return ClassificationBox(
      classId: json['class_id'] as int,
      className: (json['class_name'] ?? '') as String,
      confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
    );
  }
}

/// Combined detection + classification result from `POST /analyze`.
class AnalysisResponse {
  final bool success;
  final String detectionModel;
  final String classificationModel;
  final List<Prediction> detections;
  final double detectionTimeMs;
  final ClassificationResult? classification;
  final double classificationTimeMs;
  final String? message;

  /// The top-confidence detection box the classifier was run on (null when no
  /// classification was produced). Lets the UI tie the classification label to
  /// one specific detected object instead of a global label.
  final ClassificationBox? classificationBox;

  /// Uncalibrated area-based density estimate (may be null when no boxes of
  /// the configured weed class were found).
  final DensityEstimate? densityEstimate;

  /// Whether the detector class labels are confirmed. When false, the detections
  /// must not be presented as "weed"/"groundnut" identification.
  final bool labelsConfirmed;
  final String? labelNote;

  /// Machine-readable failure kind: "not_plant_image" | "no_detections" |
  /// "classification_unavailable" | null.
  final String? errorCode;

  AnalysisResponse({
    required this.success,
    required this.detectionModel,
    required this.classificationModel,
    required this.detections,
    required this.detectionTimeMs,
    this.classification,
    required this.classificationTimeMs,
    this.message,
    this.classificationBox,
    this.densityEstimate,
    this.labelsConfirmed = false,
    this.labelNote,
    this.errorCode,
  });

  factory AnalysisResponse.fromJson(Map<String, dynamic> json) {
    final rawDetections = (json['detections'] ?? []) as List;
    final rawClassification = json['classification'] as Map<String, dynamic>?;
    final rawDensity = json['density_estimate'] as Map<String, dynamic>?;
    final rawClassBox = json['classification_box'] as Map<String, dynamic>?;
    return AnalysisResponse(
      success: (json['success'] ?? false) as bool,
      detectionModel: (json['detection_model'] ?? '') as String,
      classificationModel: (json['classification_model'] ?? '') as String,
      detections: rawDetections
          .map((e) => Prediction.fromJson(e as Map<String, dynamic>))
          .toList(),
      detectionTimeMs: (json['detection_time_ms'] as num? ?? 0.0).toDouble(),
      classification: rawClassification != null
          ? ClassificationResult.fromJson(rawClassification)
          : null,
      classificationTimeMs: (json['classification_time_ms'] as num? ?? 0.0).toDouble(),
      message: json['message'] as String?,
      classificationBox: rawClassBox != null
          ? ClassificationBox.fromJson(rawClassBox)
          : null,
      densityEstimate: rawDensity != null
          ? DensityEstimate.fromJson(rawDensity)
          : null,
      labelsConfirmed: (json['labels_confirmed'] ?? false) as bool,
      labelNote: json['label_note'] as String?,
      errorCode: json['error_code'] as String?,
    );
  }
}