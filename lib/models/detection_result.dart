class BoundingBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  
  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });
  
  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: (json['x1'] as num).toDouble(),
      y1: (json['y1'] as num).toDouble(),
      x2: (json['x2'] as num).toDouble(),
      y2: (json['y2'] as num).toDouble(),
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'x1': x1,
      'y1': y1,
      'x2': x2,
      'y2': y2,
    };
  }
}

class Prediction {
  final int classId;
  final String className;
  final double confidence;
  final BoundingBox boundingBox;
  final String? segmentationMask; // Base64 encoded PNG mask
  
  Prediction({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.boundingBox,
    this.segmentationMask,
  });
  
  factory Prediction.fromJson(Map<String, dynamic> json) {
    return Prediction(
      classId: json['class_id'] as int,
      className: json['class_name'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      boundingBox: BoundingBox.fromJson(json['bounding_box'] as Map<String, dynamic>),
      segmentationMask: json['segmentation_mask'] as String?,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'class_id': classId,
      'class_name': className,
      'confidence': confidence,
      'bounding_box': boundingBox.toJson(),
      if (segmentationMask != null) 'segmentation_mask': segmentationMask,
    };
  }
}

class DetectionResponse {
  final bool success;
  final List<Prediction> predictions;
  final int numPredictions;
  final double inferenceTimeMs;
  final String? message;
  final String? segmentationMask; // Base64 encoded PNG full-image mask
  final Map<String, double>? coverage; // Coverage percentage per class
  
  DetectionResponse({
    required this.success,
    required this.predictions,
    required this.numPredictions,
    required this.inferenceTimeMs,
    this.message,
    this.segmentationMask,
    this.coverage,
  });
  
  factory DetectionResponse.fromJson(Map<String, dynamic> json) {
    // The backend returns detections for /detect and predictions for /predict
    // (see backend/app/schemas/*.py). Accept both key variants so the two
    // endpoints share this model.
    final rawList = (json['predictions'] ?? json['detections']) as List;
    final predictionsList = rawList.map((e) => e as Map<String, dynamic>).toList();
    final rawCount = (json['num_predictions'] ?? json['num_detections']);
    return DetectionResponse(
      success: json['success'] as bool,
      predictions: predictionsList
          .map((e) => Prediction.fromJson(e))
          .toList(),
      numPredictions: (rawCount as num).toInt(),
      inferenceTimeMs: (json['inference_time_ms'] as num).toDouble(),
      message: json['message'] as String?,
      segmentationMask: json['segmentation_mask'] as String?,
      coverage: json['coverage'] as Map<String, double>?,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'predictions': predictions.map((e) => e.toJson()).toList(),
      'num_predictions': numPredictions,
      'inference_time_ms': inferenceTimeMs,
      'message': message,
      if (segmentationMask != null) 'segmentation_mask': segmentationMask,
      if (coverage != null) 'coverage': coverage,
    };
  }
}
