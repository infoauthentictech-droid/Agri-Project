class DetectionHistory {
  final String id;
  final DateTime dateTime;
  final String imagePath;

  /// Optional base64-encoded embedded copy of the image. Kept so history
  /// records survive even when the original temp file is cleaned up by the OS
  /// (and so web builds, which have no local filesystem, can render it).
  final String? imageData;

  final String detectedClass;
  final String? weedSpecies;
  final double confidence;
  final String weedDensity;
  final bool hasRecommendation;
  final String? recommendationId;

  DetectionHistory({
    required this.id,
    required this.dateTime,
    required this.imagePath,
    this.imageData,
    required this.detectedClass,
    this.weedSpecies,
    required this.confidence,
    required this.weedDensity,
    required this.hasRecommendation,
    this.recommendationId,
  });

  factory DetectionHistory.fromJson(Map<String, dynamic> json) {
    return DetectionHistory(
      id: json['id'] as String,
      dateTime: DateTime.parse(json['dateTime'] as String),
      imagePath: json['imagePath'] as String,
      imageData: json['imageData'] as String?,
      detectedClass: json['detectedClass'] as String,
      weedSpecies: json['weedSpecies'] as String?,
      confidence: (json['confidence'] as num).toDouble(),
      weedDensity: json['weedDensity'] as String,
      hasRecommendation: json['hasRecommendation'] as bool,
      recommendationId: json['recommendationId'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'dateTime': dateTime.toIso8601String(),
      'imagePath': imagePath,
      'imageData': imageData,
      'detectedClass': detectedClass,
      'weedSpecies': weedSpecies,
      'confidence': confidence,
      'weedDensity': weedDensity,
      'hasRecommendation': hasRecommendation,
      'recommendationId': recommendationId,
    };
  }

  DetectionHistory copyWith({
    String? id,
    DateTime? dateTime,
    String? imagePath,
    String? imageData,
    String? detectedClass,
    String? weedSpecies,
    double? confidence,
    String? weedDensity,
    bool? hasRecommendation,
    String? recommendationId,
  }) {
    return DetectionHistory(
      id: id ?? this.id,
      dateTime: dateTime ?? this.dateTime,
      imagePath: imagePath ?? this.imagePath,
      imageData: imageData ?? this.imageData,
      detectedClass: detectedClass ?? this.detectedClass,
      weedSpecies: weedSpecies ?? this.weedSpecies,
      confidence: confidence ?? this.confidence,
      weedDensity: weedDensity ?? this.weedDensity,
      hasRecommendation: hasRecommendation ?? this.hasRecommendation,
      recommendationId: recommendationId ?? this.recommendationId,
    );
  }
}
