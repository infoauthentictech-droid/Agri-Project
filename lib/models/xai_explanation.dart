class Explanation {
  final String prediction;
  final String explanationText;
  final String visualizationDescription;
  final List<String> limitations;

  /// Which explanation method produced the visualization:
  /// `gradcam` (real, gradient-computed) or `fallback_activation` (non-gradient
  /// emphasis, clearly not Grad-CAM).
  final String method;

  Explanation({
    required this.prediction,
    required this.explanationText,
    required this.visualizationDescription,
    required this.limitations,
    this.method = 'gradcam',
  });

  factory Explanation.fromJson(Map<String, dynamic> json) {
    final limitationsList = json['limitations'] as List;
    return Explanation(
      prediction: json['prediction'] as String,
      explanationText: json['explanation_text'] as String,
      visualizationDescription: json['visualization_description'] as String,
      limitations: limitationsList.map((e) => e as String).toList(),
      method: (json['method'] ?? 'gradcam') as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'prediction': prediction,
      'explanation_text': explanationText,
      'visualization_description': visualizationDescription,
      'limitations': limitations,
      'method': method,
    };
  }
}

class ExplanationResponse {
  final bool success;
  final List<Map<String, dynamic>> predictions;
  final List<Explanation> explanations;
  final Map<String, String> savedFiles;
  final String? message;

  /// Base64-encoded Grad-CAM visualizations returned by the backend (keys:
  /// `overlay_0`, `heatmap_0`, ...). The mobile client cannot reach the
  /// server-side files, so the backend embeds the images directly in the JSON
  /// payload (see `backend/app/services/xai_service.py`).
  final Map<String, String> visualizations;

  ExplanationResponse({
    required this.success,
    required this.predictions,
    required this.explanations,
    required this.savedFiles,
    this.visualizations = const {},
    this.message,
  });

  factory ExplanationResponse.fromJson(Map<String, dynamic> json) {
    final explanationsList = (json['explanations'] ?? []) as List;
    final savedFilesMap = (json['saved_files'] ?? {}) as Map<String, dynamic>;
    final visualizationsMap =
        (json['visualizations'] ?? {}) as Map<String, dynamic>;

    return ExplanationResponse(
      success: json['success'] as bool,
      predictions: (json['predictions'] ?? [])
          .map((e) => e as Map<String, dynamic>)
          .toList(),
      explanations: explanationsList
          .map((e) => Explanation.fromJson(e as Map<String, dynamic>))
          .toList(),
      savedFiles: savedFilesMap.map((k, v) => MapEntry(k, v as String)),
      visualizations:
          visualizationsMap.map((k, v) => MapEntry(k, v as String)),
      message: json['message'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'predictions': predictions,
      'explanations': explanations.map((e) => e.toJson()).toList(),
      'saved_files': savedFiles,
      'visualizations': visualizations,
      'message': message,
    };
  }
}
