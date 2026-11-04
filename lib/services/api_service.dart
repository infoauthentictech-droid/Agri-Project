import 'dart:async' as async;
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../core/api_config.dart';
import '../utils/app_media.dart';
import '../models/detection_result.dart';
import '../models/analysis.dart';
import '../models/recommendation.dart';
import '../models/xai_explanation.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';

/// HTTP client for the Groundnut AI backend.
///
/// Web-safe: no `dart:io` usage. Image files (or blob URLs on web) are read
/// through the conditional-import media helpers and encoded asynchronously,
/// so large phone photos never block the UI isolate while being base64-encoded.
class ApiService {
  final String baseUrl;
  final int timeout;

  /// Longer budget for the heavy combined /analyze and /explain requests
  /// (defaults to [ApiConfig.analyzeTimeout], see its doc comment).
  final int analyzeTimeout;

  ApiService({
    String? customBaseUrl,
    this.timeout = ApiConfig.connectTimeout,
    this.analyzeTimeout = ApiConfig.analyzeTimeout,
  }) : baseUrl = customBaseUrl ?? ApiConfig.baseUrl;

  Map<String, String> get headers => {...ApiConfig.defaultHeaders};

  Future<T> _handleResponse<T>(
    http.Response response,
    T Function(Map<String, dynamic>) fromJson, {
    bool requireDetections = true,
  }) async {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      try {
        final jsonData = json.decode(response.body) as Map<String, dynamic>;

        // Check for low confidence response
        if (jsonData.containsKey('warning') &&
            jsonData['warning'] != null &&
            jsonData['warning'].toString().toLowerCase().contains('confidence')) {
          throw ValidationException(ErrorMessages.lowConfidence);
        }

        // Recommendation-unavailable states (non-fatal, honest messages).
        if (jsonData.containsKey('status') && jsonData['status'] != null) {
          final status = jsonData['status'].toString();
          if (status == 'unknown_weed' || status == 'unsupported_crop') {
            throw ValidationException(ErrorMessages.weedNotFound);
          }
          if (status == 'insufficient_data') {
            throw ValidationException(ErrorMessages.insufficientInformation);
          }
          if (status == 'no_valid_records') {
            // No validated records matched - usually because the knowledge base
            // is still awaiting agronomic review. Surface the backend reason.
            final reason = jsonData['reason'] as String?;
            throw ValidationException(
              reason != null && reason.isNotEmpty
                  ? reason
                  : ErrorMessages.noValidatedRecommendations,
            );
          }
        }

        // Empty-detection guard. The /analyze flow passes requireDetections:
        // false so the designated no-detection screen (and its retake prompt)
        // is reachable instead of an exception.
        if (requireDetections) {
          final predictionList =
              jsonData['predictions'] ?? jsonData['detections'];
          if (predictionList is List && predictionList.isEmpty) {
            throw ValidationException(ErrorMessages.noDetection);
          }
        }

        return fromJson(jsonData);
      } catch (e) {
        if (e is ValidationException) {
          rethrow;
        }
        throw ServerException(ErrorMessages.serverError);
      }
    } else if (response.statusCode == 401) {
      throw AuthenticationException(ErrorMessages.authenticationFailed);
    } else if (response.statusCode == 400) {
      throw ValidationException(ErrorMessages.badRequest);
    } else if (response.statusCode == 404) {
      throw NotFoundException(ErrorMessages.modelUnavailable);
    } else if (response.statusCode == 413) {
      throw ValidationException(ErrorMessages.payloadTooLarge);
    } else if (response.statusCode == 415) {
      throw ValidationException(ErrorMessages.unsupportedMediaType);
    } else if (response.statusCode == 422) {
      throw ValidationException(ErrorMessages.validationError);
    } else if (response.statusCode == 500) {
      throw ServerException(ErrorMessages.serverError);
    } else if (response.statusCode == 503) {
      throw ServerException(ErrorMessages.serviceUnavailable);
    } else if (response.statusCode >= 500) {
      throw ServerException(ErrorMessages.serverError);
    } else {
      throw AppException(ErrorMessages.serverError);
    }
  }

  /// Encode a local image (or a web blob URL) to base64 without a blocking
  /// synchronous full-file read.
  Future<String> _imageToBase64(String path) async =>
      base64Encode(await readMediaBytes(path));

  /// Direct /predict (kept for API compatibility; the app flow uses /analyze).
  Future<DetectionResponse> predictWithImage({
    required String imagePath,
    double confidenceThreshold = 0.5,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.predict}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'confidence_threshold': confidenceThreshold,
            }),
          )
          .timeout(Duration(milliseconds: timeout));
      return await _handleResponse(
          response, (json) => DetectionResponse.fromJson(json));
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }
  /// Direct /detect (kept for API compatibility; the app flow uses /analyze).
  Future<DetectionResponse> detectWithImage({
    required String imagePath,
    double confidenceThreshold = 0.25,
    double iouThreshold = 0.45,
    String? modelName,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.detect}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'confidence_threshold': confidenceThreshold,
              'iou_threshold': iouThreshold,
              'model_name': modelName,
            }),
          )
          .timeout(Duration(milliseconds: timeout));
      return await _handleResponse(
          response, (json) => DetectionResponse.fromJson(json));
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// Direct /classify (kept for API compatibility; the app flow uses /analyze).
  Future<ClassifyResponse> classifyWithImage({
    required String imagePath,
    String? modelName,
    int topK = 3,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.classify}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'model_name': modelName,
              'top_k': topK,
            }),
          )
          .timeout(Duration(milliseconds: timeout));
      return await _handleResponse(
          response,
          (json) => ClassifyResponse.fromJson(json),
          requireDetections: false,
      );
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// Combined /analyze — the main flow used by the app.
  Future<AnalysisResponse> analyzeWithImage({
    required String imagePath,
    String? detectionModel,
    String? classificationModel,
    double confidenceThreshold = 0.25,
    double iouThreshold = 0.45,
    int topK = 3,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.analyze}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'detection_model': detectionModel,
              'classification_model': classificationModel,
              'confidence_threshold': confidenceThreshold,
              'iou_threshold': iouThreshold,
              'top_k': topK,
            }),
          )
          // /analyze runs several models in sequence (plant gate, YOLO,
          // density, classifier) and may lazily load some of them on first
          // use, so it gets its own longer timeout instead of the 30 s default.
          .timeout(Duration(milliseconds: analyzeTimeout));

      // requireDetections: false — an empty detection list is a legitimate
      // response that the result screen renders as its no-detection view.
      return await _handleResponse(
        response,
        (json) => AnalysisResponse.fromJson(json),
        requireDetections: false,
      );
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// POST /plant_gate — is this a crop-field photo? Runs before model selection.
  Future<PlantGateResult> checkPlantImage({required String imagePath}) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.plantGate}'),
            headers: headers,
            body: json.encode({'image_data': await _imageToBase64(imagePath)}),
          )
          // The gate's ImageNet featurizer may load lazily on first use.
          .timeout(Duration(milliseconds: analyzeTimeout));
      return await _handleResponse(
        response,
        (json) => PlantGateResult.fromJson(json),
        requireDetections: false,
      );
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// GET /models — the model registry for the selection screen.
  Future<ModelCatalog> getModels() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl${ApiConfig.models}'), headers: headers)
          .timeout(Duration(milliseconds: timeout));

      return await _handleResponse(
        response,
        (json) => ModelCatalog.fromJson(json),
        requireDetections: false,
      );
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    }
  }
  /// Get XAI explanation with image (as base64 JSON).
  Future<ExplanationResponse> explainWithImage({
    required String imagePath,
    bool includeVisualization = true,
  }) async {
    try {
      if (!await mediaExists(imagePath)) {
        throw ValidationException(ErrorMessages.noImageSelected);
      }

      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.explain}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'include_visualization': includeVisualization,
            }),
          )
          // Grad-CAM + visualization is as heavy as /analyze (XAI engine runs
          // a second forward pass), so it uses the same extended timeout.
          .timeout(Duration(milliseconds: analyzeTimeout));

      return await _handleResponse(
        response,
        (json) => ExplanationResponse.fromJson(json),
        requireDetections: false,
      );
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// Get herbicide recommendation
  Future<RecommendationResponse> getRecommendation({
    required String crop,
    String? weedSpecies,
    String? weedCategory,
    String? cropGrowthStage,
    String? season,
    String? region,
    String? weedDensity,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.recommend}'),
            headers: headers,
            body: json.encode({
              'crop': crop,
              'weed_species': weedSpecies,
              'weed_category': weedCategory,
              'crop_growth_stage': cropGrowthStage,
              'season': season,
              'region': region,
              'weed_density': weedDensity,
            }),
          )
          .timeout(Duration(milliseconds: timeout));

      return await _handleResponse(
          response, (json) => RecommendationResponse.fromJson(json));
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }

  /// Health check to verify backend is available
  Future<bool> isBackendAvailable() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl${ApiConfig.health}'),
            headers: headers,
          )
          .timeout(Duration(milliseconds: timeout));

      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (e) {
      return false;
    }
  }

  /// Semantic segmentation endpoint
  Future<Map<String, dynamic>> segmentWithImage({
    required String imagePath,
    int targetSize = 224,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl${ApiConfig.segment}'),
            headers: headers,
            body: json.encode({
              'image_data': await _imageToBase64(imagePath),
              'target_size': targetSize,
            }),
          )
          .timeout(Duration(milliseconds: timeout));

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final jsonData = json.decode(response.body) as Map<String, dynamic>;
        return jsonData;
      } else {
        throw ServerException('Segmentation failed: ${response.statusCode}');
      }
    } on http.ClientException {
      throw NetworkException(ErrorMessages.networkUnavailable);
    } on async.TimeoutException {
      throw TimeoutException(ErrorMessages.requestTimeout);
    } on FormatException {
      throw ServerException(ErrorMessages.inferenceFailure);
    }
  }
}
