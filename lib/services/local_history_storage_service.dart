import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';
import '../models/detection_history.dart';
import 'history_storage_service.dart';

/// Local storage implementation using SharedPreferences
/// This can be replaced with a server-side database implementation later
class LocalHistoryStorageService implements HistoryStorageService {
  static const String _storageKey = 'detection_history';
  final Uuid _uuid = const Uuid();

  @override
  Future<void> saveDetection(DetectionHistory detection) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getAllDetections();
    
    // Add new detection to the beginning of the list
    history.insert(0, detection);
    
    // Keep only the last 100 records to prevent storage bloat
    if (history.length > 100) {
      history.removeRange(100, history.length);
    }
    
    // Only the newest records keep their embedded thumbnail, so the single
    // SharedPreferences string (localStorage on web, ~5 MB quota) stays small.
    final jsonString = jsonEncode([
      for (var i = 0; i < history.length; i++)
        {
          ...history[i].toJson(),
          if (i >= _maxEmbeddedImages) 'imageData': null,
        },
    ]);
    await prefs.setString(_storageKey, jsonString);
  }

  static const int _maxEmbeddedImages = 20;

  @override
  Future<List<DetectionHistory>> getAllDetections() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonString = prefs.getString(_storageKey);
    
    if (jsonString == null) {
      return [];
    }
    
    try {
      final List<dynamic> jsonList = jsonDecode(jsonString);
      return jsonList
          .map((json) => DetectionHistory.fromJson(json as Map<String, dynamic>))
          .toList();
    } catch (e) {
      // If there's an error parsing, return empty list
      return [];
    }
  }

  @override
  Future<DetectionHistory?> getDetectionById(String id) async {
    final detections = await getAllDetections();
    try {
      return detections.firstWhere((d) => d.id == id);
    } catch (e) {
      return null;
    }
  }

  @override
  Future<void> deleteDetection(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getAllDetections();
    
    history.removeWhere((d) => d.id == id);
    
    final jsonString = jsonEncode(
      history.map((d) => d.toJson()).toList(),
    );
    await prefs.setString(_storageKey, jsonString);
  }

  @override
  Future<void> deleteAllDetections() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }

  @override
  Future<int> getDetectionCount() async {
    final detections = await getAllDetections();
    return detections.length;
  }

  @override
  Future<List<DetectionHistory>> getDetectionsByDateRange(
    DateTime startDate,
    DateTime endDate,
  ) async {
    final detections = await getAllDetections();
    return detections.where((d) {
      return d.dateTime.isAfter(startDate) && d.dateTime.isBefore(endDate);
    }).toList();
  }

  /// Create a new detection history record from detection data
  DetectionHistory createDetectionRecord({
    required String imagePath,
    String? imageData,
    required String detectedClass,
    String? weedSpecies,
    required double confidence,
    required String weedDensity,
    bool hasRecommendation = false,
    String? recommendationId,
  }) {
    return DetectionHistory(
      id: _uuid.v4(),
      dateTime: DateTime.now(),
      imagePath: imagePath,
      imageData: imageData,
      detectedClass: detectedClass,
      weedSpecies: weedSpecies,
      confidence: confidence,
      weedDensity: weedDensity,
      hasRecommendation: hasRecommendation,
      recommendationId: recommendationId,
    );
  }
}
