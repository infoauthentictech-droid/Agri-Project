import '../models/detection_history.dart';

/// Abstract interface for detection history storage
/// This allows for easy replacement with server-side database later
abstract class HistoryStorageService {
  /// Save a detection history record
  Future<void> saveDetection(DetectionHistory detection);
  
  /// Get all detection history records
  Future<List<DetectionHistory>> getAllDetections();
  
  /// Get a specific detection by ID
  Future<DetectionHistory?> getDetectionById(String id);
  
  /// Delete a specific detection by ID
  Future<void> deleteDetection(String id);
  
  /// Delete all detection history
  Future<void> deleteAllDetections();
  
  /// Get detection count
  Future<int> getDetectionCount();
  
  /// Get detections within a date range
  Future<List<DetectionHistory>> getDetectionsByDateRange(
    DateTime startDate,
    DateTime endDate,
  );
}
