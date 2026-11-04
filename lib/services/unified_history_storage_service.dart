import 'package:flutter/foundation.dart';
import '../models/detection_history.dart';
import 'history_storage_service.dart';
import 'local_history_storage_service.dart';

/// Unified history storage service that can switch between local and remote storage
/// This provides a single source of truth for history storage in the app
class UnifiedHistoryStorageService implements HistoryStorageService {
  final HistoryStorageService _localService;
  final HistoryStorageService? _remoteService;
  bool _useRemoteStorage;

  UnifiedHistoryStorageService({
    HistoryStorageService? localService,
    HistoryStorageService? remoteService,
    bool useRemoteStorage = false,
  }) : _localService = localService ?? LocalHistoryStorageService(),
       _remoteService = remoteService,
       _useRemoteStorage = useRemoteStorage {
    // If remote service is provided but not enabled, warn the user
    if (remoteService != null && !useRemoteStorage) {
      debugPrint(
        'Warning: Remote storage service provided but not enabled. Use setStorageMode to enable it.',
      );
    }
  }

  /// Get the currently active storage service
  HistoryStorageService get _activeService {
    return _useRemoteStorage && _remoteService != null
        ? _remoteService
        : _localService;
  }

  /// Switch between local and remote storage
  void setStorageMode(bool useRemote) {
    if (useRemote && _remoteService == null) {
      throw StateError('Remote storage not available');
    }
    _useRemoteStorage = useRemote;
  }

  /// Check if remote storage is available
  bool get isRemoteStorageAvailable => _remoteService != null;

  /// Check if currently using remote storage
  bool get isUsingRemoteStorage => _useRemoteStorage;

  @override
  Future<void> saveDetection(DetectionHistory detection) async {
    await _activeService.saveDetection(detection);
  }

  @override
  Future<List<DetectionHistory>> getAllDetections() async {
    return await _activeService.getAllDetections();
  }

  @override
  Future<DetectionHistory?> getDetectionById(String id) async {
    return await _activeService.getDetectionById(id);
  }

  @override
  Future<void> deleteDetection(String id) async {
    await _activeService.deleteDetection(id);
  }

  @override
  Future<void> deleteAllDetections() async {
    await _activeService.deleteAllDetections();
  }

  @override
  Future<int> getDetectionCount() async {
    return await _activeService.getDetectionCount();
  }

  @override
  Future<List<DetectionHistory>> getDetectionsByDateRange(
    DateTime startDate,
    DateTime endDate,
  ) async {
    return await _activeService.getDetectionsByDateRange(startDate, endDate);
  }

  /// Sync local storage to remote storage (if available)
  Future<void> syncToRemote() async {
    if (_remoteService == null) {
      throw StateError('Remote storage not available');
    }

    final localDetections = await _localService.getAllDetections();

    // Clear remote and save all local detections
    await _remoteService.deleteAllDetections();
    for (final detection in localDetections) {
      await _remoteService.saveDetection(detection);
    }
  }

  /// Sync remote storage to local storage (if available)
  Future<void> syncFromRemote() async {
    if (_remoteService == null) {
      throw StateError('Remote storage not available');
    }

    final remoteDetections = await _remoteService.getAllDetections();

    // Clear local and save all remote detections
    await _localService.deleteAllDetections();
    for (final detection in remoteDetections) {
      await _localService.saveDetection(detection);
    }
  }

  /// Get storage statistics
  Future<Map<String, dynamic>> getStorageStats() async {
    final localDetections = await _localService.getAllDetections();
    final stats = <String, dynamic>{
      'currentMode': _useRemoteStorage ? 'remote' : 'local',
      'localCount': localDetections.length,
      'remoteAvailable': _remoteService != null,
    };

    if (_remoteService != null) {
      final remoteDetections = await _remoteService.getAllDetections();
      stats['remoteCount'] = remoteDetections.length;
    }

    return stats;
  }
}
