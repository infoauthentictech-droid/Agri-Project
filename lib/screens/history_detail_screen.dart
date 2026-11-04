import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../widgets/app_card.dart';
import '../widgets/app_button.dart';
import '../widgets/history_image.dart';
import '../models/detection_history.dart';
import '../services/local_history_storage_service.dart';

class HistoryDetailScreen extends StatefulWidget {
  const HistoryDetailScreen({super.key});

  @override
  State<HistoryDetailScreen> createState() => _HistoryDetailScreenState();
}

class _HistoryDetailScreenState extends State<HistoryDetailScreen> {
  final LocalHistoryStorageService _storageService = LocalHistoryStorageService();
  DetectionHistory? _detection;
  bool _isLoading = true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (args != null && args.containsKey('detectionId')) {
      _loadDetection(args['detectionId'] as String);
    }
  }

  Future<void> _loadDetection(String id) async {
    setState(() {
      _isLoading = true;
    });

    final detection = await _storageService.getDetectionById(id);

    if (mounted) {
      setState(() {
        _detection = detection;
        _isLoading = false;
      });
    }

    if (detection == null && mounted) {
      Navigator.pop(context);
    }
  }

  String _formatDateTime(DateTime dateTime) {
    return '${dateTime.day}/${dateTime.month}/${dateTime.year} at '
        '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
  }

  Future<void> _deleteDetection() async {
    if (_detection == null) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Detection'),
        content: const Text('Are you sure you want to delete this detection record?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: AppTheme.error),
            child: const Text('Delete'),
          ),
        ],
      ),
    ).then((value) => value ?? false);

    if (confirmed) {
      await _storageService.deleteDetection(_detection!.id);
      if (mounted) {
        Navigator.pop(context);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Details'),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: _deleteDetection,
            tooltip: 'Delete',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _detection == null
              ? const Center(child: Text('Detection not found'))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Image
                      Container(
                        width: double.infinity,
                        height: 250,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          color: AppTheme.surfaceColor,
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: HistoryImage(
                          imageData: _detection!.imageData,
                          imagePath: _detection!.imagePath,
                          fit: BoxFit.cover,
                        ),
                      ),
                      const SizedBox(height: 24),

                      // Detection Info Card
                      AppCard(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Detection Information',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 16),
                            _buildInfoRow('Date/Time', _formatDateTime(_detection!.dateTime)),
                            const SizedBox(height: 12),
                            _buildInfoRow('Detected Class', _detection!.detectedClass),
                            if (_detection!.weedSpecies != null) ...[
                              const SizedBox(height: 12),
                              _buildInfoRow('Weed Species', _detection!.weedSpecies!),
                            ],
                            const SizedBox(height: 12),
                            _buildInfoRow(
                              'Confidence',
                              '${(_detection!.confidence * 100).toStringAsFixed(1)}%',
                            ),
                            const SizedBox(height: 12),
                            _buildInfoRow('Weed Density', _detection!.weedDensity),
                            const SizedBox(height: 12),
                            _buildInfoRow(
                              'Recommendation Available',
                              _detection!.hasRecommendation ? 'Yes' : 'No',
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),

                      // Action Buttons
                      Row(
                        children: [
                          Expanded(
                            child: AppButton(
                              text: 'Close',
                              icon: Icons.close,
                              onPressed: () => Navigator.pop(context),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 140,
          child: Text(
            '$label:',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: AppTheme.textSecondary,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }
}
