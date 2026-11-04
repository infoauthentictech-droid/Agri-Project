import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_button.dart';
import '../utils/app_media.dart';
import '../services/api_service.dart';
import '../services/image_service.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';

class ImagePreviewScreen extends StatefulWidget {
  const ImagePreviewScreen({super.key});

  @override
  State<ImagePreviewScreen> createState() => _ImagePreviewScreenState();
}

class _ImagePreviewScreenState extends State<ImagePreviewScreen> {
  String? _imagePath;
  final ImageService _imageService = ImageService();
  final ApiService _apiService = ApiService();
  bool _isCropping = false;

  /// True while POST /plant_gate is checking the photo.
  bool _isChecking = false;

  /// Set when the plant gate rejected the current photo; the screen then
  /// replaces Analyze with a "not a plant image" message and a Retake button.
  String? _notPlantMessage;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (_imagePath == null && args != null) {
      _imagePath = args['imagePath'] as String;
    }
  }

  /// Back to the camera / gallery / home screen that opened this preview.
  void _retake() => Navigator.pop(context);

  /// Check the photo with the plant gate first: plant photos continue to model
  /// selection, anything else shows the "not a plant image" state.
  Future<void> _analyzeImage() async {
    if (_imagePath == null || _isChecking) return;
    final messenger = ScaffoldMessenger.of(context);
    setState(() => _isChecking = true);

    try {
      final gate = await _apiService.checkPlantImage(imagePath: _imagePath!);
      if (!mounted) return;
      if (gate.isPlant) {
        Navigator.pushNamed(
          context,
          AppConstants.modelSelectionScreen,
          arguments: {'imagePath': _imagePath},
        );
      } else {
        setState(() {
          _notPlantMessage = gate.reasons.isNotEmpty
              ? gate.reasons.first
              : ErrorMessages.notPlantImage;
        });
      }
    } on AppException catch (e) {
      messenger.showSnackBar(
        SnackBar(content: Text(e.message), backgroundColor: AppTheme.error),
      );
    } catch (_) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(ErrorMessages.unexpectedError),
          backgroundColor: AppTheme.error,
        ),
      );
    } finally {
      if (mounted) setState(() => _isChecking = false);
    }
  }

  /// Crop the selected image (native only) so the cropped result becomes the
  /// image that flows into model selection and analysis. This wires CropService
  /// into the normal capture → crop → analyze journey.
  Future<void> _cropImage() async {
    if (_imagePath == null || _isCropping) return;
    setState(() => _isCropping = true);
    try {
      final croppedPath = await _imageService.cropImage(imagePath: _imagePath!);
      if (croppedPath == null || !mounted) return;
      setState(() {
        _imagePath = croppedPath;
        // A new crop is a new photo: let the gate judge it again.
        _notPlantMessage = null;
      });
    } on AppException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.message), backgroundColor: AppTheme.error),
      );
    } finally {
      if (mounted) setState(() => _isCropping = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Preview'),
        actions: [
          IconButton(
            icon: const Icon(Icons.close),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
      body: _imagePath != null
          ? Column(
              children: [
                Expanded(
                  child: localImage(_imagePath!, fit: BoxFit.contain),
                ),
                Container(
                  padding: const EdgeInsets.all(16),
                  child: _notPlantMessage != null
                      ? _buildNotPlantPanel(context)
                      : _buildActions(),
                ),
              ],
            )
          : const Center(
              child: CircularProgressIndicator(),
            ),
    );
  }

  Widget _buildActions() {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: AppButton(
                text: 'Crop',
                icon: Icons.crop,
                onPressed: _isChecking ? null : _cropImage,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: AppButton(
                text: 'Retake',
                isSecondary: true,
                onPressed: _isChecking ? null : _retake,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: AppButton(
            text: _isChecking ? 'Checking photo...' : 'Analyze',
            icon: Icons.analytics,
            isLoading: _isChecking,
            onPressed: _analyzeImage,
          ),
        ),
      ],
    );
  }

  Widget _buildNotPlantPanel(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.all(AppTheme.spacingSm),
          decoration: BoxDecoration(
            color: AppTheme.warning.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            border: Border.all(color: AppTheme.warning),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.warning_amber, color: AppTheme.warning),
              const SizedBox(width: AppTheme.spacingXs),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Not a plant image',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                    ),
                    const SizedBox(height: AppTheme.spacingXxs),
                    Text(
                      _notPlantMessage!,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: AppTheme.textSecondary,
                          ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        AppButton(
          text: 'Retake',
          icon: Icons.camera_alt,
          onPressed: _retake,
        ),
      ],
    );
  }
}
