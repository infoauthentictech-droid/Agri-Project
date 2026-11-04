import 'dart:convert';
import 'dart:typed_data' show Uint8List;
import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../widgets/app_card.dart';
import '../utils/app_media.dart';
import '../widgets/app_button.dart';
import '../services/api_service.dart';
import '../services/image_service.dart';
import '../models/xai_explanation.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';
import '../utils/loading_state.dart';
import '../utils/error_state.dart';

class XAIScreen extends StatefulWidget {
  const XAIScreen({super.key});

  @override
  State<XAIScreen> createState() => _XAIScreenState();
}

class _XAIScreenState extends State<XAIScreen> {
  String? _imagePath;
  bool _isLoading = false;
  bool _hasError = false;
  String? _errorMessage;
  ExplanationResponse? _explanation;
  final ApiService _apiService = ApiService();
  final ImageService _imageService = ImageService();
  bool _showHeatmap = false;
  bool _isPickingImage = false;
  final TransformationController _transformationController = TransformationController();

  /// Guards against re-running the (expensive) explain request every time the
  /// route dependencies change, e.g. when the user returns to this screen.
  bool _explainStarted = false;

  /// Decoded Grad-CAM visuals embedded in the backend response
  /// (`visualizations.overlay_0` / `visualizations.heatmap_0`).
  Uint8List? _overlayBytes;
  Uint8List? _heatmapBytes;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (args != null && args.containsKey('imagePath')) {
      setState(() {
        _imagePath = args['imagePath'] as String?;
      });
    }
    // Auto-run the explanation once when an image path is provided. Don't
    // auto-run when only a detectionResponse is provided without an image.
    if (_imagePath != null && !_explainStarted) {
      _explainStarted = true;
      _runExplanation();
    }
  }

  /// Let the user pick an image directly on this screen, so the Home-screen
  /// entry works even when no result screen precedes it.
  Future<void> _pickImageFromDevice() async {
    if (_isPickingImage) return;
    setState(() {
      _isPickingImage = true;
      _hasError = false;
      _errorMessage = null;
    });
    try {
      final path = await _imageService.pickImageFromGallery();
      if (path == null) return;
      if (mounted) {
        setState(() {
          _imagePath = path;
          _explanation = null;
          _explainStarted = false;
        });
      }
      await _runExplanation();
    } on PermissionDeniedException catch (e) {
      if (mounted) {
        setState(() {
          _hasError = true;
          _errorMessage = e.message;
        });
      }
    } on AppException catch (e) {
      if (mounted) {
        setState(() {
          _hasError = true;
          _errorMessage = e.message;
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isPickingImage = false;
        });
      }
    }
  }

  Future<void> _runExplanation() async {
    if (_imagePath == null) {
      if (mounted) {
        setState(() {
          _hasError = true;
          _errorMessage = 'Please choose an image first (gallery or camera).';
        });
      }
      return;
    }

    setState(() {
      _isLoading = true;
      _hasError = false;
      _errorMessage = null;
    });

    try {
      final response = await _apiService.explainWithImage(
        imagePath: _imagePath!,
        includeVisualization: true,
      );

      // Decode the embedded Grad-CAM visuals (base64) so the toggle can render
      // them without a second round-trip to the server.
      Uint8List? overlayBytes;
      Uint8List? heatmapBytes;
      final overlay = response.visualizations['overlay_0'];
      final heatmap = response.visualizations['heatmap_0'];
      try {
        if (overlay != null && overlay.isNotEmpty) {
          overlayBytes = base64Decode(overlay);
        }
      } catch (_) {
        overlayBytes = null;
      }
      try {
        if (heatmap != null && heatmap.isNotEmpty) {
          heatmapBytes = base64Decode(heatmap);
        }
      } catch (_) {
        heatmapBytes = null;
      }

      if (mounted) {
        setState(() {
          _isLoading = false;
          _explanation = response;
          _overlayBytes = overlayBytes;
          _heatmapBytes = heatmapBytes;
          // Show the heatmap overlay automatically when one is available.
          if (!_showHeatmap && _overlayBytes != null) {
            _showHeatmap = true;
          }
        });
      }
    } on NetworkException {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.networkUnavailable;
        });
      }
    } on ServerException {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.serverError;
        });
      }
    } on ValidationException catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = ErrorMessages.unexpectedError;
        });
      }
    }
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Explainable AI'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_isLoading)
              const LoadingWidget(message: 'Generating explanation...')
            else if (_hasError)
              ErrorStateWidget(
                message: 'XAI Unavailable',
                subMessage: _errorMessage,
                onRetry: _runExplanation,
              )
            else if (_explanation == null)
              _buildInfoView()
            else
              _buildExplanationView(),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(
          Icons.psychology_outlined,
          size: 64,
          color: AppTheme.primaryGreen.withValues(alpha: 0.5),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        Text(
          'Explainable AI',
          style: Theme.of(context).textTheme.displaySmall?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: AppTheme.spacingXs),
        Text(
          'Understanding how the AI makes decisions',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            color: AppTheme.textSecondary,
          ),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        AppCard(
          color: AppTheme.primaryGreen.withValues(alpha: 0.1),
          borderColor: AppTheme.primaryGreen,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.info_outline, color: AppTheme.primaryGreen),
                  const SizedBox(width: AppTheme.spacingXs),
                  Text(
                    'About XAI',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: AppTheme.primaryGreen,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: AppTheme.spacingSm),
              Text(
                'Explainable AI helps you understand which parts of the image '
                'influenced the model\'s decision. This transparency builds trust '
                'and helps verify that the AI is focusing on the right features.',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        AppButton(
          text: _isPickingImage ? 'Opening Gallery...' : 'Choose an Image',
          icon: Icons.photo_library,
          onPressed: _pickImageFromDevice,
        ),
        const SizedBox(height: AppTheme.spacingSm),
        AppButton(
          text: 'Generate Explanation',
          isSecondary: true,
          icon: Icons.play_arrow,
          onPressed: _runExplanation,
        ),
        const SizedBox(height: AppTheme.spacingSm),
        Text(
          'You can also enter this screen from a detection result, which '
          'passes the analysed image automatically.',
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: AppTheme.textHint,
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildExplanationView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Image with Heatmap Toggle
        if (_imagePath != null) ...[
          _buildImageSection(),
          const SizedBox(height: AppTheme.spacingLg),
        ],
        
        // Explanation Details
        if (_explanation != null) ...[
          _buildExplanationDetails(),
          const SizedBox(height: AppTheme.spacingLg),
        ],
        
        // Action Buttons
        Row(
          children: [
            Expanded(
              child: AppButton(
                text: 'New Image',
                icon: Icons.camera_alt,
                onPressed: () {
                  Navigator.popUntil(context, (route) => route.isFirst);
                },
              ),
            ),
            const SizedBox(width: AppTheme.spacingMd),
            Expanded(
              child: AppButton(
                text: 'Back',
                isSecondary: true,
                onPressed: () => Navigator.pop(context),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildImageSection() {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Image Visualization',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              Switch(
                value: _showHeatmap,
                onChanged: (value) {
                  setState(() {
                    _showHeatmap = value;
                  });
                },
                activeTrackColor: AppTheme.primaryGreen.withValues(alpha: 0.5),
                thumbColor: WidgetStateProperty.resolveWith((states) {
                  if (states.contains(WidgetState.selected)) {
                    return AppTheme.primaryGreen;
                  }
                  return Colors.grey;
                }),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingSm),
          Text(
            _showHeatmap && _overlayBytes != null
                ? _isFallbackExplanation()
                    ? 'Activation Overlay (not Grad-CAM)'
                    : 'Grad-CAM Overlay'
                : _showHeatmap && _heatmapBytes != null
                    ? _isFallbackExplanation()
                        ? 'Activation Heatmap (not Grad-CAM)'
                        : 'Grad-CAM Heatmap'
                    : 'Original Image',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: AppTheme.spacingSm),
          Container(
            height: 300,
            width: double.infinity,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
              border: Border.all(color: AppTheme.textHint.withValues(alpha: 0.3)),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
              child: _showHeatmap && _overlayBytes != null
                  ? Image.memory(_overlayBytes!, fit: BoxFit.contain)
                  : _showHeatmap && _heatmapBytes != null
                      ? Image.memory(_heatmapBytes!, fit: BoxFit.contain)
                      : localImage(_imagePath!, fit: BoxFit.contain),
            ),
          ),
        ],
      ),
    );
  }

  bool _isFallbackExplanation() {
    final explanations = _explanation?.explanations ?? [];
    return explanations.isNotEmpty &&
        explanations.every((e) => e.method == 'fallback_activation');
  }

  Widget _buildExplanationDetails() {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.info_outline, color: AppTheme.primaryGreen),
              const SizedBox(width: AppTheme.spacingXs),
              Text(
                'Explanation Details',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingSm),
          if (_isFallbackExplanation())
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(AppTheme.spacingSm),
              decoration: BoxDecoration(
                color: AppTheme.warning.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                border: Border.all(color: AppTheme.warning),
              ),
              child: const Text(
                'The gradient-based explanation could not be generated for '
                'this image. The overlay shown is a non-gradient activation '
                'emphasis — it is NOT a model-gradient Grad-CAM map.',
                style: TextStyle(color: AppTheme.warning, fontSize: 12),
              ),
            ),
          if (_isFallbackExplanation()) const SizedBox(height: AppTheme.spacingSm),
          if (_explanation!.explanations.isNotEmpty) ...[
            ..._explanation!.explanations.map((explanation) => Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildInfoRow('Prediction', explanation.prediction),
                const SizedBox(height: AppTheme.spacingXxs),
                if (explanation.explanationText.isNotEmpty) ...[
                  Text(
                    explanation.explanationText,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppTheme.textSecondary,
                    ),
                  ),
                  const SizedBox(height: AppTheme.spacingSm),
                ],
              ],
            )),
          ],
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 120,
          child: Text(
            '$label:',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: AppTheme.textSecondary,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }
}
