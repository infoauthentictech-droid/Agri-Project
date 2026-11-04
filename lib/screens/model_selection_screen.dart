import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../utils/app_media.dart';
import '../widgets/app_button.dart';
import '../widgets/app_card.dart';
import '../services/api_service.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';
import '../utils/error_state.dart';
import '../utils/loading_state.dart';
import '../utils/empty_state.dart';
import '../models/analysis.dart';

/// Screen shown right after a photo is captured or selected.
///
/// Lets the user choose the **classification** model from the models the
/// backend actually has (from `GET /models`). The detection model is
/// auto-selected to the server default (the backend falls back to it when the
/// app does not pick one). Analysis runs via `POST /analyze`, then the result
/// screen renders the combined output.
class ModelSelectionScreen extends StatefulWidget {
  const ModelSelectionScreen({super.key});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  String? _imagePath;
  final ApiService _apiService = ApiService();

  ModelCatalog? _catalog;
  bool _isLoadingModels = false;
  bool _hasError = false;
  String? _errorMessage;

  String? _selectedClassificationModel;
  bool _isAnalyzing = false;

  /// Ensures the catalog is fetched exactly once per screen instance. Without
  /// this guard, every rebuild re-fired _loadModels() which repeatedly probed
  /// the backend and could leave the page trapped in a loading/error loop.
  bool _autoLoadStarted = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (_imagePath == null && args != null && args.containsKey('imagePath')) {
      _imagePath = args['imagePath'] as String;
    }
    if (!_autoLoadStarted) {
      _autoLoadStarted = true;
      _loadModels();
    }
  }

  Future<void> _loadModels() async {
    if (_catalog != null || _isLoadingModels) return;
    setState(() {
      _isLoadingModels = true;
      _hasError = false;
      _errorMessage = null;
    });

    try {
      // Single round-trip (GET /models): the typed catches below map every
      // failure to a clear message, and _fail always resets the loading flag.
      final catalog = await _apiService.getModels();
      if (mounted) {
        setState(() {
          _catalog = catalog;
          _isLoadingModels = false;
          // Only the classifier is presented to the user for selection.
          _selectedClassificationModel =
              _firstAvailable(catalog);
        });
      }
    } on AuthenticationException catch (e) {
      _fail('Cannot connect securely to the server.',
          details: e.message);
    } on NetworkException catch (e) {
      _fail('Cannot reach the server.',
          details: '${e.message}\n'
              'Start the project with run_project.ps1 on the computer and '
              'keep the phone connected by USB or on the same Wi-Fi.');
    } on TimeoutException catch (e) {
      _fail('The server took too long to respond.',
          details: '${e.message}\nCheck the connection and try again.');
    } on ServerException catch (e) {
      _fail('The server reported an error.',
          details: '${e.message}\nTry again in a moment.');
    } on FormatException catch (_) {
      _fail('The server sent an unexpected response.',
          details: 'Try again in a moment.');
    } catch (e) {
      _fail('Unexpected error: $e', details: 'Try again in a moment.');
    }
  }

  String? _firstAvailable(ModelCatalog catalog) {
    final list = catalog.classificationModels;
    final defaults = catalog.defaultClassificationModel;
    for (final option in list) {
      if (option.available && option.name == defaults) return option.name;
    }
    for (final option in list) {
      if (option.available) return option.name;
    }
    return null;
  }

  void _fail(String headline, {String? details}) {
    final combined = (details == null || details.isEmpty)
        ? headline
        : '$headline\n$details';
    // Always reset the flags; setState only when mounted (the framework
    // requires it), otherwise the next rebuild picks up the same fields.
    _isLoadingModels = false;
    _hasError = true;
    _errorMessage = combined;
    if (mounted) {
      setState(() {
        _isLoadingModels = false;
        _hasError = true;
        _errorMessage = combined;
      });
    }
  }

  Future<void> _runAnalysis() async {
    if (_imagePath == null || _isAnalyzing) return;
    // Only the classifier is required from the user.
    final classificationModel = _selectedClassificationModel;
    // Capture the messenger before any async gap (avoids re-entering context).
    final messenger = ScaffoldMessenger.of(context);
    if (classificationModel == null) {
      messenger.showSnackBar(
        const SnackBar(content: Text('Please choose a classification model first')),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
    });

    try {
      final analysis = await _apiService.analyzeWithImage(
        imagePath: _imagePath!,
        detectionModel: _catalog?.defaultDetectionModel,
        classificationModel: classificationModel,
      );

      if (mounted) {
        Navigator.pushNamed(
          context,
          AppConstants.detectionResultScreen,
          arguments: {
            'imagePath': _imagePath,
            'analysis': analysis,
          },
        );
      }
    } on NetworkException catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text('Network error: ${e.message}'),
          backgroundColor: AppTheme.error,
        ),
      );
    } on ServerException catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text('Server error: ${e.message}'),
          backgroundColor: AppTheme.error,
        ),
      );
    } on ValidationException catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(e.message),
          backgroundColor: AppTheme.error,
        ),
      );
    } on TimeoutException {
      messenger.showSnackBar(
        SnackBar(
          content: Text(
            'Analysis is taking longer than usual. The server may still be '
            'loading the AI models — please wait a moment and tap Analyze '
            'again.',
          ),
          backgroundColor: AppTheme.error,
        ),
      );
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(ErrorMessages.unexpectedError),
          backgroundColor: AppTheme.error,
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choose Model'),
        actions: [
          IconButton(
            icon: const Icon(Icons.close),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Select the classification model to analyze this image',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
            const SizedBox(height: AppTheme.spacingLg),

            if (_imagePath != null)
              Container(
                height: 220,
                width: double.infinity,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppTheme.radiusLg),
                ),
                clipBehavior: Clip.antiAlias,
                child: localImage(_imagePath!, fit: BoxFit.cover),
              ),

            const SizedBox(height: AppTheme.spacingLg),

            if (_isLoadingModels)
              const LoadingWidget(message: 'Loading available models...')
            else if (_hasError)
              ErrorStateWidget(
                message: 'Cannot load models',
                subMessage: _errorMessage,
                onRetry: _loadModels,
              )
            else if (_catalog != null &&
                _catalog!.classificationModels.isEmpty)
              EmptyStateWidget(
                message: 'No models available',
                subMessage:
                    'The server did not report any classification models. '
                    'Make sure the backend is started (run_project.ps1) and '
                    'try again.',
                icon: Icons.extension_off,
              )
            else if (_catalog != null) ...[
              _buildServerDefaultsCard(context),
              const SizedBox(height: AppTheme.spacingMd),
              _buildClassificationDropdown(context),
              const SizedBox(height: AppTheme.spacingMd),
              _buildAccuracyBasisCard(context),
              const SizedBox(height: AppTheme.spacingLg),
              // NOTE: never wrap this button in Expanded/Flexible. The button
              // lives inside a Column that is inside a SingleChildScrollView,
              // whose height is unbounded — an Expanded here throws a RenderFlex
              // "unbounded height" exception and the dropdown would never render.
              AppButton(
                text: 'Analyze Image',
                icon: Icons.analytics,
                isLoading: _isAnalyzing,
                onPressed: (_selectedClassificationModel != null)
                    ? _runAnalysis
                    : null,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildClassificationDropdown(BuildContext context) {
    final options = _catalog?.classificationModels ?? [];
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Classification Model',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: AppTheme.spacingXxs),
          Text(
            'Classifies the detected patch (patch-level crop/weed experiment; '
            'not weed-species identification)',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: AppTheme.spacingSm),
          if (options.isEmpty)
            _buildNoModelsNote(
                'No classification models were reported by the server.')
          else
            // expandedInsets makes the menu fill the card; a SizedBox wrapper
            // does not widen a DropdownMenu.
            DropdownMenu<String>(
              expandedInsets: EdgeInsets.zero,
              leadingIcon: const Icon(Icons.check_circle),
              hintText: 'Classification model',
              initialSelection: _selectedClassificationModel,
              // Selection only: typing into the field would not change the model.
              requestFocusOnTap: false,
              dropdownMenuEntries: [
                for (final option in options)
                  DropdownMenuEntry<String>(
                    value: option.name,
                    label: option.available
                        ? option.name
                        : '${option.name} (unavailable)',
                    trailingIcon: option.accuracy != null
                        ? Text('${(option.accuracy! * 100).toStringAsFixed(1)}%')
                        : null,
                    enabled: option.available,
                    leadingIcon: option.available
                        ? const Icon(Icons.check)
                        : const Icon(Icons.block),
                  ),
              ],
              onSelected: (String? value) {
                if (value != null && mounted) {
                  setState(() {
                    _selectedClassificationModel = value;
                  });
                }
              },
            ),
        ],
      ),
    );
  }

  Widget _buildNoModelsNote(String message) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppTheme.spacingSm),
      decoration: BoxDecoration(
        color: AppTheme.warning.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(AppTheme.radiusSm),
        border: Border.all(color: AppTheme.warning),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: AppTheme.warning, size: 18),
          const SizedBox(width: AppTheme.spacingXs),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAccuracyBasisCard(BuildContext context) {
    final options = _catalog?.classificationModels
        .where((o) => o.available && o.accuracy != null)
        .toList() ??
        [];
    if (options.isEmpty) return const SizedBox.shrink();
    final lines = options
        .map((o) =>
            '${o.name}: ${(o.accuracy! * 100).toStringAsFixed(1)}%')
        .join('\n');
    return AppCard(
      color: AppTheme.primaryGreen.withValues(alpha: 0.05),
      borderColor: AppTheme.primaryGreen,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Accuracy basis (held-out test set)',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
          ),
          const SizedBox(height: AppTheme.spacingXxs),
          Text(
            'These are the tested model accuracies from the evaluation run — they '
            'are model statistics, NOT the per-image confidence shown on the '
            'result screen.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.textSecondary,
                  fontSize: 12,
                ),
          ),
          const SizedBox(height: AppTheme.spacingXxs),
          Text(
            lines,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.primaryGreen,
                  fontWeight: FontWeight.w500,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildServerDefaultsCard(BuildContext context) {
    final classificationCount = _catalog?.classificationModels
        .where((o) => o.available)
        .length ??
        0;
    return AppCard(
      color: AppTheme.primaryGreen.withValues(alpha: 0.08),
      borderColor: AppTheme.primaryGreen,
      child: Row(
        children: [
          Icon(Icons.info_outline, color: AppTheme.primaryGreen, size: 24),
          const SizedBox(width: AppTheme.spacingXs),
          Expanded(
            child: Text(
              '$classificationCount classification model(s) available on the server.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}