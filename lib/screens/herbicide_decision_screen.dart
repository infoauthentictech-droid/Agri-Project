import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_card.dart';
import '../widgets/app_button.dart';
import '../models/detection_result.dart';
import '../models/recommendation.dart';
import '../services/api_service.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';
import '../utils/loading_state.dart';
import '../utils/error_state.dart';

class HerbicideDecisionScreen extends StatefulWidget {
  const HerbicideDecisionScreen({super.key});

  @override
  State<HerbicideDecisionScreen> createState() => _HerbicideDecisionScreenState();
}

class _HerbicideDecisionScreenState extends State<HerbicideDecisionScreen> {
  DetectionResponse? _detectionResponse;
  RecommendationResponse? _recommendation;
  bool _isLoading = false;
  bool _hasError = false;
  String? _errorMessage;
  final ApiService _apiService = ApiService();

  /// Guards the automatic recommendation fetch from didChangeDependencies so it
  /// only fires once per route entry. The retry/action buttons still call
  /// _fetchRecommendation() directly.
  bool _autoFetchStarted = false;

  /// The automatic flow does NOT ask the user for a weed species or category.
  /// The app honestly reports that the species is unidentified and the engine
  /// returns only broad-spectrum (general) options.
  static const String kAutomaticWeedCategory = 'not_specified';

  /// The weed class name reported by the detector (e.g. class_0/class_1). This
  /// is display-only; the automatic flow never claims an exact species.
  String? _detectedSpecies() {
    if (_detectionResponse == null ||
        _detectionResponse!.predictions.isEmpty) {
      return null;
    }
    return _detectionResponse!.predictions.first.className;
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map?;
    if (args != null && args.containsKey('detectionResponse')) {
      setState(() {
        _detectionResponse = args['detectionResponse'] as DetectionResponse;
      });
      // Automatic mode: always fetch the recommendation without asking the
      // user to identify the weed. The engine is told the species is unknown
      // and returns only broad-spectrum options (never invented species).
      if (!_autoFetchStarted) {
        _autoFetchStarted = true;
        _fetchRecommendation();
      }
    }
  }

  Future<void> _fetchRecommendation() async {
    if (_detectionResponse == null) return;

    // Automatic detection: the app never fabricates a species. It reports
    // "species not identified" (weed_category = not_specified) so the engine
    // can return general-spectrum records only, and clearly labels them.
    final weedSpecies = null;
    final weedCategory = kAutomaticWeedCategory;

    setState(() {
      _isLoading = true;
      _hasError = false;
      _errorMessage = null;
    });

    try {
      final response = await _apiService.getRecommendation(
        crop: 'groundnut',
        weedSpecies: weedSpecies,
        weedCategory: weedCategory,
      );

      if (mounted) {
        setState(() {
          _isLoading = false;
          _recommendation = response;
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

  String? _getDetectedLabel() {
    final name = _detectedSpecies();
    if (name == null || name.trim().isEmpty) return null;
    if (RegExp(r'^class_\d+$').hasMatch(name.trim())) {
      return '$name (species not identified)';
    }
    return name;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Herbicide Decision Support'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_isLoading)
              const LoadingWidget(message: 'Analyzing automatically...')
            else if (_hasError)
              ErrorStateWidget(
                message: 'Failed to Get Recommendation',
                subMessage: _errorMessage,
                onRetry: () => _fetchRecommendation(),
              )
            else if (_recommendation == null)
              _buildInitialView()
            else
              _buildRecommendationView(),
          ],
        ),
      ),
    );
  }

  Widget _buildInitialView() {
    return Column(
      children: [
        if (_detectionResponse != null) ...[
          _buildAIDetectionSection(),
          const SizedBox(height: AppTheme.spacingLg),
        ],
        // Automatic mode: the AI detects the weed automatically; the user
        // never has to pick a species or category manually.
        _buildAutomaticDetectionCard(),
        const SizedBox(height: AppTheme.spacingLg),
        Icon(
          Icons.medication_outlined,
          size: 64,
          color: AppTheme.primaryGreen.withValues(alpha: 0.5),
        ),
        const SizedBox(height: AppTheme.spacingLg),
        Text(
          'Automatic Weed Detection',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: AppTheme.spacingXs),
        Text(
          'Weed detection and recommendation are automatic. If the model '
          'cannot identify the exact species, broad-spectrum (general) '
          'recommendations are shown with a clear notice.',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            color: AppTheme.textSecondary,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: AppTheme.spacingLg),
        AppButton(
          text: 'Get Recommendation',
          icon: Icons.medication,
          isLoading: _isLoading,
          onPressed: () => _fetchRecommendation(),
        ),
      ],
    );
  }

  /// Tells the user that the weed was detected automatically and that no
  /// manual species/category selection is needed.
  Widget _buildAutomaticDetectionCard() {
    return AppCard(
      color: AppTheme.primaryGreen.withValues(alpha: 0.08),
      borderColor: AppTheme.primaryGreen,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.rocket_launch,
                color: AppTheme.primaryGreen,
                size: 20,
              ),
              const SizedBox(width: AppTheme.spacingXs),
              Expanded(
                child: Text(
                  'Automatic detection',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: AppTheme.primaryGreen,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingXs),
          Text(
            'The AI identifies the weed automatically from the image. When '
            'the exact species cannot be identified, the app is transparent '
            'about it and recommends a broad-spectrum (general) product — '
            'it never invents a species-specific claim.',
            style: TextStyle(color: AppTheme.textSecondary, height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendationView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // AI Detection Section
        if (_detectionResponse != null) ...[
          _buildAIDetectionSection(),
          const SizedBox(height: 24),
        ],
        
        // Herbicide Decision Support Section
        _buildDecisionSupportSection(),
        
        const SizedBox(height: AppTheme.spacingLg),
        
        // Disclaimer
        _buildDisclaimer(),
        
        const SizedBox(height: AppTheme.spacingLg),
        
        // Action Buttons
        Row(
          children: [
            Expanded(
              child: AppButton(
                text: 'New Detection',
                icon: Icons.camera_alt,
                onPressed: () {
                  Navigator.pushNamedAndRemoveUntil(
                    context,
                    AppConstants.homeScreen,
                    (route) => false,
                  );
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

  Widget _buildAIDetectionSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: AppTheme.spacingSm, vertical: AppTheme.spacingXs),
          decoration: BoxDecoration(
            color: AppTheme.primaryGreen.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            border: Border.all(color: AppTheme.primaryGreen),
          ),
          child: Row(
            children: [
              Icon(
                Icons.psychology,
                color: AppTheme.primaryGreen,
                size: 20,
              ),
              const SizedBox(width: AppTheme.spacingXs),
              Text(
                'AI Detection',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppTheme.primaryGreen,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppTheme.spacingSm),
        
        if (_detectionResponse!.predictions.isNotEmpty) ...[
          AppCard(
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: AppTheme.primaryGreen.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                  ),
                  child: Icon(
                    Icons.grass,
                    color: AppTheme.primaryGreen,
                    size: 24,
                  ),
                ),
                const SizedBox(width: AppTheme.spacingSm),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detected (automatic)',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          color: AppTheme.textSecondary,
                        ),
                      ),
                      const SizedBox(height: AppTheme.spacingXxs),
                      Text(
                        _getDetectedLabel() ?? 'Weed',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w600,
                          color: AppTheme.primaryGreen,
                        ),
                      ),
                      const SizedBox(height: AppTheme.spacingXxs),
                      Text(
                        'Confidence: ${(_detectionResponse!.predictions.first.confidence * 100).toStringAsFixed(0)}% · species auto-detected',
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
        ],
      ],
    );
  }

  Widget _buildDecisionSupportSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: AppTheme.spacingSm, vertical: AppTheme.spacingXs),
          decoration: BoxDecoration(
            color: AppTheme.warning.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            border: Border.all(color: AppTheme.warning),
          ),
          child: Row(
            children: [
              Icon(
                Icons.agriculture,
                color: AppTheme.warning,
                size: 20,
              ),
              const SizedBox(width: AppTheme.spacingXs),
              Text(
                'Herbicide Decision Support',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppTheme.warning,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppTheme.spacingSm),
        
        if (_recommendation!.candidateManagementOptions.isNotEmpty)
          _buildRecommendationContent()
        else
          _buildNoRecommendationView(),
      ],
    );
  }

  Widget _buildRecommendationContent() {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Detected Weed
          if (_recommendation!.detectedWeed != null) ...[
            _buildInfoRow('Detected Weed', _recommendation!.detectedWeed!),
            const SizedBox(height: AppTheme.spacingSm),
          ],
          
          // Weed Category
          if (_recommendation!.weedCategory != null) ...[
            _buildInfoRow('Weed Category', _recommendation!.weedCategory!),
            const SizedBox(height: AppTheme.spacingSm),
          ],
          
          // Crop
          _buildInfoRow('Crop', 'Groundnut'),
          const SizedBox(height: AppTheme.spacingSm),
          
          // Management Options
          if (_recommendation!.candidateManagementOptions.isNotEmpty) ...[
            Text(
              'Validated Management Options',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: AppTheme.spacingXs),
            ..._recommendation!.candidateManagementOptions.map((option) => Padding(
              padding: const EdgeInsets.only(bottom: AppTheme.spacingXxs),
              child: Row(
                children: [
                  Icon(
                    Icons.check_circle,
                    color: AppTheme.primaryGreen,
                    size: 16,
                  ),
                  const SizedBox(width: AppTheme.spacingXs),
                  Expanded(
                    child: Text(
                      option,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.textSecondary,
                      ),
                    ),
                  ),
                ],
              ),
            )),
            const SizedBox(height: AppTheme.spacingSm),
          ],
          
          // Active Ingredient
          if (_recommendation!.activeIngredient != null) ...[
            _buildInfoRow('Active Ingredient', _recommendation!.activeIngredient!),
            const SizedBox(height: AppTheme.spacingSm),
          ],
          
          // Application Information
          if (_recommendation!.applicationInformation != null) ...[
            Text(
              'Application Information',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: AppTheme.spacingXs),
            _buildApplicationInfo(_recommendation!.applicationInformation!),
            const SizedBox(height: AppTheme.spacingSm),
          ],
          
          // Reason
          _buildInfoRow('Reason', _recommendation!.reason),
          const SizedBox(height: AppTheme.spacingSm),
          
          // Source
          _buildInfoRow('Source', _recommendation!.source ?? 'Unknown'),
          const SizedBox(height: AppTheme.spacingSm),
          
          // Confidence
          _buildInfoRow('Confidence Level', _recommendation!.confidence ?? 'Not specified'),
          
          // Warnings
          if (_recommendation!.warnings.isNotEmpty) ...[
            const SizedBox(height: AppTheme.spacingSm),
            Container(
              padding: const EdgeInsets.all(AppTheme.spacingSm),
              decoration: BoxDecoration(
                color: AppTheme.warning.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(AppTheme.radiusSm),
                border: Border.all(color: AppTheme.warning),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.warning_amber,
                        color: AppTheme.warning,
                        size: 20,
                      ),
                      const SizedBox(width: AppTheme.spacingXs),
                      Text(
                        'Important Warnings',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                          color: AppTheme.warning,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: AppTheme.spacingXs),
                  ..._recommendation!.warnings.map((warning) => Padding(
                    padding: const EdgeInsets.only(bottom: AppTheme.spacingXxs),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('• ', style: TextStyle(color: AppTheme.warning)),
                        Expanded(
                          child: Text(
                            warning,
                            style: const TextStyle(color: AppTheme.warning),
                          ),
                        ),
                      ],
                    ),
                  )),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildNoRecommendationView() {
    return AppCard(
      child: Column(
        children: [
          Icon(
            Icons.info_outline,
            size: 48,
            color: AppTheme.textHint,
          ),
          const SizedBox(height: AppTheme.spacingLg),
          Text(
            'No Validated Recommendation',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: AppTheme.spacingXs),
          Text(
            'No validated recommendation is available for this detection.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppTheme.textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: AppTheme.spacingXs),
          Text(
            'Please consult with local agricultural experts for guidance on managing this weed species.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textHint,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    if (value.isEmpty) return const SizedBox.shrink();
    
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 140,
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

  Widget _buildApplicationInfo(ApplicationInformation info) {
    return Container(
      padding: const EdgeInsets.all(AppTheme.spacingSm),
      decoration: BoxDecoration(
        color: AppTheme.surface.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(AppTheme.radiusSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (info.applicationStage != null) ...[
            _buildInfoRow('Application Stage', info.applicationStage!),
            const SizedBox(height: AppTheme.spacingXs),
          ],
          if (info.doseAiPerHa != null) ...[
            _buildInfoRow('AI Dose', info.doseAiPerHa!),
            const SizedBox(height: AppTheme.spacingXs),
          ],
          if (info.doseFormulationPerHa != null) ...[
            _buildInfoRow('Formulation Dose', info.doseFormulationPerHa!),
            const SizedBox(height: AppTheme.spacingXs),
          ],
          if (info.phiWaitingPeriodDays != null)
            _buildInfoRow('PHI (Days)', info.phiWaitingPeriodDays.toString()),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(AppTheme.spacingMd),
      decoration: BoxDecoration(
        color: AppTheme.error.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
        border: Border.all(color: AppTheme.error),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.warning,
                color: AppTheme.error,
                size: 24,
              ),
              const SizedBox(width: AppTheme.spacingSm),
              Expanded(
                child: Text(
                  'Important Disclaimer',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: AppTheme.error,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTheme.spacingSm),
          Text(
            'Herbicide options are drawn from a knowledge base that is still '
            'pending agronomic review; dose, PHI, and ingredient fields may be '
            'incomplete. Always follow local regulations, read product labels, '
            'and consult agricultural experts before applying herbicides.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}
