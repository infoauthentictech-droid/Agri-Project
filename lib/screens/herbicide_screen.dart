import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../widgets/app_card.dart';
import '../widgets/app_text_field.dart';
import '../widgets/app_button.dart';
import '../services/api_service.dart';
import '../models/recommendation.dart';
import '../models/weed_options.dart';
import '../utils/error_handler.dart';

class HerbicideScreen extends StatefulWidget {
  const HerbicideScreen({super.key});

  @override
  State<HerbicideScreen> createState() => _HerbicideScreenState();
}

class _HerbicideScreenState extends State<HerbicideScreen> {
  final _formKey = GlobalKey<FormState>();
  final _cropController = TextEditingController(text: 'groundnut');
  final _cropGrowthStageController = TextEditingController();
  final _seasonController = TextEditingController();
  final _regionController = TextEditingController();

  /// Selected from the validated option lists — the free-text fields above were
  /// replaced with dropdowns so typos can never produce a failed
  /// recommendation.
  String? _selectedSpecies;
  String? _selectedCategory;

  bool _isLoading = false;
  RecommendationResponse? _recommendation;
  final ApiService _apiService = ApiService();

  Future<void> _getRecommendation() async {
    if (!_formKey.currentState!.validate()) return;

    // The engine needs at least a species or a category.
    if (_selectedSpecies == null && _selectedCategory == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a weed species or category first'),
          backgroundColor: AppTheme.error,
        ),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      // Check backend availability
      final isAvailable = await _apiService.isBackendAvailable();
      if (!isAvailable) {
        throw ServerException('Backend is not available. Please check your connection.');
      }

      final response = await _apiService.getRecommendation(
        crop: _cropController.text,
        weedSpecies: _selectedSpecies,
        weedCategory: _selectedCategory,
        cropGrowthStage: _cropGrowthStageController.text.isEmpty ? null : _cropGrowthStageController.text,
        season: _seasonController.text.isEmpty ? null : _seasonController.text,
        region: _regionController.text.isEmpty ? null : _regionController.text,
      );

      if (mounted) {
        setState(() {
          _isLoading = false;
          _recommendation = response;
        });
      }
    } on NetworkException catch (e) {
      _showError('Network error: ${e.message}');
    } on ServerException catch (e) {
      _showError('Server error: ${e.message}');
    } on ValidationException catch (e) {
      _showError('Validation error: ${e.message}');
    } catch (e) {
      _showError('Request timed out. Please try again.');
    } finally {
      if (mounted && _isLoading) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showError(String message) {
    if (mounted) {
      setState(() {
        _isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: AppTheme.error,
        ),
      );
    }
  }

  @override
  void dispose() {
    _cropController.dispose();
    _cropGrowthStageController.dispose();
    _seasonController.dispose();
    _regionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Herbicide Recommendations'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Enter Crop and Weed Information',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 24),
              
              AppTextField(
                label: 'Crop',
                hint: 'e.g., groundnut',
                controller: _cropController,
                icon: Icons.agriculture,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Crop is required';
                  }
                  return null;
                },
              ),
              
              const SizedBox(height: 16),
              
              SizedBox(
                width: double.infinity,
                child: DropdownMenu<String>(
                  leadingIcon: const Icon(Icons.grass),
                  hintText: 'Weed Species',
                  initialSelection: _selectedSpecies,
                  dropdownMenuEntries: [
                    for (final s in kWeedSpeciesOptions)
                      DropdownMenuEntry<String>(value: s, label: s),
                  ],
                  onSelected: (String? value) {
                    if (value != null && mounted) {
                      setState(() {
                        _selectedSpecies = value;
                      });
                    }
                  },
                ),
              ),
              
              const SizedBox(height: 16),
              
              SizedBox(
                width: double.infinity,
                child: DropdownMenu<String>(
                  leadingIcon: const Icon(Icons.category),
                  hintText: 'Weed Category (optional)',
                  initialSelection: _selectedCategory,
                  dropdownMenuEntries: [
                    for (final c in kWeedCategoryOptions)
                      DropdownMenuEntry<String>(value: c, label: c),
                  ],
                  onSelected: (String? value) {
                    if (value != null && mounted) {
                      setState(() {
                        _selectedCategory = value;
                      });
                    }
                  },
                ),
              ),
              
              const SizedBox(height: 16),
              
              AppTextField(
                label: 'Crop Growth Stage',
                hint: 'e.g., post_emergence',
                controller: _cropGrowthStageController,
                icon: Icons.eco,
              ),
              
              const SizedBox(height: 16),
              
              AppTextField(
                label: 'Season',
                hint: 'e.g., kharif',
                controller: _seasonController,
                icon: Icons.wb_sunny,
              ),
              
              const SizedBox(height: 16),
              
              AppTextField(
                label: 'Region',
                hint: 'e.g., Tamil Nadu',
                controller: _regionController,
                icon: Icons.location_on,
              ),
              
              const SizedBox(height: 24),
              
              AppButton(
                text: 'Get Recommendation',
                icon: Icons.medication,
                isLoading: _isLoading,
                onPressed: _getRecommendation,
              ),
              
              if (_recommendation != null) ...[
                const SizedBox(height: 32),
                const Text(
                  'Recommendation',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16),
                
                AppCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (_recommendation!.detectedWeed != null)
                        _buildInfoRow('Detected Weed', _recommendation!.detectedWeed),
                      if (_recommendation!.weedCategory != null)
                        _buildInfoRow('Weed Category', _recommendation!.weedCategory),
                      if (_recommendation!.candidateManagementOptions.isNotEmpty)
                        _buildInfoRow(
                          'Management Options',
                          _recommendation!.candidateManagementOptions.join(', '),
                        ),
                      if (_recommendation!.activeIngredient != null)
                        _buildInfoRow('Active Ingredient', _recommendation!.activeIngredient),
                      if (_recommendation!.applicationInformation != null)
                        _buildApplicationInfo(_recommendation!.applicationInformation!),
                      _buildInfoRow('Source', _recommendation!.source),
                      _buildInfoRow('Confidence', _recommendation!.confidence),
                      _buildInfoRow('Reason', _recommendation!.reason),
                      
                      if (_recommendation!.warnings.isNotEmpty)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const SizedBox(height: 16),
                            const Text(
                              'Warnings:',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: AppTheme.warning,
                              ),
                            ),
                            ..._recommendation!.warnings.map((warning) => Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text(
                                '• $warning',
                                style: const TextStyle(color: AppTheme.warning),
                              ),
                            )),
                          ],
                        ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String? value) {
    if (value == null || value.isEmpty) return const SizedBox.shrink();
    
    return Padding(
      padding:	const EdgeInsets.only(bottom: 8),
      child: Row(
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
            child: Text(value),
          ),
        ],
      ),
    );
  }

  Widget _buildApplicationInfo(ApplicationInformation info) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 8),
        const Text(
          'Application Information:',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: AppTheme.textSecondary,
          ),
        ),
        if (info.applicationStage != null)
          _buildInfoRow('  Stage', info.applicationStage),
        if (info.doseAiPerHa != null)
          _buildInfoRow('  AI Dose', info.doseAiPerHa),
        if (info.doseFormulationPerHa != null)
          _buildInfoRow('  Formulation Dose', info.doseFormulationPerHa),
        if (info.phiWaitingPeriodDays != null)
          _buildInfoRow('  PHI (days)', info.phiWaitingPeriodDays.toString()),
      ],
    );
  }
}
