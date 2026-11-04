class ApplicationInformation {
  final String? applicationStage;
  final String? doseAiPerHa;
  final String? doseFormulationPerHa;
  final int? phiWaitingPeriodDays;
  
  ApplicationInformation({
    this.applicationStage,
    this.doseAiPerHa,
    this.doseFormulationPerHa,
    this.phiWaitingPeriodDays,
  });
  
  factory ApplicationInformation.fromJson(Map<String, dynamic> json) {
    return ApplicationInformation(
      applicationStage: json['application_stage'] as String?,
      doseAiPerHa: json['dose_ai_per_ha'] as String?,
      doseFormulationPerHa: json['dose_formulation_per_ha'] as String?,
      phiWaitingPeriodDays: json['phi_waiting_period_days'] as int?,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'application_stage': applicationStage,
      'dose_ai_per_ha': doseAiPerHa,
      'dose_formulation_per_ha': doseFormulationPerHa,
      'phi_waiting_period_days': phiWaitingPeriodDays,
    };
  }
}

class RecommendationResponse {
  final bool success;
  final String? detectedWeed;
  final String? weedCategory;
  final List<String> candidateManagementOptions;
  final String? activeIngredient;
  final ApplicationInformation? applicationInformation;
  final String reason;
  final String? source;
  final List<String> warnings;
  final String? confidence;
  final String status;
  
  RecommendationResponse({
    required this.success,
    this.detectedWeed,
    this.weedCategory,
    required this.candidateManagementOptions,
    this.activeIngredient,
    this.applicationInformation,
    required this.reason,
    this.source,
    required this.warnings,
    this.confidence,
    required this.status,
  });
  
  factory RecommendationResponse.fromJson(Map<String, dynamic> json) {
    final optionsList = json['candidate_management_options'] as List;
    final warningsList = json['warnings'] as List;
    
    return RecommendationResponse(
      success: json['success'] as bool,
      detectedWeed: json['detected_weed'] as String?,
      weedCategory: json['weed_category'] as String?,
      candidateManagementOptions:
          optionsList.map((e) => e as String).toList(),
      activeIngredient: json['active_ingredient'] as String?,
      applicationInformation: json['application_information'] != null
          ? ApplicationInformation.fromJson(
              json['application_information'] as Map<String, dynamic>)
          : null,
      reason: json['reason'] as String,
      source: json['source'] as String?,
      warnings: warningsList.map((e) => e as String).toList(),
      confidence: json['confidence'] as String?,
      status: json['status'] as String,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'detected_weed': detectedWeed,
      'weed_category': weedCategory,
      'candidate_management_options': candidateManagementOptions,
      'active_ingredient': activeIngredient,
      'application_information': applicationInformation?.toJson(),
      'reason': reason,
      'source': source,
      'warnings': warnings,
      'confidence': confidence,
      'status': status,
    };
  }
}
