/// Centralized user-friendly error messages
/// Never expose stack traces or internal implementation details to users
class ErrorMessages {
  // IMAGE ERRORS
  static const String noImageSelected = 'No image selected. Please choose an image to continue.';
  static const String unsupportedFormat = 'This image format is not supported. Please use JPEG or PNG format.';
  static const String corruptedImage = 'The image file appears to be corrupted. Please try a different image.';
  static const String imageTooLarge = 'This image is too large. Please use an image smaller than 10MB.';
  static const String imageTooSmall = 'This image is too small. Please use a higher resolution image.';
  static const String imageBlurry = 'This image appears to be blurry. Please capture a clearer image.';
  static const String imageTooDark = 'This image is too dark. Please capture in better lighting.';
  static const String noPlantDetected = 'No plant was detected in this image. Please ensure the image contains a clear view of the plant.';
  static const String notPlantImage = 'This image does not appear to be a crop-field photo. Please retake or upload a clear photo of the groundnut field or weeds.';
  
  // ML ERRORS
  static const String modelUnavailable = 'The AI model is currently unavailable. Please try again later.';
  static const String inferenceFailure = 'Failed to analyze the image. Please try again.';
  static const String unknownClass = 'The detected object is not recognized. Please try with a clearer image.';
  static const String lowConfidence = 'The AI confidence is low. Please capture a clearer image for better results.';
  static const String multipleDetections = 'Multiple objects detected. Please focus on a single plant.';
  static const String noDetection = 'No supported groundnut or weed object was confidently detected.';
  
  // BACKEND ERRORS
  static const String serverUnavailable = 'Unable to connect to the server. Please check your internet connection.';
  static const String requestTimeout = 'The request timed out. Please check your connection and try again.';
  static const String badRequest = 'Invalid request. Please try again.';
  static const String payloadTooLarge = 'The image is too large for the server. Please use a smaller image.';
  static const String unsupportedMediaType = 'The server does not support this image format.';
  static const String validationError = 'Invalid data provided. Please try again.';
  static const String serverError = 'Server error occurred. Please try again later.';
  static const String serviceUnavailable = 'The service is temporarily unavailable. Please try again later.';
  static const String authenticationFailed = 'The app could not authenticate with the server. Re-run run_project.ps1 once so the app and the computer use the same connection key, then open the app again.';
  
  // RECOMMENDATION ERRORS
  static const String weedNotFound = 'No recommendation found for this weed species.';
  static const String missingHerbicideData = 'Herbicide information is not available for this weed.';
  static const String conflictingRecords = 'Conflicting information found. Please consult agricultural experts.';
  static const String missingSource = 'Source information is not available for this recommendation.';
  static const String unsupportedWeed = 'This weed species is not currently supported.';
  static const String insufficientInformation = 'Insufficient information to provide a recommendation.';
    static const String noValidatedRecommendations =
        'No validated herbicide recommendations are available yet. The '
        'knowledge base is awaiting agronomic review. Start the backend with '
        'research mode enabled to preview draft broad-spectrum options.';
  
  // MOBILE ERRORS
  static const String cameraPermissionDenied = 'Camera permission is required to capture images. Please grant permission in settings.';
  static const String galleryPermissionDenied = 'Gallery permission is required to select images. Please grant permission in settings.';
  static const String networkUnavailable = 'No internet connection. Please check your network settings.';
  static const String applicationInterrupted = 'The operation was interrupted. Please try again.';
  static const String invalidBackendUrl = 'Server configuration error. Please contact support.';
  
  // GENERIC ERRORS
  static const String unexpectedError = 'An unexpected error occurred. Please try again.';
  static const String operationCancelled = 'Operation cancelled by user.';
  
  /// Get user-friendly message for exception type
  static String getMessageForException(dynamic exception) {
    final exceptionStr = exception.toString().toLowerCase();
    
    if (exceptionStr.contains('permission')) {
      if (exceptionStr.contains('camera')) {
        return cameraPermissionDenied;
      }
      if (exceptionStr.contains('storage') || exceptionStr.contains('gallery')) {
        return galleryPermissionDenied;
      }
    }
    
    if (exceptionStr.contains('network') || exceptionStr.contains('socket')) {
      return networkUnavailable;
    }
    
    if (exceptionStr.contains('timeout')) {
      return requestTimeout;
    }
    
    if (exceptionStr.contains('format')) {
      return unsupportedFormat;
    }
    
    if (exceptionStr.contains('size')) {
      return imageTooLarge;
    }
    
    return unexpectedError;
  }
}
