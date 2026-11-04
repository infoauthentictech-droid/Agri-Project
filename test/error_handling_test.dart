import 'package:flutter_test/flutter_test.dart';
import 'package:groundnut_ai/utils/error_messages.dart';

void main() {
  group('ErrorMessages', () {
    test('should have all required error messages defined', () {
      // IMAGE ERRORS
      expect(ErrorMessages.noImageSelected, isNotEmpty);
      expect(ErrorMessages.unsupportedFormat, isNotEmpty);
      expect(ErrorMessages.corruptedImage, isNotEmpty);
      expect(ErrorMessages.imageTooLarge, isNotEmpty);
      expect(ErrorMessages.imageTooSmall, isNotEmpty);
      expect(ErrorMessages.imageBlurry, isNotEmpty);
      expect(ErrorMessages.imageTooDark, isNotEmpty);
      expect(ErrorMessages.noPlantDetected, isNotEmpty);
      
      // ML ERRORS
      expect(ErrorMessages.modelUnavailable, isNotEmpty);
      expect(ErrorMessages.inferenceFailure, isNotEmpty);
      expect(ErrorMessages.unknownClass, isNotEmpty);
      expect(ErrorMessages.lowConfidence, isNotEmpty);
      expect(ErrorMessages.multipleDetections, isNotEmpty);
      expect(ErrorMessages.noDetection, isNotEmpty);
      
      // BACKEND ERRORS
      expect(ErrorMessages.serverUnavailable, isNotEmpty);
      expect(ErrorMessages.requestTimeout, isNotEmpty);
      expect(ErrorMessages.badRequest, isNotEmpty);
      expect(ErrorMessages.payloadTooLarge, isNotEmpty);
      expect(ErrorMessages.unsupportedMediaType, isNotEmpty);
      expect(ErrorMessages.validationError, isNotEmpty);
      expect(ErrorMessages.serverError, isNotEmpty);
      expect(ErrorMessages.serviceUnavailable, isNotEmpty);
      
      // RECOMMENDATION ERRORS
      expect(ErrorMessages.weedNotFound, isNotEmpty);
      expect(ErrorMessages.missingHerbicideData, isNotEmpty);
      expect(ErrorMessages.conflictingRecords, isNotEmpty);
      expect(ErrorMessages.missingSource, isNotEmpty);
      expect(ErrorMessages.unsupportedWeed, isNotEmpty);
      expect(ErrorMessages.insufficientInformation, isNotEmpty);
      expect(ErrorMessages.noValidatedRecommendations, isNotEmpty);
      
      // MOBILE ERRORS
      expect(ErrorMessages.cameraPermissionDenied, isNotEmpty);
      expect(ErrorMessages.galleryPermissionDenied, isNotEmpty);
      expect(ErrorMessages.networkUnavailable, isNotEmpty);
      expect(ErrorMessages.applicationInterrupted, isNotEmpty);
      expect(ErrorMessages.invalidBackendUrl, isNotEmpty);
      
      // GENERIC ERRORS
      expect(ErrorMessages.unexpectedError, isNotEmpty);
      expect(ErrorMessages.operationCancelled, isNotEmpty);
    });
    
    test('getMessageForException should return appropriate message for permission errors', () {
      final message = ErrorMessages.getMessageForException('Camera permission denied');
      expect(message, equals(ErrorMessages.cameraPermissionDenied));
    });
    
    test('getMessageForException should return appropriate message for network errors', () {
      final message = ErrorMessages.getMessageForException('SocketException');
      expect(message, equals(ErrorMessages.networkUnavailable));
    });
    
    test('getMessageForException should return appropriate message for timeout errors', () {
      final message = ErrorMessages.getMessageForException('TimeoutException');
      expect(message, equals(ErrorMessages.requestTimeout));
    });
    
    test('getMessageForException should return generic message for unknown errors', () {
      final message = ErrorMessages.getMessageForException('Unknown error');
      expect(message, equals(ErrorMessages.unexpectedError));
    });
    
    test('all error messages should not contain technical jargon', () {
      final messages = [
        ErrorMessages.noImageSelected,
        ErrorMessages.unsupportedFormat,
        ErrorMessages.corruptedImage,
        ErrorMessages.imageTooLarge,
        ErrorMessages.imageTooSmall,
        ErrorMessages.imageBlurry,
        ErrorMessages.imageTooDark,
        ErrorMessages.noPlantDetected,
        ErrorMessages.modelUnavailable,
        ErrorMessages.inferenceFailure,
        ErrorMessages.unknownClass,
        ErrorMessages.lowConfidence,
        ErrorMessages.multipleDetections,
        ErrorMessages.noDetection,
        ErrorMessages.serverUnavailable,
        ErrorMessages.requestTimeout,
        ErrorMessages.badRequest,
        ErrorMessages.payloadTooLarge,
        ErrorMessages.unsupportedMediaType,
        ErrorMessages.validationError,
        ErrorMessages.serverError,
        ErrorMessages.serviceUnavailable,
        ErrorMessages.weedNotFound,
        ErrorMessages.missingHerbicideData,
        ErrorMessages.conflictingRecords,
        ErrorMessages.missingSource,
        ErrorMessages.unsupportedWeed,
        ErrorMessages.insufficientInformation,
        ErrorMessages.noValidatedRecommendations,
        ErrorMessages.cameraPermissionDenied,
        ErrorMessages.galleryPermissionDenied,
        ErrorMessages.networkUnavailable,
        ErrorMessages.applicationInterrupted,
        ErrorMessages.invalidBackendUrl,
        ErrorMessages.unexpectedError,
        ErrorMessages.operationCancelled,
      ];
      
      // Check that no message contains technical terms like "stack trace", "exception", "null pointer", etc.
      final forbiddenTerms = ['stack trace', 'exception', 'null pointer', 'NullPointerException', 'SocketException', 'HttpException', 'FormatException', '500', '404', '403', '401', '422', '413', '415', '400'];
      
      for (final message in messages) {
        final lowerMessage = message.toLowerCase();
        for (final term in forbiddenTerms) {
          expect(lowerMessage, isNot(contains(term)), reason: 'Message "$message" should not contain technical term "$term"');
        }
      }
    });
  });
}
