class AppException implements Exception {
  final String message;
  final String? code;
  
  AppException(this.message, {this.code});
  
  @override
  String toString() => message;
}

class NetworkException extends AppException {
  NetworkException(super.message) : super(code: 'NETWORK_ERROR');
}

class ValidationException extends AppException {
  ValidationException(super.message) : super(code: 'VALIDATION_ERROR');
}

/// Thrown when a required OS permission (camera / storage / photos) is denied
/// or permanently blocked, so the UI can offer a "Open Settings" action.
class PermissionDeniedException extends AppException {
  PermissionDeniedException(super.message) : super(code: 'PERMISSION_DENIED');
}

class ServerException extends AppException {
  ServerException(super.message) : super(code: 'SERVER_ERROR');
}

class AuthenticationException extends AppException {
  AuthenticationException(super.message) : super(code: 'AUTH_ERROR');
}

class NotFoundException extends AppException {
  NotFoundException(super.message) : super(code: 'NOT_FOUND');
}

class TimeoutException extends AppException {
  TimeoutException(super.message) : super(code: 'TIMEOUT');
}
