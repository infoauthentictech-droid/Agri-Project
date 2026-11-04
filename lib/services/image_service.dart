import 'package:flutter/foundation.dart' show debugPrint, defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/services.dart' show PlatformException;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart' as permission_handler;
import 'package:image_cropper/image_cropper.dart';
import '../core/constants.dart';
import '../utils/app_media.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';
import 'crop_service.dart';

/// Image capture / gallery service (web-safe).
///
/// On the web there is no local filesystem: ``image_picker`` returns ``XFile``s
/// whose ``path`` is a ``blob:`` URL, and permission_handler is a no-op.
/// Everything that used ``dart:io`` now goes through the conditional-import
/// media helpers (``app_media.dart``) so the same code compiles for native and
/// web builds.
class ImageService {
  final ImagePicker _picker = ImagePicker();
  final CropService _cropService = CropService();

  /// Request camera permission. Web builds ask the browser through
  /// image_picker and never require permission_handler.
  Future<bool> requestCameraPermission() async {
    if (kIsWeb) return true;
    final status = await permission_handler.Permission.camera.request();
    return status.isGranted;
  }

  /// Best-effort request of the media/photos permission (native only).
  ///
  /// gallery selection goes through the system photo picker which needs no
  /// runtime permission, so this is only used for future save-to-gallery
  /// features. On Android we try the new photo permission first and fall back
  /// to legacy storage permission for older API levels.
  Future<bool> requestStoragePermission() async {
    if (kIsWeb) return true;
    if (defaultTargetPlatform != TargetPlatform.android) return true;

    var status = await permission_handler.Permission.photos.request();
    if (status.isGranted) return true;
    status = await permission_handler.Permission.storage.request();
    return status.isGranted;
  }

  /// Pick image from camera
  Future<String?> pickImageFromCamera() async {
    final hasPermission = await requestCameraPermission();
    if (!hasPermission) {
      throw PermissionDeniedException(ErrorMessages.cameraPermissionDenied);
    }

    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: AppConstants.maxImageDimension.toDouble(),
        maxHeight: AppConstants.maxImageDimension.toDouble(),
        imageQuality: 85,
      );

      if (image == null) {
        return null; // User cancelled
      }

      await _validateImage(image);
      return image.path;
    } on AppException {
      rethrow;
    } on PlatformException catch (e) {
      if (_isPermissionError(e)) {
        throw PermissionDeniedException(ErrorMessages.cameraPermissionDenied);
      }
      throw AppException(ErrorMessages.unexpectedError);
    } catch (e) {
      debugPrint('pickImageFromCamera failed: $e');
      throw AppException(ErrorMessages.unexpectedError);
    }
  }

  /// Pick image from gallery
  Future<String?> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: AppConstants.maxImageDimension.toDouble(),
        maxHeight: AppConstants.maxImageDimension.toDouble(),
        imageQuality: 85,
      );

      if (image == null) {
        return null; // User cancelled
      }

      await _validateImage(image);
      return image.path;
    } on AppException {
      rethrow;
    } on PlatformException catch (e) {
      if (_isPermissionError(e)) {
        throw PermissionDeniedException(ErrorMessages.galleryPermissionDenied);
      }
      throw AppException(ErrorMessages.unexpectedError);
    } catch (e) {
      debugPrint('pickImageFromGallery failed: $e');
      throw AppException(ErrorMessages.unexpectedError);
    }
  }

  /// Best-effort detection of an OS "permission denied" platform error.
  bool _isPermissionError(PlatformException e) {
    final code = e.code.toLowerCase();
    final message = (e.message ?? '').toLowerCase();
    return code.contains('permission') ||
        code.contains('access_denied') ||
        message.contains('permission') ||
        message.contains('denied');
  }

  /// Validate image file/blob (web-safe size/format checks).
  Future<void> _validateImage(XFile image) async {
    try {
      if (!await mediaExists(image.path)) {
        throw ValidationException(ErrorMessages.corruptedImage);
      }

      final fileSize = await mediaLength(image.path);
      final sizeMB = fileSize / (1024 * 1024);
      if (sizeMB > AppConstants.maxImageSizeMB) {
        throw ValidationException(ErrorMessages.imageTooLarge);
      }

      if (!_hasAllowedFormat(image)) {
        throw ValidationException(ErrorMessages.unsupportedFormat);
      }
    } on ValidationException {
      rethrow;
    } catch (e) {
      throw ValidationException(ErrorMessages.corruptedImage);
    }
  }

  /// Format check that tolerates web blob/data URLs (which carry no filename
  /// extension). Prefers the mime-type reported by image_picker when present.
  bool _hasAllowedFormat(XFile image) {
    final mime = (image.mimeType ?? '').toLowerCase();
    if (mime.isNotEmpty) {
      return mime.contains('jpeg') || mime.contains('png');
    }
    final path = image.path.toLowerCase();
    if (path.startsWith('blob:') || path.startsWith('data:')) {
      // Browser pickers only return images; trust the picker here.
      return true;
    }
    final extension = path.split('.').last;
    return AppConstants.allowedImageFormats.contains(extension);
  }

  /// Open app settings for permissions
  Future<void> openAppSettings() async {
    await permission_handler.openAppSettings();
  }

  /// Crop image with optional aspect ratio (native only; web throws a clear
  /// error because image_cropper does not support browsers).
  Future<String?> cropImage({
    required String imagePath,
    CropAspectRatioPreset? aspectRatio,
    bool allowRotation = true,
    bool allowResize = true,
  }) async {
    if (kIsWeb) {
      throw AppException('Image cropping is not available on the web build.');
    }
    try {
      final croppedPath = await _cropService.cropImage(
        imagePath: imagePath,
        aspectRatio: aspectRatio,
        allowRotation: allowRotation,
        allowResize: allowResize,
      );

      if (await _cropService.validateCroppedImage(croppedPath)) {
        return croppedPath;
      } else {
        throw AppException(ErrorMessages.corruptedImage);
      }
    } on AppException {
      rethrow;
    } catch (e) {
      throw AppException(ErrorMessages.unexpectedError);
    }
  }
}
