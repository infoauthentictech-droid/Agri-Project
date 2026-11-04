import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_cropper/image_cropper.dart';
import '../utils/app_media.dart';
import '../utils/error_handler.dart';
import '../utils/error_messages.dart';

/// Native image-cropping service (web throws a clear error: image_cropper has
/// no browser support). No `dart:io` usage so the web build compiles.
class CropService {
  /// Opens the native image cropper for the image at [imagePath].
  ///
  /// Parameters:
  /// - [imagePath]: The local file path of the image to crop.
  /// - [aspectRatio]: Optional aspect ratio lock (e.g. square).
  /// - [allowRotation]: Whether the user can rotate the image (default: true).
  /// - [allowResize]: Whether the user can resize the crop area (default: true).
  ///
  /// Returns the cropped image's file path on success.
  /// Throws an exception if the user cancels or an error occurs.
  Future<String> cropImage({
    required String imagePath,
    CropAspectRatioPreset? aspectRatio,
    bool allowRotation = true,
    bool allowResize = true,
  }) async {
    if (kIsWeb) {
      throw AppException('Image cropping is not available on the web build.');
    }
    try {
      // Verify the source file exists before attempting to crop
      if (!await mediaExists(imagePath)) {
        throw AppException(ErrorMessages.noImageSelected);
      }

      // Configure the available aspect ratio presets
      const List<CropAspectRatioPresetData> aspectRatioPresets = [
        CropAspectRatioPreset.original,
        CropAspectRatioPreset.square,
        CropAspectRatioPreset.ratio4x3,
        CropAspectRatioPreset.ratio16x9,
      ];

      // Configure the cropper with user-friendly options
      final CroppedFile? croppedFile = await ImageCropper().cropImage(
        sourcePath: imagePath,
        // Apply a locked aspect ratio if provided
        aspectRatio: aspectRatio != null
            ? CropAspectRatio(
                ratioX: aspectRatio.data!.$1.toDouble(),
                ratioY: aspectRatio.data!.$2.toDouble(),
              )
            : null,
        compressFormat: ImageCompressFormat.jpg,
        compressQuality: 85,
        uiSettings: [
          // Android-specific UI settings
          AndroidUiSettings(
            toolbarTitle: 'Crop Image',
            toolbarColor: const Color(0xFF2E7D32), // Dark green for agriculture theme
            toolbarWidgetColor: const Color(0xFFFFFFFF),
            statusBarLight: true,
            initAspectRatio: aspectRatio ?? CropAspectRatioPreset.original,
            lockAspectRatio: aspectRatio != null,
            hideBottomControls: false,
            showCropGrid: true,
            aspectRatioPresets: aspectRatioPresets,
          ),
          // iOS-specific UI settings
          IOSUiSettings(
            title: 'Crop Image',
            aspectRatioLockEnabled: aspectRatio != null,
            resetAspectRatioEnabled: true,
            rotateButtonsHidden: !allowRotation,
            aspectRatioPresets: aspectRatioPresets,
          ),
        ],
      );

      // Handle user cancellation gracefully
      if (croppedFile == null) {
        throw AppException('Image cropping was cancelled.');
      }

      return croppedFile.path;
    } catch (e) {
      if (e is AppException) {
        rethrow;
      }
      throw AppException(
        'An unexpected error occurred while cropping the image. Please try again.',
      );
    }
  }

  /// Validates that a cropped image file exists and is readable.
  ///
  /// Returns `true` if the file at [filePath] exists and is a valid image.
  Future<bool> validateCroppedImage(String filePath) async {
    try {
      if (!await mediaExists(filePath)) return false;
      final fileSize = await mediaLength(filePath);
      // Ensure the cropped file is not empty
      return fileSize > 0;
    } catch (_) {
      return false;
    }
  }
}
