import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_button.dart';
import '../widgets/app_card.dart';
import '../services/image_service.dart';
import '../utils/error_handler.dart';

class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key});

  @override
  State<GalleryScreen> createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  final ImageService _imageService = ImageService();
  bool _isLoading = false;

  Future<void> _pickFromGallery() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final imagePath = await _imageService.pickImageFromGallery();
      
      if (imagePath != null && mounted) {
        Navigator.pushNamed(
          context,
          AppConstants.imagePreviewScreen,
          arguments: {'imagePath': imagePath},
        );
      }
    } on AppException catch (e) {
      _showErrorDialog(e.message);
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showErrorDialog(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: AppTheme.error,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Upload Image'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Upload Image',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 24),
                  
                  AppCard(
                    child: Column(
                      children: [
                        Icon(
                          Icons.photo_library,
                          size: 64,
                          color: AppTheme.primaryGreen,
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Gallery',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Select from device gallery',
                          style: TextStyle(
                            color: AppTheme.textSecondary,
                          ),
                        ),
                        const SizedBox(height: 16),
                        AppButton(
                          text: 'Open Gallery',
                          icon: Icons.image,
                          onPressed: _pickFromGallery,
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 24),
                  
                  Text(
                    'Supported formats: JPEG, PNG, JPG',
                    style: TextStyle(
                      color: AppTheme.textHint,
                      fontSize: 12,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Maximum size: ${AppConstants.maxImageSizeMB}MB',
                    style: TextStyle(
                      color: AppTheme.textHint,
                      fontSize: 12,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Maximum dimension: ${AppConstants.maxImageDimension}px',
                    style: TextStyle(
                      color: AppTheme.textHint,
                      fontSize: 12,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
    );
  }
}
