import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_card.dart';
import '../services/image_service.dart';
import '../utils/error_handler.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
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
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.message),
            backgroundColor: AppTheme.error,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConstants.appName),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'Detection History',
            onPressed: () {
              Navigator.pushNamed(context, AppConstants.historyScreen);
            },
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: () {
              Navigator.pushNamed(context, AppConstants.settingsScreen);
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Welcome Section
            const SizedBox(height: AppTheme.spacingSm),
            Text(
              'Welcome to Groundnut AI',
              style: Theme.of(context).textTheme.displaySmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: AppTheme.spacingXs),
            Text(
              'Detect diseases, manage weeds, and get herbicide recommendations',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: AppTheme.textSecondary,
              ),
            ),
            const SizedBox(height: AppTheme.spacingXl),
            
            // Main Actions
            _buildActionCard(
              context,
              icon: Icons.camera_alt,
              title: 'Take Photo',
              subtitle: 'Capture image with camera',
              onTap: () {
                Navigator.pushNamed(context, AppConstants.cameraScreen);
              },
            ),
            _buildActionCard(
              context,
              icon: Icons.photo_library,
              title: 'Upload Image',
              subtitle: 'Select from gallery',
              onTap: _isLoading ? null : _pickFromGallery,
              isLoading: _isLoading,
            ),
            _buildActionCard(
              context,
              icon: Icons.medication,
              title: 'Herbicide Recommendations',
              subtitle: 'Get weed management advice',
              onTap: () {
                Navigator.pushNamed(context, AppConstants.herbicideScreen);
              },
            ),
            _buildActionCard(
              context,
              icon: Icons.visibility,
              title: 'XAI Explanations',
              subtitle: 'Understand model decisions',
              onTap: () {
                Navigator.pushNamed(context, AppConstants.xaiScreen);
              },
            ),
            
            const SizedBox(height: AppTheme.spacingLg),
            
            // Info Card
            AppCard(
              color: AppTheme.lightGreen.withValues(alpha: 0.1),
              borderColor: AppTheme.lightGreen,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline, color: AppTheme.primaryGreen),
                      const SizedBox(width: AppTheme.spacingXs),
                      const Text(
                        'About',
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: AppTheme.spacingXs),
                  Text(
                    'This canary app uses a research AI model to flag objects '
                    'in groundnut field photos and suggests herbicide options '
                    'from a knowledge base that is still pending agronomic '
                    'review. Detector classes and density are unconfirmed; '
                    'always verify with a specialist.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppTheme.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback? onTap,
    bool isLoading = false,
  }) {
    return AppCard(
      onTap: onTap,
      margin: const EdgeInsets.only(bottom: AppTheme.spacingXs),
      child: Row(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: AppTheme.primaryGreen.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(AppTheme.radiusMd),
            ),
            child: isLoading
                ? const Center(
                    child: SizedBox(
                      width: 24,
                      height: 24,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(AppTheme.primaryGreen),
                      ),
                    ),
                  )
                : Icon(
                    icon,
                    color: AppTheme.primaryGreen,
                    size: 32,
                  ),
          ),
          const SizedBox(width: AppTheme.spacingMd),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: AppTheme.spacingXxs),
                Text(
                  subtitle,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppTheme.textSecondary,
                  ),
                ),
              ],
            ),
          ),
          isLoading
              ? const SizedBox.shrink()
              : Icon(
                  Icons.arrow_forward_ios,
                  color: AppTheme.textHint,
                  size: 20,
                ),
        ],
      ),
    );
  }
}
