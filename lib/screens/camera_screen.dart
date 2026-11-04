import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_button.dart';
import '../services/image_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isPermissionDenied = false;
  bool _isLoading = true;
  final ImageService _imageService = ImageService();

  @override
  void initState() {
    super.initState();
    _checkPermissionAndInitializeCamera();
  }

  Future<void> _checkPermissionAndInitializeCamera() async {
    setState(() {
      _isLoading = true;
    });

    try {
      // Check camera permission
      final hasPermission = await _imageService.requestCameraPermission();
      
      if (!hasPermission) {
        if (mounted) {
          setState(() {
            _isPermissionDenied = true;
            _isLoading = false;
          });
        }
        return;
      }

      // Initialize camera
      await _initializeCamera();
    } catch (e) {
      debugPrint('Error checking permission: $e');
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        _controller = CameraController(
          _cameras![0],
          ResolutionPreset.high,
        );
        await _controller!.initialize();
        if (mounted) {
          setState(() {
            _isInitialized = true;
            _isLoading = false;
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    try {
      final image = await _controller!.takePicture();
      if (mounted) {
        Navigator.pushNamed(
          context,
          AppConstants.imagePreviewScreen,
          arguments: {'imagePath': image.path},
        );
      }
    } catch (e) {
      debugPrint('Error taking picture: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to take picture')),
        );
      }
    }
  }

  Future<void> _openSettings() async {
    await openAppSettings();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera'),
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(),
            )
          : _isPermissionDenied
              ? _buildPermissionDeniedView()
              : _isInitialized && _controller != null
                  ? _buildCameraView()
                  : _buildNoCameraView(),
    );
  }

  Widget _buildPermissionDeniedView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.camera_alt_outlined,
              size: 64,
              color: AppTheme.textHint,
            ),
            const SizedBox(height: 16),
            const Text(
              'Camera Permission Required',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Please grant camera permission to use this feature.',
              style: TextStyle(
                color: AppTheme.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            AppButton(
              text: 'Grant Permission',
              icon: Icons.settings,
              onPressed: _openSettings,
            ),
            const SizedBox(height: 16),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraView() {
    return Stack(
      children: [
        CameraPreview(_controller!),
        Positioned(
          bottom: 32,
          left: 0,
          right: 0,
          child: Center(
            child: AppButton(
              text: 'Capture',
              icon: Icons.camera,
              onPressed: _takePicture,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildNoCameraView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.camera_alt_outlined,
              size: 64,
              color: AppTheme.textHint,
            ),
            const SizedBox(height: 16),
            const Text(
              'No Camera Available',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'This device does not have a camera or it is not available.',
              style: TextStyle(
                color: AppTheme.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Go Back'),
            ),
          ],
        ),
      ),
    );
  }
}
