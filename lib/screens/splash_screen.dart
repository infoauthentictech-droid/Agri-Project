import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  /// Guards the delayed auto-navigation so it is scheduled exactly once per
  /// state instance, even if both initState and didChangeDependencies fire.
  bool _navigationScheduled = false;

  @override
  void initState() {
    super.initState();
    _scheduleNavigationToHome();
  }

  /// Also (re)arm the auto-navigation whenever the route is revived (e.g. via
  /// popUntil / pushNamedAndRemoveUntil). Without this, a re-shown splash would
  /// sit on its loading view forever because initState only runs once.
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Reset the navigation guard when the route is revived to allow navigation
    _navigationScheduled = false;
    _scheduleNavigationToHome();
  }

  void _scheduleNavigationToHome() {
    if (_navigationScheduled) return;
    _navigationScheduled = true;
    _navigateToHome();
  }

  Future<void> _navigateToHome() async {
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) {
      Navigator.pushReplacementNamed(context, AppConstants.homeScreen);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryGreen,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Logo or Icon
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(
                Icons.agriculture,
                size: 80,
                color: AppTheme.primaryGreen,
              ),
            ),
            const SizedBox(height: 32),
            // App Name
            const Text(
              AppConstants.appName,
              style: TextStyle(
                fontSize: 32,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            // Tagline
            const Text(
              'AI-Powered Groundnut Management',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 48),
            // Loading Indicator
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
            ),
          ],
        ),
      ),
    );
  }
}
