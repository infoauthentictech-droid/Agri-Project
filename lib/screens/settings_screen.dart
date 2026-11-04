import 'package:flutter/material.dart';
import '../core/api_config.dart';
import '../core/app_theme.dart';
import '../core/constants.dart';
import '../widgets/app_card.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'App Information',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildInfoRow('App Name', AppConstants.appName),
                  _buildInfoRow('Version', AppConstants.appVersion),
                  _buildInfoRow('Organization', 'com.groundnutai'),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            const Text(
              'API Configuration',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildInfoRow('Base URL', ApiConfig.baseUrl),
                  _buildInfoRow('Timeout', '${ApiConfig.connectTimeout ~/ 1000} seconds'),
                  _buildInfoRow('API Key', ApiConfig.apiKey.isNotEmpty ? 'Configured' : 'Not set'),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            const Text(
              'About',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Groundnut AI',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'An AI-powered application for groundnut disease detection, '
                    'weed management, and herbicide recommendations.',
                    style: TextStyle(
                      color: AppTheme.textSecondary,
                      height: 1.5,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Features:',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  _buildFeature('• Object detection using YOLO'),
                  _buildFeature('• Herbicide recommendations'),
                  _buildFeature('• Explainable AI (Grad-CAM)'),
                  _buildFeature('• Detection history'),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            AppCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.warning_amber_outlined, color: AppTheme.warning),
                      const SizedBox(width: 8),
                      const Text(
                        'Disclaimer',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: AppTheme.warning,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'This application provides AI-powered recommendations for '
                    'informational purposes only. Always consult with agricultural '
                    'experts before making decisions about herbicide applications.',
                    style: TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 12,
                      height: 1.4,
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

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              '$label:',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: AppTheme.textSecondary,
              ),
            ),
          ),
          Expanded(
            child: Text(value),
          ),
        ],
      ),
    );
  }

  Widget _buildFeature(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Text(
        text,
        style: TextStyle(
          color: AppTheme.textSecondary,
        ),
      ),
    );
  }
}
