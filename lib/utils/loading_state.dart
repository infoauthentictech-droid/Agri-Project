import 'package:flutter/material.dart';
import '../core/app_theme.dart';

enum LoadingState {
  initial,
  loading,
  success,
  error,
}

class LoadingWidget extends StatelessWidget {
  final String? message;
  final bool showBackground;
  
  const LoadingWidget({
    this.message,
    this.showBackground = true,
    super.key,
  });
  
  @override
  Widget build(BuildContext context) {
    final content = Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(
          width: 48,
          height: 48,
          child: CircularProgressIndicator(
            strokeWidth: 3,
            valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primaryGreen),
          ),
        ),
        if (message != null) ...[
          const SizedBox(height: AppTheme.spacingMd),
          Text(
            message!,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              color: AppTheme.textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ],
    );
    
    if (showBackground) {
      return Container(
        color: Colors.white,
        child: Center(child: content),
      );
    }
    
    return Center(child: content);
  }
}
