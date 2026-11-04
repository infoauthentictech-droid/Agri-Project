import 'package:flutter/material.dart';
import '../core/app_theme.dart';
import '../widgets/app_button.dart';

class ErrorStateWidget extends StatelessWidget {
  final String message;
  final String? subMessage;
  final IconData? icon;
  final Widget? action;
  final VoidCallback? onRetry;
  final Color? iconColor;
  
  const ErrorStateWidget({
    required this.message,
    this.subMessage,
    this.icon,
    this.action,
    this.onRetry,
    this.iconColor,
    super.key,
  });
  
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppTheme.spacingXl),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(
                color: (iconColor ?? AppTheme.error).withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                icon ?? Icons.error_outline,
                size: 48,
                color: iconColor ?? AppTheme.error,
              ),
            ),
            const SizedBox(height: AppTheme.spacingLg),
            Text(
              message,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                color: AppTheme.textPrimary,
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
            ),
            if (subMessage != null) ...[
              const SizedBox(height: AppTheme.spacingXs),
              Text(
                subMessage!,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppTheme.textSecondary,
                ),
                textAlign: TextAlign.center,
              ),
            ],
            const SizedBox(height: AppTheme.spacingLg),
            if (onRetry != null)
              AppButton(
                text: 'Retry',
                onPressed: onRetry,
                icon: Icons.refresh,
                width: 150,
              ),
            if (action != null) ...[
              const SizedBox(height: AppTheme.spacingMd),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}
