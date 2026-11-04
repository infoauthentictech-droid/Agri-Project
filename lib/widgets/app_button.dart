import 'package:flutter/material.dart';
import '../core/app_theme.dart';

class AppButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final bool isLoading;
  final bool isSecondary;
  final bool isDestructive;
  final IconData? icon;
  final double? width;
  
  const AppButton({
    required this.text,
    this.onPressed,
    this.isLoading = false,
    this.isSecondary = false,
    this.isDestructive = false,
    this.icon,
    this.width,
    super.key,
  });
  
  @override
  Widget build(BuildContext context) {
    final buttonWidth = width ?? double.infinity;
    
    if (isSecondary) {
      return SizedBox(
        width: buttonWidth,
        child: OutlinedButton(
          onPressed: isLoading ? null : onPressed,
          style: OutlinedButton.styleFrom(
            foregroundColor: isDestructive ? AppTheme.error : AppTheme.primaryGreen,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            ),
            side: BorderSide(
              color: isDestructive ? AppTheme.error : AppTheme.primaryGreen,
              width: 1.5,
            ),
            textStyle: const TextStyle(
              fontSize: AppTheme.fontSizeMd,
              fontWeight: FontWeight.w600,
            ),
          ),
          child: _buildContent(),
        ),
      );
    }
    
    return SizedBox(
      width: buttonWidth,
      child: ElevatedButton(
        onPressed: isLoading ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isDestructive ? AppTheme.error : AppTheme.primaryGreen,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
          ),
          textStyle: const TextStyle(
            fontSize: AppTheme.fontSizeMd,
            fontWeight: FontWeight.w600,
          ),
          elevation: 2,
        ),
        child: _buildContent(),
      ),
    );
  }
  
  Widget _buildContent() {
    if (isLoading) {
      return const SizedBox(
        height: 20,
        width: 20,
        child: CircularProgressIndicator(
          strokeWidth: 2,
          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
        ),
      );
    }
    
    if (icon != null) {
      // FittedBox scales the icon+label down when the button is narrower than
      // its natural content width. Without it, long labels overflow ("RenderFlex
      // overflowed by N pixels on the right") on small phones / two-button rows.
      return FittedBox(
        fit: BoxFit.scaleDown,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20),
            const SizedBox(width: AppTheme.spacingXs),
            Text(text),
          ],
        ),
      );
    }

    return Text(text);
  }
}
