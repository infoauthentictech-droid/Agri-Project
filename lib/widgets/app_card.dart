import 'package:flutter/material.dart';
import '../core/app_theme.dart';

class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final VoidCallback? onTap;
  final Color? color;
  final Color? borderColor;
  final double? elevation;
  final bool hasShadow;
  
  const AppCard({
    required this.child,
    this.padding,
    this.margin,
    this.onTap,
    this.color,
    this.borderColor,
    this.elevation,
    this.hasShadow = false,
    super.key,
  });
  
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: margin ?? const EdgeInsets.symmetric(horizontal: AppTheme.spacingMd, vertical: AppTheme.spacingXs),
      color: color,
      elevation: elevation ?? (hasShadow ? 2 : 0),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
        side: borderColor != null
            ? BorderSide(color: borderColor!, width: 1)
            : BorderSide.none,
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(AppTheme.spacingMd),
          child: child,
        ),
      ),
    );
  }
}
