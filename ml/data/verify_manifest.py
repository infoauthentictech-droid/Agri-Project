"""Verify dataset manifest."""
import json
from pathlib import Path

manifest_path = Path('datasets/dataset_manifest.json')
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Dataset Manifest Summary:")
    print("=" * 60)
    print(f"Timestamp: {manifest['metadata']['timestamp']}")
    print(f"Random Seed: {manifest['metadata']['random_seed']}")
    print(f"Splits: {manifest['metadata']['splits']}")
    print(f"Augmentation Factor: {manifest['metadata']['augmentation_factor']}")
    print()
    print(f"Validation Errors: {len(manifest['validation_errors'])}")
    if manifest['validation_errors']:
        print("  Errors:")
        for err in manifest['validation_errors'][:5]:
            print(f"    - {err['file']}: {err['error']}")
    print()
    print(f"Duplicate Groups: {sum(len(d) for d in manifest['duplicates'].values())}")
    print()
    print("Dataset Breakdown:")
    for dataset_type, data in manifest['dataset_types'].items():
        print(f"\n{dataset_type.upper()}:")
        for task, counts in data.items():
            if isinstance(counts, dict):
                print(f"  {task}:")
                for class_name, count_data in counts.items():
                    if isinstance(count_data, dict):
                        print(f"    {class_name}: {count_data.get('total', 0)} total")
                        for split, split_count in count_data.get('splits', {}).items():
                            print(f"      {split}: {split_count}")
                    else:
                        print(f"    {class_name}: {count_data}")
            else:
                print(f"  {task}: {counts}")
else:
    print("ERROR: Manifest not found!")
