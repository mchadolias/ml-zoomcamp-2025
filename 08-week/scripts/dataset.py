import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HairTypeDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        """
        Dataset for hair type classification.

        Args:
            data_dir (str): Path to the directory containing class subdirectories
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on labels
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.labels = []

        # Validate directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Get classes and create mapping
        self.classes = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )

        if not self.classes:
            raise ValueError(f"No class directories found in {data_dir}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # Load image paths and labels
        self._load_data()

        # Validate we have data
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def _load_data(self):
        """Load all image paths and corresponding labels."""
        for label_name in self.classes:
            label_dir = os.path.join(self.data_dir, label_name)

            # Skip if not a directory
            if not os.path.isdir(label_dir):
                continue

            # Get all image files
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            image_files = [
                f
                for f in os.listdir(label_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            for img_name in image_files:
                img_path = os.path.join(label_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                label = self.target_transform(label)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or raise exception based on your needs
            raise

        return image, label

    def get_class_distribution(self):
        """Return the distribution of classes in the dataset."""
        from collections import Counter

        return Counter(self.labels)

    def get_class_names(self):
        """Return list of class names."""
        return self.classes

    def get_class_mapping(self):
        """Return the class to index mapping."""
        return self.class_to_idx.copy()

    def get_sample_by_class(self, class_name):
        """Get all samples for a specific class."""
        if class_name not in self.class_to_idx:
            raise ValueError(f"Class {class_name} not found in dataset")

        class_idx = self.class_to_idx[class_name]
        indices = [i for i, label in enumerate(self.labels) if label == class_idx]
        return [self.image_paths[i] for i in indices]


def create_dataset_summary(dataset, include_sample_images=True, max_sample_preview=3):
    """
    Create a comprehensive summary of the HairTypeDataset.

    Args:
        dataset (HairTypeDataset): The dataset instance to summarize
        include_sample_images (bool): Whether to include sample image paths
        max_sample_preview (int): Maximum number of sample images to show per class

    Returns:
        dict: A dictionary containing the dataset summary
    """

    summary = {
        "basic_info": {},
        "class_info": {},
        "statistics": {},
        "samples": {} if include_sample_images else None,
    }

    # Basic Information
    summary["basic_info"] = {
        "dataset_size": len(dataset),
        "number_of_classes": len(dataset.classes),
        "data_directory": dataset.data_dir,
        "has_transforms": dataset.transform is not None,
        "has_target_transforms": dataset.target_transform is not None,
    }

    # Class Information
    class_distribution = dataset.get_class_distribution()
    class_mapping = dataset.get_class_mapping()

    summary["class_info"] = {
        "class_names": dataset.get_class_names(),
        "class_to_idx_mapping": class_mapping,
        "class_distribution": dict(class_distribution),
        "class_balance_ratio": {
            dataset.idx_to_class[idx]: count / len(dataset)
            for idx, count in class_distribution.items()
        },
    }

    # Statistics
    summary["statistics"] = {
        "total_samples": len(dataset),
        "samples_per_class_avg": len(dataset) / len(dataset.classes),
        "samples_per_class_min": min(class_distribution.values()),
        "samples_per_class_max": max(class_distribution.values()),
        "most_common_class": dataset.idx_to_class[
            class_distribution.most_common(1)[0][0]
        ],
        "least_common_class": dataset.idx_to_class[
            class_distribution.most_common()[-1][0]
        ],
    }

    # Sample Information
    if include_sample_images:
        for class_name in dataset.get_class_names():
            class_samples = dataset.get_sample_by_class(class_name)
            preview_samples = class_samples[:max_sample_preview]
            summary["samples"][class_name] = {
                "total_samples": len(class_samples),
                "sample_paths": preview_samples,
                "preview_count": len(preview_samples),
            }

    return summary


def print_detailed_summary(summary):
    """
    Print a formatted, human-readable version of the dataset summary.

    Args:
        summary (dict): The summary dictionary from create_dataset_summary
    """
    print("=" * 60)
    print("HAIR TYPE DATASET SUMMARY")
    print("=" * 60)

    # Basic Information
    print("\n📊 BASIC INFORMATION")
    print("-" * 40)
    basic = summary["basic_info"]
    print(f"• Dataset Size: {basic['dataset_size']:,} samples")
    print(f"• Number of Classes: {basic['number_of_classes']}")
    print(f"• Data Directory: {basic['data_directory']}")
    print(f"• Image Transforms: {'Yes' if basic['has_transforms'] else 'No'}")
    print(f"• Label Transforms: {'Yes' if basic['has_target_transforms'] else 'No'}")

    # Class Distribution
    print("\n🏷️  CLASS INFORMATION")
    print("-" * 40)
    class_info = summary["class_info"]
    distribution = class_info["class_distribution"]

    print("Class Distribution:")
    for class_name in class_info["class_names"]:
        class_idx = class_info["class_to_idx_mapping"][class_name]
        count = distribution[class_idx]
        percentage = (count / summary["basic_info"]["dataset_size"]) * 100
        print(f"  {class_name:<15}: {count:>4} samples ({percentage:>5.1f}%)")

    # Statistics
    print("\n📈 STATISTICS")
    print("-" * 40)
    stats = summary["statistics"]
    print(f"• Average samples per class: {stats['samples_per_class_avg']:.1f}")
    print(f"• Min samples per class: {stats['samples_per_class_min']}")
    print(f"• Max samples per class: {stats['samples_per_class_max']}")
    print(f"• Most common class: {stats['most_common_class']}")
    print(f"• Least common class: {stats['least_common_class']}")

    # Sample Preview
    if summary["samples"]:
        print("\n🖼️  SAMPLE PREVIEW")
        print("-" * 40)
        for class_name, sample_info in summary["samples"].items():
            print(f"{class_name}:")
            print(f"  Total: {sample_info['total_samples']} samples")
            print(f"  Preview ({sample_info['preview_count']} samples):")
            for i, path in enumerate(sample_info["sample_paths"]):
                print(f"    {i+1}. {os.path.basename(path)}")
            print()

    print("=" * 60)


if __name__ == "__main__":
    # Test the dataset class
    print("🧪 Testing HairTypeDataset class...")

    try:
        # Create a simple transform for testing
        test_transform = transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
            ]
        )

        # Try to create dataset instance
        dataset = HairTypeDataset(
            data_dir="data/dataset_hw8/train", transform=test_transform
        )
        print("✅ Dataset created successfully!")

        # Test basic functionality
        print(f"📊 Dataset size: {len(dataset)}")
        print(f"🏷️  Classes: {dataset.get_class_names()}")
        print(f"📈 Class distribution: {dict(dataset.get_class_distribution())}")

        # Test getting a sample
        sample, label = dataset[0]
        print(f"🖼️  Sample shape: {sample.shape}")
        print(f"🔢 Label: {label}")

        # Test summary functions
        print("\n📋 Testing summary functions...")
        summary = create_dataset_summary(
            dataset, include_sample_images=True, max_sample_preview=2
        )
        print_detailed_summary(summary)

        print("\n🎉 All tests passed! The dataset is working correctly.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("💡 Make sure your data directory structure is correct")
