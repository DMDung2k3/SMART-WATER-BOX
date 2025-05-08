import os
import argparse
import glob
from collections import Counter


def validate_dataset(data_path):
    """
    Kiểm tra tính hợp lệ của tập dữ liệu và thống kê thông tin.

    Args:
        data_path (str): Đường dẫn đến thư mục data/ chứa train/, valid/, test/
    """
    splits = ['train', 'valid', 'test']
    stats = {}

    for split in splits:
        img_dir = os.path.join(data_path, split, 'images')
        label_dir = os.path.join(data_path, split, 'labels')

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Error: Directory {img_dir} or {label_dir} does not exist")
            continue

        # Lấy danh sách file ảnh
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

        # Kiểm tra số lượng ảnh và nhãn
        img_count = len(img_files)
        label_count = len(label_files)
        if img_count != label_count:
            print(f"Warning: {split} has {img_count} images but {label_count} labels")

        # Kiểm tra từng ảnh có nhãn tương ứng
        missing_labels = []
        class_counts = Counter()
        for img_file in img_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            label_file = os.path.join(label_dir, f"{img_name}.txt")

            if not os.path.exists(label_file):
                missing_labels.append(img_name)
                continue

            # Kiểm tra định dạng nhãn
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Error: Invalid label format in {label_file}: {line}")
                            continue
                        class_id, x_center, y_center, width, height = map(float, parts)
                        if not (0 <= class_id <= 9):  # class_id từ 0-9
                            print(f"Warning: Invalid class_id {class_id} in {label_file}")
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            print(f"Warning: Invalid coordinates in {label_file}: {line}")
                        class_counts[int(class_id)] += 1
                    except ValueError:
                        print(f"Error: Invalid number format in {label_file}: {line}")

        stats[split] = {
            'image_count': img_count,
            'label_count': label_count,
            'missing_labels': missing_labels,
            'class_distribution': class_counts
        }

    # In thống kê
    print("\nDataset Statistics:")
    for split, info in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Images: {info['image_count']}")
        print(f"  Labels: {info['label_count']}")
        if info['missing_labels']:
            print(f"  Missing labels for: {len(info['missing_labels'])} images")
            print(f"  Examples: {info['missing_labels'][:5]}")
        print(f"  Class distribution: {dict(info['class_distribution'])}")


def convert_labels_if_needed(data_path, input_format='yolo'):
    """
    Chuyển đổi nhãn nếu cần (ví dụ: từ định dạng khác sang YOLO).
    Hiện tại giả định nhãn đã ở định dạng YOLO.
    """
    print("Assuming labels are already in YOLO format (class_id, x_center, y_center, width, height).")
    print("If conversion is needed, please specify input format and implement conversion logic.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and process water meter dataset')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset directory')
    args = parser.parse_args()

    print("Validating dataset...")
    validate_dataset(args.data_path)
    print("\nChecking label format...")
    convert_labels_if_needed(args.data_path)