import pandas as pd

from src.get_train_test.prepare_fashion_mnist_dataset import FashionMnistData


# This script answers the question: Does our data-set need cleaning?
# The answer is no

# Note cleaning is not referring to anything else than verifying the integrity
# of the data-set (missing values, correct ranges of values)

def main():
    data = FashionMnistData(original_data_set=True)
    train, test = data.get()
    print(f"Does train have missing values? {_check_missing_pixels(train)}")
    print(f"Does test have missing values? {_check_missing_pixels(test)}")
    print(f"Does train have correct pixels values? {_are_pixel_values_between_0_255(train)}")
    print(f"Does test have correct pixels values? {_are_pixel_values_between_0_255(test)}")


def _check_missing_pixels(data):
    df = pd.DataFrame(data)
    return df.isna().any().any()


def _are_pixel_values_between_0_255(data):
    gt_255 = data[data > 255]
    lt_0 = data[data < 0]

    return (len(gt_255) == 0) and (len(lt_0) == 0)


if __name__ == "__main__":
    main()
