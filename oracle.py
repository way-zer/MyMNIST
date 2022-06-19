from torchvision.datasets import MNIST


class OracleMNIST(MNIST):
    mirrors = [
        "https://raw.github.com/wm-bupt/oracle-mnist/main/data/oracle/",
        "https://raw.fastgit.org/wm-bupt/oracle-mnist/main/data/oracle/",
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", None),
        ("train-labels-idx1-ubyte.gz", None),
        ("t10k-images-idx3-ubyte.gz", None),
        ("t10k-labels-idx1-ubyte.gz", None),
    ]
    classes = [
        "0 - big - 大",
        "1 - sun - 日",
        "2 - moon - 月",
        "3 - cattle - 牛",
        "4 - next - 翌",
        "5 - field - 田",
        "6 - not - 勿",
        "7 - arrow - 矢",
        "8 - time 9-11 am - 巳",
        "9 - wood - 木",
    ]
