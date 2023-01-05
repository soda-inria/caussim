from typing import Dict


def dic2str(dic: Dict) -> str:
    return "_".join([f"{k}={v}" for k, v in dic.items() if v is not None])


def cantor_pairing(n: int, m: int) -> int:
    """
    [Cantor pairing function](https://en.wikipedia.org/wiki/Pairing_function) : Generate a unique int from two int.
    Base on [this blog post](https://odino.org/combining-two-numbers-into-a-unique-one-pairing-functions/)

    NB: it is invertible (not needed for me)

    Args:
        n (_type_): _description_
        m (_type_): _description_
    """
    return (n + m) * (n + m + 1) // 2 + m
