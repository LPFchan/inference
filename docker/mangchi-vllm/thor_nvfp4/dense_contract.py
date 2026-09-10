"""Dependency-free exact Qwen3.8-27B TP=1 dense shape contract."""

SHAPES = {(34816, 5120), (5120, 17408), (14336, 5120),
          (5120, 6144), (16384, 5120), (96, 5120)}
LOGICAL_WIDTHS = {(34816, 5120): (17408, 17408), (5120, 17408): (5120,),
                  (14336, 5120): (12288, 1024, 1024), (5120, 6144): (5120,),
                  (16384, 5120): (2048, 2048, 6144, 6144), (96, 5120): (48, 48)}


def use_cutlass(m, n, k, uniform_alpha):
    validate_shape(n, k, [n])
    if type(uniform_alpha) is not bool or not 0 <= m <= 262144:
        raise ValueError("Invalid dense hybrid dispatch metadata")
    if uniform_alpha and n == 96:
        raise ValueError("Padded B/A requires per-column CuTe alpha")
    return uniform_alpha and m >= 1568


def validate_alpha_layout(n, k, widths):
    """Every broadcast tile must remain within one checkpoint-global slice."""
    validate_shape(n, k, widths)
    if tuple(widths) != LOGICAL_WIDTHS[(n, k)]:
        raise ValueError("Unaudited logical projection layout")
    if (n, k) == (96, 5120):
        return False
    boundary = 0
    for width in widths:
        boundary += width
        if any(boundary % tile for tile in (128, 256)):
            raise ValueError("Logical alpha boundary crosses a reachable dense tile")
    return True


def select_tile_n(m, n, k):
    """Measured policy covering the scheduler's 1568-token aligned chunks."""
    validate_shape(n, k, [n])
    if m < 0 or m > 262144:
        raise ValueError("Dense M must be in [0,262144]")
    if (n, k) == (5120, 6144) or (m >= 512 and n % 256 == 0):
        return 256
    return 128


def validate_shape(n, k, widths):
    if (n, k) not in SHAPES:
        raise ValueError(f"Unaudited Thor dense NVFP4 shape N={n}, K={k}")
    if not widths or any(type(w) is not int or w <= 0 for w in widths) or sum(widths) != n:
        raise ValueError("Logical projection widths must partition N exactly")
    return (n + 127) // 128 * 128
