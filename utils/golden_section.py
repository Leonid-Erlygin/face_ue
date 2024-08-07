# Golden-section search method
import math


# def cal_f(_x):  # 4x - 1.8x^2 + 1.2x^3 - 0.3x^4
#     return 4 * _x - 1.8 * _x ** 2 + 1.2 * _x ** 3 - 0.3 * _x ** 4


def cal_d(_xu, _xl):
    R = (math.sqrt(5) - 1) * 0.5
    return R * (_xu - _xl)


def check_e(_xu, _xl, _xopt, _e, verbose):
    R = (math.sqrt(5) - 1) * 0.5
    _ea = (1 - R) * abs((_xu - _xl) / _xopt)
    if verbose:
        print("Ea = {} %".format(_ea * 100))
    if _ea < _e:
        return 1
    return 0


def golden_selection_search(_xu, _xl, _e, _n, cal_f, verbose=False):
    _x = [0, 0]
    _f = [0, 0]
    _result = 0
    for i in range(0, _n):
        _d = cal_d(_xu, _xl)
        _x[0] = _xl + _d
        _x[1] = _xu - _d
        _f[0] = cal_f(_x[0])
        _f[1] = cal_f(_x[1])
        if _f[0] > _f[1]:
            _xl = _x[1]
            _result = _x[0]
            if check_e(_xu, _xl, _x[0], _e, verbose):
                break
        else:
            _xu = _x[0]
            _result = _x[1]
            if check_e(_xu, _xl, _x[1], _e, verbose):
                break
        if verbose:
            print(
                "Iteration {}: xL = {}\t xU = {}\t x1 = {}\t fx1 = {}\t x2 = {}\t fx2 = {}\t d = {}\t".format(
                    i,
                    round(_xl, 5),
                    round(_xu, 5),
                    round(_x[0], 5),
                    round(_f[0], 5),
                    round(_x[1], 5),
                    round(_f[1], 5),
                    round(_d, 5),
                )
            )
    return _result


if __name__ == "__main__":
    xU = 4
    xL = 2
    eS = 1 / 100
    n = 100
    result = golden_selection_search(
        xU, xL, eS, n, lambda _x: 4 * _x - 1.8 * _x**2 + 1.2 * _x**3 - 0.3 * _x**4
    )
    print("x = {}".format(result))
