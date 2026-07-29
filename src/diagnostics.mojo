"""Posterior diagnostic kernels exposed through a stable C ABI."""

from std.algorithm import parallelize
from std.math import cos, log, sin, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.14159265358979323846264338327950288


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def sum_values(values: Ptr, start: Int, count: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var vector_sum = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= count:
        vector_sum += values.load[width=W](start + i)
        i += W
    var scalar_sum = vector_sum.reduce_add()
    while i < count:
        scalar_sum += values[start + i]
        i += 1
    return scalar_sum


def swap(values: Ptr, indices: Ptr, i: Int, j: Int):
    var value = values[i]
    values[i] = values[j]
    values[j] = value
    var index = indices[i]
    indices[i] = indices[j]
    indices[j] = index


def quick_sort(values: Ptr, indices: Ptr, stack: Ptr, n: Int, paired: Bool):
    if n <= 1:
        return
    var top = 2
    stack[0] = 0.0
    stack[1] = Float64(n - 1)
    while top > 0:
        top -= 2
        var low = Int(stack[top])
        var high = Int(stack[top + 1])
        while low < high:
            var i = low
            var j = high
            var pivot = values[(low + high) // 2]
            while i <= j:
                while values[i] < pivot:
                    i += 1
                while values[j] > pivot:
                    j -= 1
                if i <= j:
                    if paired:
                        swap(values, indices, i, j)
                    else:
                        var value = values[i]
                        values[i] = values[j]
                        values[j] = value
                    i += 1
                    j -= 1
            # Continue with the smaller side and stack the larger side.
            if j - low < high - i:
                if i < high:
                    stack[top] = Float64(i)
                    stack[top + 1] = Float64(high)
                    top += 2
                high = j
            else:
                if low < j:
                    stack[top] = Float64(low)
                    stack[top + 1] = Float64(j)
                    top += 2
                low = i


def normal_ppf(prob: Float64) -> Float64:
    """Acklam's inverse-normal approximation (absolute error below 1e-9)."""
    if prob < 0.02425:
        var q = sqrt(-2.0 * log(prob))
        return (
            (
                (
                    (
                        (-0.007784894002430293 * q - 0.3223964580411365) * q
                        - 2.400758277161838
                    )
                    * q
                    - 2.549732539343734
                )
                * q
                + 4.374664141464968
            )
            * q
            + 2.938163982698783
        ) / (
            (
                (
                    (0.007784695709041462 * q + 0.3224671290700398) * q
                    + 2.445134137142996
                )
                * q
                + 3.754408661907416
            )
            * q
            + 1.0
        )
    if prob > 0.97575:
        var q = sqrt(-2.0 * log(1.0 - prob))
        return -(
            (
                (
                    (
                        (
                            (-0.007784894002430293 * q - 0.3223964580411365) * q
                            - 2.400758277161838
                        )
                        * q
                        - 2.549732539343734
                    )
                    * q
                    + 4.374664141464968
                )
                * q
                + 2.938163982698783
            )
            / (
                (
                    (
                        (0.007784695709041462 * q + 0.3224671290700398) * q
                        + 2.445134137142996
                    )
                    * q
                    + 3.754408661907416
                )
                * q
                + 1.0
            )
        )
    var q = prob - 0.5
    var r = q * q
    return (
        (
            (
                (
                    (
                        (-39.69683028665376 * r + 220.9460984245205) * r
                        - 275.9285104469687
                    )
                    * r
                    + 138.3577518672690
                )
                * r
                - 30.66479806614716
            )
            * r
            + 2.506628277459239
        )
        * q
        / (
            (
                (
                    (
                        (-54.47609879822406 * r + 161.5858368580409) * r
                        - 155.6989798598866
                    )
                    * r
                    + 66.80131188771972
                )
                * r
                - 13.28068155288572
            )
            * r
            + 1.0
        )
    )


def fft(real: Ptr, imag: Ptr, n: Int, inverse: Bool):
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while (j & bit) != 0:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            var tr = real[i]
            real[i] = real[j]
            real[j] = tr
            var ti = imag[i]
            imag[i] = imag[j]
            imag[j] = ti

    var length = 2
    while length <= n:
        var angle = (2.0 if inverse else -2.0) * PI / Float64(length)
        var step_r = cos(angle)
        var step_i = sin(angle)
        var base = 0
        while base < n:
            var wr = 1.0
            var wi = 0.0
            for offset in range(length // 2):
                var even = base + offset
                var odd = even + length // 2
                var tr = wr * real[odd] - wi * imag[odd]
                var ti = wr * imag[odd] + wi * real[odd]
                var er = real[even]
                var ei = imag[even]
                real[even] = er + tr
                imag[even] = ei + ti
                real[odd] = er - tr
                imag[odd] = ei - ti
                var next_wr = wr * step_r - wi * step_i
                wi = wr * step_i + wi * step_r
                wr = next_wr
            base += length
        length <<= 1

    if inverse:
        var scale = 1.0 / Float64(n)
        comptime W = simd_width_of[DType.float64]()
        var i = 0
        while i + W <= n:
            real.store(i, real.load[width=W](i) * scale)
            imag.store(i, imag.load[width=W](i) * scale)
            i += W
        while i < n:
            real[i] *= scale
            imag[i] *= scale
            i += 1


def core_rhat(values: Ptr, chains: Int, draws: Int) -> Float64:
    var chain_mean_mean = 0.0
    var within = 0.0
    for chain in range(chains):
        var mean = 0.0
        for draw in range(draws):
            mean += values[chain * draws + draw]
        mean /= Float64(draws)
        chain_mean_mean += mean
        var variance = 0.0
        for draw in range(draws):
            var delta = values[chain * draws + draw] - mean
            variance += delta * delta
        within += variance / Float64(draws - 1)
    chain_mean_mean /= Float64(chains)
    within /= Float64(chains)

    var between = 0.0
    for chain in range(chains):
        var mean = 0.0
        for draw in range(draws):
            mean += values[chain * draws + draw]
        mean /= Float64(draws)
        var delta = mean - chain_mean_mean
        between += delta * delta
    between *= Float64(draws) / Float64(chains - 1)
    return sqrt((between / within + Float64(draws - 1)) / Float64(draws))


def rank_normalize_impl(
    source: Ptr,
    sorted_values: Ptr,
    indices: Ptr,
    dest: Ptr,
    n: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        sorted_values.store(i, source.load[width=W](i))
        var index_values = SIMD[DType.float64, W]()
        comptime for lane in range(W):
            index_values[lane] = Float64(i + lane)
        indices.store(i, index_values)
        i += W
    while i < n:
        sorted_values[i] = source[i]
        indices[i] = Float64(i)
        i += 1
    quick_sort(sorted_values, indices, dest, n, True)
    var first = 0
    while first < n:
        var last = first + 1
        while last < n and sorted_values[last] == sorted_values[first]:
            last += 1
        var average_rank = (Float64(first + 1) + Float64(last)) / 2.0
        var probability = (average_rank - 0.375) / (Float64(n) + 0.25)
        var z = normal_ppf(probability)
        for i in range(first, last):
            dest[Int(indices[i])] = z
        first = last


@export("mav_rank_normalize")
def mav_rank_normalize(
    source_addr: Int,
    sorted_addr: Int,
    indices_addr: Int,
    dest_addr: Int,
    n: Int,
) abi("C"):
    rank_normalize_impl(
        p(source_addr),
        p(sorted_addr),
        p(indices_addr),
        p(dest_addr),
        n,
    )


@export("mav_rank_normalize_pair")
def mav_rank_normalize_pair(
    first_source_addr: Int,
    first_sorted_addr: Int,
    first_indices_addr: Int,
    first_dest_addr: Int,
    second_source_addr: Int,
    second_sorted_addr: Int,
    second_indices_addr: Int,
    second_dest_addr: Int,
    n: Int,
) abi("C"):
    @parameter
    def normalize(index: Int):
        if index == 0:
            rank_normalize_impl(
                p(first_source_addr),
                p(first_sorted_addr),
                p(first_indices_addr),
                p(first_dest_addr),
                n,
            )
        else:
            rank_normalize_impl(
                p(second_source_addr),
                p(second_sorted_addr),
                p(second_indices_addr),
                p(second_dest_addr),
                n,
            )

    if n >= 65536:
        parallelize[normalize](2, 2)
    else:
        normalize(0)
        normalize(1)


@export("mav_rhat")
def mav_rhat(values_addr: Int, chains: Int, draws: Int) abi("C") -> Float64:
    return core_rhat(p(values_addr), chains, draws)


def ess_finish(
    values: Ptr,
    acov: Ptr,
    rho: Ptr,
    chains: Int,
    draws: Int,
    acov_stride: Int,
    relative: Int,
) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var mean_acov0 = 0.0
    var chain_mean_mean = 0.0
    for chain in range(chains):
        mean_acov0 += acov[chain * acov_stride]
        var mean = sum_values(values, chain * draws, draws)
        chain_mean_mean += mean / Float64(draws)
    mean_acov0 /= Float64(chains)
    chain_mean_mean /= Float64(chains)
    var mean_var = mean_acov0 * Float64(draws) / Float64(draws - 1)
    var var_plus = mean_var * Float64(draws - 1) / Float64(draws)
    if chains > 1:
        var chain_mean_var = 0.0
        for chain in range(chains):
            var mean = sum_values(values, chain * draws, draws) / Float64(draws)
            var delta = mean - chain_mean_mean
            chain_mean_var += delta * delta
        var_plus += chain_mean_var / Float64(chains - 1)

    var rho_index = 0
    while rho_index + W <= draws:
        rho.store(rho_index, SIMD[DType.float64, W](0.0))
        rho_index += W
    while rho_index < draws:
        rho[rho_index] = 0.0
        rho_index += 1
    rho[0] = 1.0
    var mean_acov1 = 0.0
    for chain in range(chains):
        mean_acov1 += acov[chain * acov_stride + 1]
    mean_acov1 /= Float64(chains)
    var rho_even = 1.0
    var rho_odd = 1.0 - (mean_var - mean_acov1) / var_plus
    rho[1] = rho_odd

    var t = 1
    while t < draws - 3 and rho_even + rho_odd > 0.0:
        var acov_even = 0.0
        var acov_odd = 0.0
        for chain in range(chains):
            acov_even += acov[chain * acov_stride + t + 1]
            acov_odd += acov[chain * acov_stride + t + 2]
        acov_even /= Float64(chains)
        acov_odd /= Float64(chains)
        rho_even = 1.0 - (mean_var - acov_even) / var_plus
        rho_odd = 1.0 - (mean_var - acov_odd) / var_plus
        if rho_even + rho_odd >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2

    var max_t = t - 2
    if rho_even > 0.0:
        rho[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:
        var current_pair = rho[t + 1] + rho[t + 2]
        var previous_pair = rho[t - 1] + rho[t]
        if current_pair > previous_pair:
            rho[t + 1] = previous_pair / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2

    var tau = -1.0
    for i in range(max_t + 1):
        tau += 2.0 * rho[i]
    tau += rho[max_t + 1]
    var sample_count = Float64(chains * draws)
    var lower_bound = 1.0 / (log(sample_count) / log(10.0))
    if tau < lower_bound:
        tau = lower_bound
    return (1.0 if relative != 0 else sample_count) / tau


@export("mav_ess")
def mav_ess(
    values_addr: Int,
    acov_addr: Int,
    real_addr: Int,
    imag_addr: Int,
    rho_addr: Int,
    chains: Int,
    draws: Int,
    fft_n: Int,
    relative: Int,
) abi("C") -> Float64:
    var values = p(values_addr)
    var acov = p(acov_addr)
    var real = p(real_addr)
    var imag = p(imag_addr)
    var rho = p(rho_addr)

    var minimum = values[0]
    var maximum = values[0]
    for i in range(1, chains * draws):
        if values[i] < minimum:
            minimum = values[i]
        if values[i] > maximum:
            maximum = values[i]
    if maximum - minimum < 1.0e-15:
        return Float64(chains * draws)

    comptime W = simd_width_of[DType.float64]()

    @parameter
    def process_chain(chain: Int):
        var mean = sum_values(values, chain * draws, draws) / Float64(draws)
        var chain_real = real + chain * fft_n
        var chain_imag = imag + chain * fft_n
        var i = 0
        while i + W <= fft_n:
            chain_real.store(i, SIMD[DType.float64, W](0.0))
            chain_imag.store(i, SIMD[DType.float64, W](0.0))
            i += W
        while i < fft_n:
            chain_real[i] = 0.0
            chain_imag[i] = 0.0
            i += 1
        var draw = 0
        while draw + W <= draws:
            chain_real.store(
                draw,
                values.load[width=W](chain * draws + draw) - mean,
            )
            draw += W
        while draw < draws:
            chain_real[draw] = values[chain * draws + draw] - mean
            draw += 1
        fft(chain_real, chain_imag, fft_n, False)
        i = 0
        while i + W <= fft_n:
            var real_values = chain_real.load[width=W](i)
            var imag_values = chain_imag.load[width=W](i)
            chain_real.store(
                i,
                real_values * real_values + imag_values * imag_values,
            )
            chain_imag.store(i, SIMD[DType.float64, W](0.0))
            i += W
        while i < fft_n:
            chain_real[i] = (
                chain_real[i] * chain_real[i] + chain_imag[i] * chain_imag[i]
            )
            chain_imag[i] = 0.0
            i += 1
        fft(chain_real, chain_imag, fft_n, True)
        var lag = 0
        var scale = 1.0 / Float64(draws)
        while lag + W <= draws:
            acov.store(
                chain * draws + lag,
                chain_real.load[width=W](lag) * scale,
            )
            lag += W
        while lag < draws:
            acov[chain * draws + lag] = chain_real[lag] * scale
            lag += 1

    if chains > 1 and chains * draws >= 32768:
        parallelize[process_chain](chains, min(chains, 8))
    else:
        for chain in range(chains):
            process_chain(chain)

    return ess_finish(values, acov, rho, chains, draws, draws, relative)


@export("mav_ess_from_acov")
def mav_ess_from_acov(
    values_addr: Int,
    acov_addr: Int,
    rho_addr: Int,
    chains: Int,
    draws: Int,
    acov_stride: Int,
    relative: Int,
) abi("C") -> Float64:
    return ess_finish(
        p(values_addr),
        p(acov_addr),
        p(rho_addr),
        chains,
        draws,
        acov_stride,
        relative,
    )


@export("mav_hdi_nearest")
def mav_hdi_nearest(
    work_addr: Int,
    dest_addr: Int,
    n: Int,
    interval_idx: Int,
) abi("C"):
    var sorted_values = p(work_addr)
    var dest = p(dest_addr)
    var n_intervals = n - interval_idx
    var best = 0
    var best_width = sorted_values[interval_idx] - sorted_values[0]
    for i in range(1, n_intervals):
        var width = sorted_values[i + interval_idx] - sorted_values[i]
        if width < best_width:
            best = i
            best_width = width
    dest[0] = sorted_values[best]
    dest[1] = sorted_values[best + interval_idx]
