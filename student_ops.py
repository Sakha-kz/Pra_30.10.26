"""
Модуль эталонных операций компьютерного зрения и сверточных нейронных сетей (CNN)
для практикума курса «Математические методы компьютерного зрения и CNN» (Satbayev University).
"""

from typing import Tuple, Union, Optional
import numpy as np
import cv2
import scipy.signal


def centered_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Вычисляет центрированное косинусное сходство (коэффициент корреляции Пирсона / NCC)
    между двумя массивами (изображениями или патчами).
    
    Формула:
        sim = sum((A - mean(A)) * (B - mean(B))) / (sqrt(sum((A - mean(A))^2)) * sqrt(sum((B - mean(B))^2)))
    
    Args:
        A: Первый массив (произвольной формы).
        B: Второй массив (той же формы, что и A).
        
    Returns:
        float: Значение сходства в диапазоне [-1.0, 1.0].
    """
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    if A_arr.shape != B_arr.shape:
        raise ValueError(f"Формы массивов должны совпадать: {A_arr.shape} vs {B_arr.shape}")
    
    A_diff = A_arr - np.mean(A_arr)
    B_diff = B_arr - np.mean(B_arr)
    
    denom = np.sqrt(np.sum(A_diff ** 2) * np.sum(B_diff ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(A_diff * B_diff) / denom)


def correlate_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Вычисляет 2D кросс-корреляцию в режиме 'valid' (без padding).
    
    Формула:
        Out[y, x] = sum_{i, j} image[y + i, x + j] * kernel[i, j]
        
    Размер выхода:
        (H - kH + 1, W - kW + 1)
        
    Args:
        image: Входное 2D изображение (H, W).
        kernel: 2D ядро фильтрации (kH, kW).
        
    Returns:
        np.ndarray: Результат корреляции типа float64.
    """
    img = np.asarray(image, dtype=np.float64)
    kern = np.asarray(kernel, dtype=np.float64)
    H, W = img.shape
    kH, kW = kern.shape
    
    if H < kH or W < kW:
        raise ValueError(f"Размер ядра ({kH}, {kW}) больше размера изображения ({H}, {W})")
        
    out_H = H - kH + 1
    out_W = W - kW + 1
    out = np.zeros((out_H, out_W), dtype=np.float64)
    
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.sum(img[i:i + kH, j:j + kW] * kern)
            
    return out


def linear_convolution_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Вычисляет точную линейную свёртку (режим 'full') с помощью 2D БПФ (FFT).
    
    Свойства:
        Истинная свёртка разворачивает ядро на 180°:
        Conv(I, K) = I * rot180(K)
        Вычисляется через умножение спектров на расширенной сетке (H + kH - 1, W + kW - 1)
        для предотвращения циклического наложения (aliasing / circular wrap-around).
        
    Args:
        image: 2D изображение (H, W).
        kernel: 2D ядро свёртки (kH, kW).
        
    Returns:
        np.ndarray: Результат линейной свёртки формы (H + kH - 1, W + kW - 1).
    """
    img = np.asarray(image, dtype=np.float64)
    kern = np.asarray(kernel, dtype=np.float64)
    H, W = img.shape
    kH, kW = kern.shape
    
    # Результирующий размер линейной свёртки
    out_H = H + kH - 1
    out_W = W + kW - 1
    
    # В частотной области умножение Фурье-образов F(f)*F(g) соответствует
    # дискретной линейной свёртке f * g при разложении с размером (H + kH - 1, W + kW - 1).
    F_img = np.fft.fft2(img, s=(out_H, out_W))
    F_kern = np.fft.fft2(kern, s=(out_H, out_W))
    
    res = np.fft.ifft2(F_img * F_kern).real
    return res


def gabor_kernel(ksize: int = 21,
                 sigma: float = 4.0,
                 theta: float = 0.0,
                 lambd: float = 10.0,
                 gamma: float = 0.5,
                 psi: float = 0.0) -> np.ndarray:
    """
    Генерирует 2D вещественное ядро Габора.
    
    Формула:
        x' =  x * cos(theta) + y * sin(theta)
        y' = -x * sin(theta) + y * cos(theta)
        g(x, y) = exp(- (x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2 * pi * x' / lambd + psi)
        
    Args:
        ksize: Размер ядра (нечетное число).
        sigma: Стандартное отклонение гауссовой огибающей.
        theta: Ориентация полос в радианах.
        lambd: Длина волны синусоидального множителя.
        gamma: Коэффициент сжатия (пространственное соотношение сторон).
        psi: Фазовый сдвиг синусоиды.
        
    Returns:
        np.ndarray: Ядро Габора размером (ksize, ksize) типа float64.
    """
    if ksize % 2 == 0:
        raise ValueError("ksize должен быть нечётным целым числом")
        
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    envelope = np.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / (2.0 * sigma ** 2))
    carrier = np.cos(2.0 * np.pi * x_theta / lambd + psi)
    
    kernel = envelope * carrier
    return kernel.astype(np.float64)


def binary_local(image: np.ndarray, kernel_size: int = 15, c: float = 5.0) -> np.ndarray:
    """
    Адаптивная локальная бинаризация (порог по локальному среднему).
    
    Пиксель становится 1 (или 255), если I(x, y) > mean_local(x, y) - c, иначе 0.
    
    Args:
        image: Входное полутоновое изображение (0..255 или 0..1).
        kernel_size: Размер локального окна (нечетный, например 15).
        c: Константа вычитания из локального среднего.
        
    Returns:
        np.ndarray: Бинарное изображение (0 и 1) типа uint8.
    """
    img = np.asarray(image, dtype=np.float64)
    # Вычисление локального среднего с помощью box-фильтра
    local_mean = cv2.boxFilter(img, ddepth=cv2.CV_64F, ksize=(kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    binary = (img > (local_mean - c)).astype(np.uint8)
    return binary


def grey_local(image: np.ndarray, kernel_size: int = 15, method: str = "mean") -> np.ndarray:
    """
    Локальная обработка яркости полутонового изображения (локальное среднее или локальная дисперсия/контраст).
    
    Args:
        image: Входное полутоновое изображение.
        kernel_size: Размер окна.
        method: "mean" для локального среднего или "std" для локального стандартного отклонения (контраста).
        
    Returns:
        np.ndarray: Результат локальной фильтрации типа float64.
    """
    img = np.asarray(image, dtype=np.float64)
    local_mean = cv2.boxFilter(img, ddepth=cv2.CV_64F, ksize=(kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    if method == "mean":
        return local_mean
    elif method == "std":
        local_sq_mean = cv2.boxFilter(img ** 2, ddepth=cv2.CV_64F, ksize=(kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        return np.sqrt(variance)
    else:
        raise ValueError(f"Неизвестный метод: {method}. Поддерживаются 'mean' и 'std'.")


def pool2d(image: np.ndarray,
           kernel_size: Union[int, Tuple[int, int]] = 2,
           stride: Optional[Union[int, Tuple[int, int]]] = None,
           mode: str = "max") -> np.ndarray:
    """
    2D операция пулинга (Max-pooling или Average-pooling) по 2D сетке.
    
    Args:
        image: Входной массив формы (H, W) или (C, H, W).
        kernel_size: Размер окна (kH, kW) или целое k.
        stride: Шаг окна (sH, sW) или целое s (по умолчанию равен kernel_size).
        mode: "max" или "avg".
        
    Returns:
        np.ndarray: Результат пулинга.
    """
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size
        
    if stride is None:
        sH, sW = kH, kW
    elif isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride
        
    img = np.asarray(image, dtype=np.float64)
    is_3d = (img.ndim == 3)
    if not is_3d:
        img = img[np.newaxis, ...]
        
    C, H, W = img.shape
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Размер ядра ({kH}, {kW}) слишком велик для карты ({H}, {W})")
        
    out = np.zeros((C, out_H, out_W), dtype=np.float64)
    
    for i in range(out_H):
        y_start = i * sH
        y_end = y_start + kH
        for j in range(out_W):
            x_start = j * sW
            x_end = x_start + kW
            patch = img[:, y_start:y_end, x_start:x_end]
            if mode == "max":
                out[:, i, j] = np.max(patch, axis=(1, 2))
            elif mode == "avg" or mode == "mean":
                out[:, i, j] = np.mean(patch, axis=(1, 2))
            else:
                raise ValueError(f"Неизвестный режим: {mode}. Используйте 'max' или 'avg'.")
                
    if not is_3d:
        return out[0]
    return out


def conv_output_size(input_size: Union[int, Tuple[int, int]],
                     kernel_size: Union[int, Tuple[int, int]],
                     padding: Union[int, Tuple[int, int]] = 0,
                     stride: Union[int, Tuple[int, int]] = 1,
                     dilation: Union[int, Tuple[int, int]] = 1) -> Tuple[int, int]:
    """
    Вычисляет пространственный размер выхода сверточного слоя CNN по канонической формуле:
        H_out = floor((H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        W_out = floor((W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        
    Args:
        input_size: (H_in, W_in) или H_in.
        kernel_size: (kH, kW) или k.
        padding: (pH, pW) или p.
        stride: (sH, sW) или s.
        dilation: (dH, dW) или d.
        
    Returns:
        Tuple[int, int]: (H_out, W_out).
    """
    def _to_tuple(val):
        return (val, val) if isinstance(val, int) else val
        
    Hin, Win = _to_tuple(input_size)
    kH, kW = _to_tuple(kernel_size)
    pH, pW = _to_tuple(padding)
    sH, sW = _to_tuple(stride)
    dH, dW = _to_tuple(dilation)
    
    Hout = int(np.floor((Hin + 2 * pH - dH * (kH - 1) - 1) / sH) + 1)
    Wout = int(np.floor((Win + 2 * pW - dW * (kW - 1) - 1) / sW) + 1)
    
    return (Hout, Wout)


def softmax_ce(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Численно стабильный Softmax и Cross-Entropy loss с вычислением градиента по логитам.
    
    Формулы:
        p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
        L = - sum_k target_k * log(p_k + eps)
        dL / dz = p - target
        
    Args:
        logits: Массив логитов формы (N, C) или (C,).
        targets: One-hot или целочисленные метки классов (N,) или (N, C).
        
    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            loss: Скалярное значение функции потерь (среднее по батчу).
            probs: Вероятности классов формы как у logits.
            grad: Аналитический градиент dL/d(logits) той же формы, что и logits.
    """
    z = np.asarray(logits, dtype=np.float64)
    is_1d = (z.ndim == 1)
    if is_1d:
        z = z[np.newaxis, :]
        
    N, C = z.shape
    t = np.asarray(targets)
    
    # Если переданы целочисленные метки (N,), переводим в one-hot
    if t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
        t_indices = t.ravel().astype(np.int64)
        one_hot = np.zeros((N, C), dtype=np.float64)
        one_hot[np.arange(N), t_indices] = 1.0
        t = one_hot
    else:
        t = t.astype(np.float64)
        if is_1d and t.ndim == 1:
            t = t[np.newaxis, :]
            
    # Численно стабильный softmax
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    eps = 1e-15
    loss = -np.sum(t * np.log(probs + eps)) / N
    
    # Градиент dL / dz = (probs - targets) / N
    grad = (probs - t) / N
    
    if is_1d:
        probs = probs[0]
        grad = grad[0]
        
    return float(loss), probs, grad


def correlate_backward(d_out: np.ndarray,
                       image: np.ndarray,
                       kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет обратный проход (backward pass) для операции 2D кросс-корреляции (valid mode):
        out = correlate_valid(image, kernel)
        
    Формулы градиентов:
        d_out: градиент потерь по выходу формы (out_H, out_W).
        
        dL / d(kernel)[i, j] = sum_{y, x} d_out[y, x] * image[y + i, x + j]
            Это корреляция d_out с входным изображением.
            
        dL / d(image)[y, x] = sum_{i, j} d_out[y - i, x - j] * kernel[i, j]
            (для граничных элементов сумма берется по валидным индексам,
             что эквивалентно полной свертке d_out с ядром).
             
    Args:
        d_out: Градиент потерь по выходу формы (H - kH + 1, W - kW + 1).
        image: Входное 2D изображение (H, W).
        kernel: 2D ядро фильтра (kH, kW).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            d_image: Градиент по изображению формы (H, W).
            d_kernel: Градиент по ядру формы (kH, kW).
    """
    g = np.asarray(d_out, dtype=np.float64)
    x = np.asarray(image, dtype=np.float64)
    w = np.asarray(kernel, dtype=np.float64)
    
    H, W = x.shape
    kH, kW = w.shape
    out_H, out_W = g.shape
    
    if out_H != H - kH + 1 or out_W != W - kW + 1:
        raise ValueError("Размеры d_out не согласуются с image и kernel")
        
    # 1. Градиент по ядру: dL/d(kernel)[i, j] = sum_{y, x} g[y, x] * x[y + i, x + j]
    d_kernel = np.zeros_like(w, dtype=np.float64)
    for i in range(kH):
        for j in range(kW):
            d_kernel[i, j] = np.sum(g * x[i:i + out_H, j:j + out_W])
            
    # 2. Градиент по входу: dL/d(image)[y, x]
    d_image = np.zeros_like(x, dtype=np.float64)
    for y in range(out_H):
        for x_idx in range(out_W):
            d_image[y:y + kH, x_idx:x_idx + kW] += g[y, x_idx] * w
            
    return d_image, d_kernel
