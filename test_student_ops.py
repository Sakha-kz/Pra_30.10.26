"""
Скрипт проверки модуля student_ops.py и сохранения артефактов визуализации
в папку output_bonus_results/
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import student_ops

def run_tests_and_visualizations():
    out_dir = Path("output_bonus_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"[1/11] Папка артефактов создана: {out_dir.resolve()}")
    
    # 1. Проверка centered_similarity
    print("\n--- Проверка 1: centered_similarity ---")
    A = np.array([1.0, 2.0, 3.0, 4.0])
    B = 2.0 * A + 5.0
    sim_identical = student_ops.centered_similarity(A, B)
    sim_opposite = student_ops.centered_similarity(A, -B)
    print(f"Сходство одинаковых (линейно зависимых): {sim_identical:.6f} (ожидается 1.0)")
    print(f"Сходство противоположных: {sim_opposite:.6f} (ожидается -1.0)")
    assert np.isclose(sim_identical, 1.0)
    assert np.isclose(sim_opposite, -1.0)
    
    # Визуализация сходства шаблона на реальном изображении
    img_scene = cv2.imread("images/templates/search_scene.png", 0).astype(np.float64) / 255.0
    img_tpl = cv2.imread("images/templates/template.png", 0).astype(np.float64) / 255.0
    th, tw = img_tpl.shape
    sh, sw = img_scene.shape
    
    # Считаем карту centered_similarity в окрестности GT (x=327, y=152)
    sample_y, sample_x = 152, 327
    patch_best = img_scene[sample_y:sample_y+th, sample_x:sample_x+tw]
    patch_random = img_scene[10:10+th, 10:10+tw]
    sim_at_gt = student_ops.centered_similarity(patch_best, img_tpl)
    sim_at_rand = student_ops.centered_similarity(patch_random, img_tpl)
    print(f"Сходство в точке GT ({sample_x}, {sample_y}): {sim_at_gt:.4f}, на случайном участке: {sim_at_rand:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(img_tpl, cmap='gray')
    axes[0].set_title("Шаблон")
    axes[1].imshow(patch_best, cmap='gray')
    axes[1].set_title(f"GT патч (sim={sim_at_gt:.3f})")
    axes[2].imshow(patch_random, cmap='gray')
    axes[2].set_title(f"Случайный (sim={sim_at_rand:.3f})")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "01_centered_similarity.png", dpi=150)
    plt.close()

    # 2. Проверка correlate_valid
    print("\n--- Проверка 2: correlate_valid ---")
    A_mat = np.array([[0,0,0,0,0],[0,1,2,3,0],[0,4,5,6,0],[0,7,8,9,0],[0,0,0,0,0]], dtype=np.float64)
    K_mat = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float64)
    corr_res = student_ops.correlate_valid(A_mat, K_mat)
    # Сравнение с scipy / torch
    t_A = torch.tensor(A_mat)[None, None]
    t_K = torch.tensor(K_mat)[None, None]
    torch_corr = F.conv2d(t_A, t_K).numpy()[0, 0]
    diff = np.max(np.abs(corr_res - torch_corr))
    print(f"Форма выхода: {corr_res.shape}, Max diff с PyTorch conv2d (correlation): {diff:.2e}")
    assert diff < 1e-10
    
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1); plt.imshow(A_mat, cmap='viridis'); plt.title("Матрица A"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(K_mat, cmap='viridis'); plt.title("Ядро K"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(corr_res, cmap='coolwarm'); plt.title("Valid Corr"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "02_correlate_valid.png", dpi=150)
    plt.close()

    # 3. Проверка linear_convolution_fft
    print("\n--- Проверка 3: linear_convolution_fft ---")
    cam = cv2.imread("images/source/camera.png", 0).astype(np.float64) / 255.0
    sobel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    fft_conv = student_ops.linear_convolution_fft(cam, sobel_v)
    import scipy.signal
    scipy_conv = scipy.signal.convolve2d(cam, sobel_v, mode='full')
    diff_fft = np.max(np.abs(fft_conv - scipy_conv))
    print(f"Линейная свертка через FFT: форма {fft_conv.shape}, Max diff со scipy.signal: {diff_fft:.2e}")
    assert diff_fft < 1e-10
    
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 3, 1); plt.imshow(cam, cmap='gray'); plt.title("Оригинал"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(fft_conv, cmap='gray'); plt.title("FFT свёртка (full)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(np.abs(fft_conv - scipy_conv), cmap='hot'); plt.title("Diff с эталоном"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "03_linear_convolution_fft.png", dpi=150)
    plt.close()

    # 4. Проверка gabor_kernel
    print("\n--- Проверка 4: gabor_kernel ---")
    gk_0 = student_ops.gabor_kernel(ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5)
    gk_45 = student_ops.gabor_kernel(ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5)
    gk_90 = student_ops.gabor_kernel(ksize=31, sigma=4.0, theta=np.pi/2, lambd=10.0, gamma=0.5)
    print(f"Сгенерированы ядра Габора: min={gk_0.min():.3f}, max={gk_0.max():.3f}, shape={gk_0.shape}")
    
    # Фильтрация кирпичной стены ядрами Габора
    brick = cv2.imread("images/source/brick.png", 0).astype(np.float64) / 255.0
    resp_0 = cv2.filter2D(brick, cv2.CV_64F, gk_0)
    resp_90 = cv2.filter2D(brick, cv2.CV_64F, gk_90)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes[0, 0].imshow(gk_0, cmap='gray'); axes[0, 0].set_title("Ядро Габора θ=0°")
    axes[0, 1].imshow(gk_45, cmap='gray'); axes[0, 1].set_title("Ядро Габора θ=45°")
    axes[0, 2].imshow(gk_90, cmap='gray'); axes[0, 2].set_title("Ядро Габора θ=90°")
    axes[1, 0].imshow(brick, cmap='gray'); axes[1, 0].set_title("Текстура кирпича")
    axes[1, 1].imshow(np.abs(resp_0), cmap='magma'); axes[1, 1].set_title("Отклик θ=0° (гориз.)")
    axes[1, 2].imshow(np.abs(resp_90), cmap='magma'); axes[1, 2].set_title("Отклик θ=90° (вертик.)")
    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "04_gabor_filter.png", dpi=150)
    plt.close()

    # 5. Проверка binary_local
    print("\n--- Проверка 5: binary_local ---")
    page = cv2.imread("images/source/page.png", 0)
    if page is None:
        page = (cam * 255).astype(np.uint8)
    bin_img = student_ops.binary_local(page, kernel_size=25, c=10.0)
    print(f"Локальная бинаризация: форма {bin_img.shape}, уникальные значения: {np.unique(bin_img)}")
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(page, cmap='gray'); plt.title("Исходное с градиентом освещения"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(bin_img, cmap='gray'); plt.title("Локальная бинаризация"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "05_binary_local.png", dpi=150)
    plt.close()

    # 6. Проверка grey_local
    print("\n--- Проверка 6: grey_local ---")
    local_m = student_ops.grey_local(cam, kernel_size=15, method="mean")
    local_s = student_ops.grey_local(cam, kernel_size=15, method="std")
    print(f"Локальные статистики (mean/std): mean in [{local_m.min():.3f}, {local_m.max():.3f}], std in [{local_s.min():.3f}, {local_s.max():.3f}]")
    
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(cam, cmap='gray'); plt.title("Camera"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(local_m, cmap='gray'); plt.title("Локальное среднее"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(local_s, cmap='inferno'); plt.title("Локальный контраст (std)"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "06_grey_local.png", dpi=150)
    plt.close()

    # 7. Проверка pool2d
    print("\n--- Проверка 7: pool2d ---")
    small_cam = cv2.resize(cam, (64, 64))
    max_p = student_ops.pool2d(small_cam, kernel_size=2, stride=2, mode="max")
    avg_p = student_ops.pool2d(small_cam, kernel_size=2, stride=2, mode="avg")
    # Проверка с PyTorch
    t_sm = torch.tensor(small_cam)[None, None]
    t_max = F.max_pool2d(t_sm, 2, 2).numpy()[0, 0]
    t_avg = F.avg_pool2d(t_sm, 2, 2).numpy()[0, 0]
    diff_max = np.max(np.abs(max_p - t_max))
    diff_avg = np.max(np.abs(avg_p - t_avg))
    print(f"Max diff с torch MaxPool2d: {diff_max:.2e}, AvgPool2d: {diff_avg:.2e}")
    assert diff_max < 1e-10 and diff_avg < 1e-10
    
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(small_cam, cmap='gray'); plt.title("Вход (64x64)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(max_p, cmap='gray'); plt.title("Max Pool 2x2 (32x32)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(avg_p, cmap='gray'); plt.title("Avg Pool 2x2 (32x32)"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "07_pool2d.png", dpi=150)
    plt.close()

    # 8. Проверка conv_output_size
    print("\n--- Проверка 8: conv_output_size ---")
    # k=3, p=2, s=2, d=2 для входа 64x64
    out_sz = student_ops.conv_output_size(input_size=64, kernel_size=3, padding=2, stride=2, dilation=2)
    print(f"conv_output_size(Hin=64, k=3, p=2, s=2, d=2) = {out_sz} (ожидается (32, 32))")
    assert out_sz == (32, 32)
    # Проверка с torch Conv2d
    c_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=2, stride=2, dilation=2)
    dummy_x = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        c_layer_out = c_layer(dummy_x)
    print(f"PyTorch фактический размер: {tuple(c_layer_out.shape[2:])}")
    assert tuple(c_layer_out.shape[2:]) == out_sz

    # 9. Проверка softmax_ce
    print("\n--- Проверка 9: softmax_ce ---")
    np.random.seed(42)
    logits = np.array([[2.0, 1.0, 0.1], [-1.0, 3.0, 0.5]], dtype=np.float64)
    targets = np.array([0, 1])
    loss, probs, grad = student_ops.softmax_ce(logits, targets)
    
    # Сравнение с PyTorch CrossEntropyLoss
    t_logits = torch.tensor(logits, requires_grad=True)
    t_targets = torch.tensor(targets, dtype=torch.long)
    t_loss = F.cross_entropy(t_logits, t_targets)
    t_loss.backward()
    
    diff_loss = abs(loss - t_loss.item())
    diff_grad = np.max(np.abs(grad - t_logits.grad.numpy()))
    print(f"Softmax CE Loss: {loss:.6f} vs PyTorch: {t_loss.item():.6f} (diff={diff_loss:.2e})")
    print(f"Max diff градиента dL/d(logits): {diff_grad:.2e}")
    assert diff_loss < 1e-7 and diff_grad < 1e-7
    
    plt.figure(figsize=(6, 3))
    plt.bar(["Класс 0", "Класс 1", "Класс 2"], probs[0], color=['dodgerblue', 'orange', 'green'])
    plt.title(f"Softmax распределение (target=0, loss={loss:.3f})")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "09_softmax_ce.png", dpi=150)
    plt.close()

    # 10. Проверка correlate_backward
    print("\n--- Проверка 10: correlate_backward ---")
    img_test = np.random.randn(5, 5).astype(np.float64)
    kern_test = np.random.randn(3, 3).astype(np.float64)
    
    # Прямой проход
    out_valid = student_ops.correlate_valid(img_test, kern_test)
    d_out = np.random.randn(*out_valid.shape).astype(np.float64)
    
    # Аналитический обратный проход
    d_img, d_kern = student_ops.correlate_backward(d_out, img_test, kern_test)
    
    # Численный градиент (Finite differences)
    eps = 1e-6
    d_kern_num = np.zeros_like(kern_test)
    for i in range(kern_test.shape[0]):
        for j in range(kern_test.shape[1]):
            k_plus = kern_test.copy()
            k_plus[i, j] += eps
            k_minus = kern_test.copy()
            k_minus[i, j] -= eps
            o_plus = student_ops.correlate_valid(img_test, k_plus)
            o_minus = student_ops.correlate_valid(img_test, k_minus)
            loss_plus = np.sum(d_out * o_plus)
            loss_minus = np.sum(d_out * o_minus)
            d_kern_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
            
    d_img_num = np.zeros_like(img_test)
    for i in range(img_test.shape[0]):
        for j in range(img_test.shape[1]):
            im_plus = img_test.copy()
            im_plus[i, j] += eps
            im_minus = img_test.copy()
            im_minus[i, j] -= eps
            o_plus = student_ops.correlate_valid(im_plus, kern_test)
            o_minus = student_ops.correlate_valid(im_minus, kern_test)
            loss_plus = np.sum(d_out * o_plus)
            loss_minus = np.sum(d_out * o_minus)
            d_img_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
            
    diff_k = np.max(np.abs(d_kern - d_kern_num))
    diff_im = np.max(np.abs(d_img - d_img_num))
    print(f"Max diff градиента по ядру (численный vs аналитический): {diff_k:.2e}")
    print(f"Max diff градиента по изображению (численный vs аналитический): {diff_im:.2e}")
    assert diff_k < 1e-6 and diff_im < 1e-6
    
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(d_out, cmap='coolwarm'); plt.title("d_out"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(d_img, cmap='coolwarm'); plt.title("dL/d(image)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(d_kern, cmap='coolwarm'); plt.title("dL/d(kernel)"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "10_correlate_backward.png", dpi=150)
    plt.close()

    print("\n[УСПЕХ] Все 10 математических операций в student_ops.py прошли строгое тестирование!")
    print(f"Все артефакты визуализации сохранены в {out_dir.resolve()}")

if __name__ == "__main__":
    run_tests_and_visualizations()
