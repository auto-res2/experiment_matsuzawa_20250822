import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from train import (
    ensure_dir, set_seed, get_device,
    QToyCNN, Int8RowQuant, GateMLP, SketchBank,
    BiasNormAdapter, QuantScaleCalibrator,
    build_synthetic_dataset, build_synthetic_kws_dataset,
    eval_accuracy, confusion_matrix,
    plot_curves, plot_confusion,
    learn_P, fit_S_group,
    collect_bias_bn_params_toy, collect_quant_params_toy,
    compute_gate_feats, parity_guard
)


def load_model(path: str, in_ch: int, num_classes: int, device: torch.device) -> QToyCNN:
    model = QToyCNN(in_ch=in_ch, num_classes=num_classes).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    return model


def save_sketch_package(package_path: str, P: torch.Tensor, S_groups: dict):
    out = {'P': P.cpu()}
    # Serialize Int8RowQuant as tensors
    ser = {}
    for g, lst in S_groups.items():
        ser[g] = []
        for Sk in lst:
            ser[g].append({'Q': Sk.Q.cpu(), 'scale': Sk.scale.cpu()})
    out['S_groups'] = ser
    torch.save(out, package_path)


def load_sketch_package(package_path: str):
    pkg = torch.load(package_path, map_location='cpu')
    P = pkg['P'].contiguous()
    S_groups = {}
    for g, lst in pkg['S_groups'].items():
        S_groups[g] = []
        for item in lst:
            S_groups[g].append(Int8RowQuant.from_q_and_scale(item['Q'], item['scale']))
    return P, S_groups


# ==========================
# Experiment 1: Vision domain-shift personalization (toy)
# ==========================

def experiment_vision(model_path: str, package_path: str, image_dir: str, device: torch.device, quick: bool = True):
    print('[Exp-1] Vision: starting...')
    ensure_dir(image_dir)
    C = 10
    model = load_model(model_path, in_ch=3, num_classes=C, device=device)

    # Data
    test_clean = build_synthetic_dataset(N=100, C=C, patterns=['clean'])
    test_shift = build_synthetic_dataset(N=100, C=C, patterns=['bright', 'noise', 'contrast', 'patch'])

    # Baseline
    base_fp_acc = eval_accuracy(model, test_clean, mode='fp', device=device)
    base_q_acc = eval_accuracy(model, test_clean, mode='quant', device=device)
    print(f'[Exp-1] Baseline clean: FP={base_fp_acc:.3f}, 8-bit={base_q_acc:.3f}')

    # Load SketchBank package
    P, S_groups = load_sketch_package(package_path)
    gate = GateMLP(in_dim=4, K=1).to(device)
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
    sketchbank = SketchBank(P_fp32=P, S_groups=S_groups, gate=gate)

    # Adapters
    bias_bn_params = collect_bias_bn_params_toy(model)
    quant_params = collect_quant_params_toy(model)
    adapters = {
        'bias_bn': BiasNormAdapter(bias_bn_params, lr=0.3, mom=0.9, clip=0.05),
        'quant': QuantScaleCalibrator(quant_params, lr=0.05, mom=0.9, clip=0.02)
    }

    # Online adaptation stream
    stream = build_synthetic_dataset(N=180 if quick else 400, C=C, patterns=['bright', 'noise', 'contrast', 'patch', 'clean'])
    accs, losses, gaps = [], [], []
    t0 = time.perf_counter()
    for step, (xb, y) in enumerate(stream):
        xb = xb.to(device)
        y_t = torch.tensor([y], device=device)
        logits_q = model.forward_quant(xb)
        loss = F.cross_entropy(logits_q, y_t)
        p = torch.softmax(logits_q.flatten(), dim=-1)
        e = p - F.one_hot(y_t[0], C).float()
        e_prime = sketchbank.project_error(e)
        feats = compute_gate_feats(model, xb, logits_q, e_prime)
        ok, gap = parity_guard(model, xb, bits_list=[8, 6, 4], tau=0.02)
        if not ok:
            adapters['quant'].lr *= 0.5
        sketchbank.apply_update(adapters, e_prime, feats)
        with torch.no_grad():
            logits_q2 = model.forward_quant(xb)
            acc = int(logits_q2.argmax(dim=-1).item() == y)
        accs.append(acc)
        losses.append(float(loss.item()))
        gaps.append(gap)
        if (step + 1) % 30 == 0:
            print(f'[Exp-1] Step {step+1:03d}: acc={np.mean(accs[-30:]):.3f}, loss={np.mean(losses[-30:]):.3f}, gap={np.mean(gaps[-30:]):.4f}')
    t1 = time.perf_counter()
    print(f'[Exp-1] Online adaptation time: {(t1 - t0)*1000:.1f} ms over {len(stream)} steps')

    final_q_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-1] Final shifted 8-bit accuracy: {final_q_acc:.3f}')

    # Plots
    xs = list(range(1, len(accs) + 1))
    plot_curves(xs, {'sketchbank': accs}, 'Online Accuracy (toy vision)', 'accuracy', os.path.join(image_dir, 'exp1_accuracy_sketchbank.pdf'))
    plot_curves(xs, {'sketchbank': losses}, 'Online Loss (toy vision)', 'loss', os.path.join(image_dir, 'exp1_training_loss_sketchbank.pdf'))
    plot_curves(xs, {'parity_gap': gaps}, 'FP–int8 parity gap (toy vision)', 'gap', os.path.join(image_dir, 'exp1_parity_gap_sketchbank.pdf'))
    cm = confusion_matrix(model, test_shift, C=C, mode='quant', device=device)
    plot_confusion(cm, classes=[str(i) for i in range(C)], title='Confusion (toy vision, shifted)', filename=os.path.join(image_dir, 'exp1_confusion_matrix_sketchbank.pdf'))

    # Save adapted model
    adapted_path = os.path.join(os.path.dirname(model_path), 'qtcnn_vision_adapted.pt')
    torch.save({'state_dict': model.state_dict()}, adapted_path)
    print(f'[Exp-1] Saved figures to {image_dir} and adapted model to {adapted_path}')


# ==========================
# Experiment 2: Quantization-control stress test & parity safeguards (toy)
# ==========================

def experiment_quant_only(model_path: str, image_dir: str, device: torch.device, quick: bool = True):
    print('[Exp-2] Quant-control stress test: starting...')
    ensure_dir(image_dir)
    C = 10
    model = load_model(model_path, in_ch=3, num_classes=C, device=device)

    train_data = build_synthetic_dataset(N=400, C=C, patterns=['clean'])
    test_shift = build_synthetic_dataset(N=100, C=C, patterns=['bright', 'noise', 'contrast'])

    base_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-2] Baseline shifted 8-bit accuracy: {base_acc:.3f}')

    # Perturb quant params
    with torch.no_grad():
        for mod in [model.c1, model.c2, model.c3]:
            mod.s_mult.mul_(1.0 + 0.2 * (2 * torch.rand_like(mod.s_mult) - 1))
            mod.act.act_clip.mul_(0.7)
        model.head.s_mult.mul_(1.2)

    perturbed_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-2] After induced miscalibration, acc: {perturbed_acc:.3f}')

    # Learn P and S for quant group on-the-fly (proxy)
    proxy_data = train_data[:200]
    P = learn_P(model, proxy_data, C=C, r_e=8, device=device)
    quant_params = collect_quant_params_toy(model)
    S_quant = fit_S_group(model, proxy_data, P, quant_params, lam=1e-2, mode='quant', device=device)
    gate = GateMLP(in_dim=4, K=1).to(device)
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
    sketchbank = SketchBank(P_fp32=P, S_groups={'quant': [S_quant]}, gate=gate)
    adapters = {'quant': QuantScaleCalibrator(quant_params, lr=0.05, mom=0.9, clip=0.02)}

    stream = build_synthetic_dataset(N=160 if quick else 400, C=C, patterns=['bright', 'noise', 'contrast'])
    accs, gaps = [], []
    t0 = time.perf_counter()
    for step, (xb, y) in enumerate(stream):
        xb = xb.to(device)
        logits_q = model.forward_quant(xb)
        p = torch.softmax(logits_q.flatten(), dim=-1)
        e = p * (1.0 + torch.log(torch.clamp(p, 1e-6, 1.0)))  # entropy-grad surrogate
        e_prime = sketchbank.project_error(e)
        feats = torch.zeros(4, device=device)
        ok, gap = parity_guard(model, xb, bits_list=[8, 6, 4], tau=0.01)
        if not ok:
            adapters['quant'].lr *= 0.5
        sketchbank.apply_update(adapters, e_prime, feats)
        with torch.no_grad():
            acc = int(model.forward_quant(xb).argmax(dim=-1).item() == y)
        accs.append(acc)
        gaps.append(gap)
        if (step + 1) % 40 == 0:
            print(f'[Exp-2] Step {step+1:03d}: acc={np.mean(accs[-40:]):.3f}, gap={np.mean(gaps[-40:]):.4f}')
    t1 = time.perf_counter()
    print(f'[Exp-2] Online quant-only time: {(t1 - t0)*1000:.1f} ms over {len(stream)} steps')

    final_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-2] Final accuracy after quant-only adaptation: {final_acc:.3f} (delta vs perturbed: {final_acc - perturbed_acc:+.3f})')

    xs = list(range(1, len(accs) + 1))
    plot_curves(xs, {'quant_only': accs}, 'Quant-only accuracy (toy vision)', 'accuracy', os.path.join(image_dir, 'exp2_accuracy_quant_only.pdf'))
    plot_curves(xs, {'parity_gap': gaps}, 'FP–int8 parity gap (quant-only)', 'gap', os.path.join(image_dir, 'exp2_parity_gap_quant_only.pdf'))
    print(f'[Exp-2] Saved figures to {image_dir}')


# ==========================
# Experiment 3: KWS-like (toy)
# ==========================

def experiment_kws(image_dir: str, device: torch.device, quick: bool = True):
    print('[Exp-3] KWS: starting...')
    ensure_dir(image_dir)
    C = 12

    # Pretrain a fresh KWS model quickly (self-contained for this toy exp)
    model = QToyCNN(in_ch=1, num_classes=C).to(device)
    train_data = build_synthetic_kws_dataset(N=400, C=C, patterns=['clean'])
    # quick pretraining
    from train import pretrain_toy_model
    pretrain_toy_model(model, train_data, steps=120 if quick else 300, lr=5e-3, device=device)

    test_shift = build_synthetic_kws_dataset(N=120, C=C, patterns=['noise_low_snr', 'reverb'])
    base_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-3] Baseline shifted 8-bit accuracy: {base_acc:.3f}')

    # Offline compute P and S
    proxy_data = train_data[:200]
    P = learn_P(model, proxy_data, C=C, r_e=8, device=device)
    bias_bn_params = collect_bias_bn_params_toy(model)
    quant_params = collect_quant_params_toy(model)
    S_bias = fit_S_group(model, proxy_data, P, bias_bn_params, lam=1e-2, mode='fp', device=device)
    S_quant = fit_S_group(model, proxy_data, P, quant_params, lam=1e-2, mode='quant', device=device)

    gate = GateMLP(in_dim=4, K=1).to(device)
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
    sketchbank = SketchBank(P_fp32=P, S_groups={'bias_bn': [S_bias], 'quant': [S_quant]}, gate=gate)
    adapters = {
        'bias_bn': BiasNormAdapter(bias_bn_params, lr=0.25, mom=0.9, clip=0.05),
        'quant': QuantScaleCalibrator(quant_params, lr=0.05, mom=0.9, clip=0.02)
    }

    stream = build_synthetic_kws_dataset(N=160 if quick else 400, C=C, patterns=['noise_low_snr', 'reverb', 'clean'])
    accs, gaps = [], []
    for step, (xb, y) in enumerate(stream):
        xb = xb.to(device)
        logits_q = model.forward_quant(xb)
        p = torch.softmax(logits_q.flatten(), dim=-1)
        e = p * (1.0 + torch.log(torch.clamp(p, 1e-6, 1.0)))
        e_prime = sketchbank.project_error(e)
        feats = torch.zeros(4, device=device)
        ok, gap = parity_guard(model, xb, bits_list=[8, 6, 4], tau=0.02)
        if not ok:
            adapters['quant'].lr *= 0.5
        sketchbank.apply_update(adapters, e_prime, feats)
        with torch.no_grad():
            acc = int(model.forward_quant(xb).argmax(dim=-1).item() == y)
        accs.append(acc)
        gaps.append(gap)
        if (step + 1) % 40 == 0:
            print(f'[Exp-3] Step {step+1:03d}: acc={np.mean(accs[-40:]):.3f}, gap={np.mean(gaps[-40:]):.4f}')

    final_acc = eval_accuracy(model, test_shift, mode='quant', device=device)
    print(f'[Exp-3] Final shifted 8-bit accuracy: {final_acc:.3f}')

    xs = list(range(1, len(accs) + 1))
    plot_curves(xs, {'kws_sketchbank': accs}, 'KWS accuracy (toy)', 'accuracy', os.path.join(image_dir, 'exp3_accuracy_sketchbank.pdf'))
    plot_curves(xs, {'parity_gap': gaps}, 'FP–int8 parity gap (toy KWS)', 'gap', os.path.join(image_dir, 'exp3_parity_gap_sketchbank.pdf'))
    print(f'[Exp-3] Saved figures to {image_dir}')


# ==========================
# Orchestrator
# ==========================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SketchBank - Evaluation & adaptation (toy)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get('device'))
    set_seed(int(cfg.get('seed', 1234)))

    images_root = cfg.get('output_image_dir', '.research/iteration1/images')
    ensure_dir(images_root)

    models_dir = cfg.get('models_dir', 'models')

    quick = bool(cfg.get('quick', True))

    # Experiment 1 - Vision
    if cfg.get('experiments', {}).get('enable_vision', True):
        model_path = os.path.join(models_dir, 'qtcnn_vision.pt')
        package_path = os.path.join(models_dir, 'sketchbank_vision.pt')
        if not (os.path.exists(model_path) and os.path.exists(package_path)):
            print('[Exp-1] Missing model or sketch package. Please run preprocess first or main.py orchestrator.')
        else:
            experiment_vision(model_path, package_path, os.path.join(images_root, 'exp1'), device, quick=quick)

    # Experiment 2 - Quant-only stress test
    if cfg.get('experiments', {}).get('enable_quant_only', True):
        model_path = os.path.join(models_dir, 'qtcnn_vision.pt')
        if not os.path.exists(model_path):
            print('[Exp-2] Missing model. Please run train or main.py orchestrator.')
        else:
            experiment_quant_only(model_path, os.path.join(images_root, 'exp2'), device, quick=quick)

    # Experiment 3 - KWS
    if cfg.get('experiments', {}).get('enable_kws', True):
        experiment_kws(os.path.join(images_root, 'exp3'), device, quick=quick)
