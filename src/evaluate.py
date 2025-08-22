import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from typing import Dict, List, Tuple, Any
import ast
import sys
import uuid
from dataclasses import dataclass

from train import MLPClassifier, TemperatureScaler, TorchProbaWrapper, PartitionedQuantileCI
from preprocess import set_seed, ast_feature_counts


@dataclass
class TestCase:
    args: Dict[str, Any]
    expected: Any = None
    kind: str = "gold"


@dataclass
class TestOutcome:
    covered_lines: List[int]
    failed: bool
    exception_str: str = None
    mismatch: bool = False
    kind: str = "gold"


def make_module_and_get_func(code: str) -> Tuple[callable, str, str]:
    """Create module from code and extract function."""
    tree = ast.parse(code)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert fn_names, "No function defined in code"
    func_name = fn_names[0]
    filename = f"{func_name}_{uuid.uuid4().hex[:8]}.py"
    code_obj = compile(code, filename, 'exec')
    ns: Dict[str, Any] = {}
    exec(code_obj, ns, ns)
    return ns[func_name], func_name, filename


def run_one_test_with_trace(func: callable, filename: str, test: TestCase) -> TestOutcome:
    """Run test with line coverage tracing."""
    covered: List[int] = []
    
    def tracer(frame, event, arg):
        if frame.f_code.co_filename == filename and event == 'line':
            covered.append(frame.f_lineno)
        return tracer
    
    sys.settrace(tracer)
    exc = None
    mismatch = False
    
    try:
        out = func(**test.args)
        if test.expected is not None:
            mismatch = not safe_equals(out, test.expected)
    except Exception as e:
        exc = e
    finally:
        sys.settrace(None)
    
    failed = (exc is not None) or mismatch
    exc_str = repr(exc) if exc is not None else None
    
    return TestOutcome(
        covered_lines=sorted(set(covered)), 
        failed=failed, 
        exception_str=exc_str, 
        mismatch=mismatch, 
        kind=test.kind
    )


def safe_equals(a: Any, b: Any) -> bool:
    """Safe equality check."""
    try:
        return a == b
    except Exception:
        return False


def compute_ochiai(coverage_matrix: List[List[int]], outcomes: List[bool], all_lines: List[int]) -> Dict[int, float]:
    """Compute Ochiai SBFL scores."""
    total_fails = sum(1 for o in outcomes if o)
    if total_fails == 0:
        return {ln: 0.0 for ln in all_lines}
    
    stats: Dict[int, Tuple[int, int]] = {ln: [0, 0] for ln in all_lines}  # fail_i, pass_i
    
    for hits, fail in zip(coverage_matrix, outcomes):
        for ln in all_lines:
            if ln in hits:
                stats[ln][0 if fail else 1] += 1
    
    ochiai = {}
    for ln in all_lines:
        fail_i, pass_i = stats[ln]
        total_i = fail_i + pass_i
        if total_i == 0 or total_fails == 0:
            ochiai[ln] = 0.0
        else:
            denom = np.sqrt(total_fails * total_i)
            ochiai[ln] = fail_i / denom if denom > 0 else 0.0
    
    return ochiai


def generate_test_cases_for_function(func_name: str) -> List[TestCase]:
    """Generate test cases for common functions."""
    if func_name == "factorial":
        return [
            TestCase({"n": 0}, 1),
            TestCase({"n": 1}, 1),
            TestCase({"n": 5}, 120),
            TestCase({"n": 3}, 6),
        ]
    elif func_name == "fibonacci":
        return [
            TestCase({"n": 0}, 0),
            TestCase({"n": 1}, 1),
            TestCase({"n": 5}, 5),
            TestCase({"n": 7}, 13),
        ]
    elif func_name == "sum_range":
        return [
            TestCase({"start": 1, "end": 5}, 10),
            TestCase({"start": 0, "end": 3}, 3),
            TestCase({"start": 5, "end": 5}, 0),
        ]
    elif func_name == "find_max":
        return [
            TestCase({"arr": [1, 3, 2]}, 3),
            TestCase({"arr": [5]}, 5),
            TestCase({"arr": []}, None),
            TestCase({"arr": [-1, -5, -2]}, -1),
        ]
    elif func_name == "binary_search":
        return [
            TestCase({"arr": [1, 2, 3, 4, 5], "target": 3}, 2),
            TestCase({"arr": [1, 2, 3, 4, 5], "target": 6}, -1),
            TestCase({"arr": [], "target": 1}, -1),
        ]
    else:
        return [TestCase({})]


def evaluate_single_function(code: str, model_wrapper: TorchProbaWrapper, 
                            conformal: PartitionedQuantileCI) -> Dict[str, Any]:
    """Evaluate a single function with AST-SelfCheck methodology."""
    try:
        func, func_name, filename = make_module_and_get_func(code)
    except Exception as e:
        return {"error": f"Failed to parse function: {e}"}
    
    try:
        ast_features = ast_feature_counts(code)
        feature_vector = np.array([[
            ast_features["num_Add"], ast_features["num_Sub"], 
            ast_features["num_Mult"], ast_features["num_FloorDiv"],
            ast_features["num_Call"], ast_features["num_Compare"],
            ast_features["num_For"], ast_features["num_If"],
            ast_features["num_Return"], len(code.splitlines())
        ]])
    except Exception as e:
        return {"error": f"Failed to extract features: {e}"}
    
    try:
        probs = model_wrapper.predict_proba(feature_vector)
        p_error = probs[0, 1]  # Probability of error
        
        ci_low, ci_high = conformal.interval(p_error)
    except Exception as e:
        return {"error": f"Model prediction failed: {e}"}
    
    test_cases = generate_test_cases_for_function(func_name)
    outcomes = []
    coverage_matrix = []
    
    for test in test_cases:
        try:
            outcome = run_one_test_with_trace(func, filename, test)
            outcomes.append(outcome.failed)
            coverage_matrix.append(outcome.covered_lines)
        except Exception:
            outcomes.append(True)  # Assume failure if can't run
            coverage_matrix.append([])
    
    all_lines = list(range(1, len(code.splitlines()) + 1))
    ochiai_scores = compute_ochiai(coverage_matrix, outcomes, all_lines)
    
    line_error_probs = {}
    for line_num in all_lines:
        sbfl_score = ochiai_scores.get(line_num, 0.0)
        line_prob = 0.7 * p_error + 0.3 * sbfl_score
        line_error_probs[line_num] = min(1.0, line_prob)
    
    return {
        "function_name": func_name,
        "p_error_global": float(p_error),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "line_error_probs": line_error_probs,
        "ochiai_scores": ochiai_scores,
        "test_failures": sum(outcomes),
        "total_tests": len(outcomes),
        "ast_features": ast_features
    }


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Evaluate calibration metrics."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)
    
    brier = np.mean((y_prob - y_true) ** 2)
    
    return {"ECE": ece, "MCE": mce, "Brier": brier}


def create_evaluation_plots(results: Dict[str, Any], save_dir: str):
    """Create evaluation plots and save as PDFs."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    if 'fpr' in results and 'tpr' in results:
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], linewidth=2, 
                label=f'ROC (AUC = {results["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - AST-SelfCheck Error Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'precision' in results and 'recall' in results:
        plt.figure(figsize=(8, 6))
        plt.plot(results['recall'], results['precision'], linewidth=2,
                label=f'PR (AP = {results["average_precision"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - AST-SelfCheck Error Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'calibration' in results:
        cal = results['calibration']
        plt.figure(figsize=(8, 6))
        
        metrics_text = f"ECE: {cal['ECE']:.3f}\nMCE: {cal['MCE']:.3f}\nBrier: {cal['Brier']:.3f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot - AST-SelfCheck')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_plot.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'confusion_matrix' in results:
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Correct', 'Buggy'], 
                   yticklabels=['Correct', 'Buggy'])
        plt.title('Confusion Matrix - AST-SelfCheck')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'feature_importance' in results:
        plt.figure(figsize=(10, 6))
        features = ['Add', 'Sub', 'Mult', 'FloorDiv', 'Call', 'Compare', 'For', 'If', 'Return', 'Lines']
        importance = results['feature_importance']
        
        plt.barh(features, importance)
        plt.xlabel('Feature Importance')
        plt.title('AST Feature Importance for Error Detection')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Evaluation plots saved to {save_dir}/")


def evaluate_models():
    """Main evaluation function."""
    print("Starting model evaluation...")
    set_seed(42)
    
    models_dir = "models"
    data_dir = "data"
    
    with open(os.path.join(models_dir, "training_info.pkl"), "rb") as f:
        training_info = pickle.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(training_info["feature_dim"])
    model.load_state_dict(torch.load(os.path.join(models_dir, "mlp_model.pth"), map_location=device))
    
    scaler = TemperatureScaler()
    scaler.load_state_dict(torch.load(os.path.join(models_dir, "temperature_scaler.pth"), map_location=device))
    
    wrapper = TorchProbaWrapper(model, scaler, device)
    
    with open(os.path.join(models_dir, "conformal_predictor.pkl"), "rb") as f:
        conformal = pickle.load(f)
    
    features = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    codes = np.load(os.path.join(data_dir, "codes.npy"), allow_pickle=True)
    
    with open(os.path.join(data_dir, "fault_lines.pkl"), "rb") as f:
        fault_lines = pickle.load(f)
    
    test_size = int(0.2 * len(features))
    test_features = features[-test_size:]
    test_labels = labels[-test_size:]
    test_codes = codes[-test_size:]
    
    print(f"Evaluating on {len(test_features)} test samples")
    
    test_probs = wrapper.predict_proba(test_features)
    test_preds = np.argmax(test_probs, axis=1)
    
    auc = roc_auc_score(test_labels, test_probs[:, 1])
    ap = average_precision_score(test_labels, test_probs[:, 1])
    
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    precision, recall, _ = precision_recall_curve(test_labels, test_probs[:, 1])
    
    cm = confusion_matrix(test_labels, test_preds)
    calibration = evaluate_calibration(test_labels, test_probs[:, 1])
    
    print(f"\nEvaluation Results:")
    print(f"AUC: {auc:.3f}")
    print(f"Average Precision: {ap:.3f}")
    print(f"Accuracy: {np.mean(test_preds == test_labels):.3f}")
    print(f"ECE: {calibration['ECE']:.3f}")
    print(f"MCE: {calibration['MCE']:.3f}")
    print(f"Brier Score: {calibration['Brier']:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds))
    
    print("\nEvaluating individual functions...")
    function_results = []
    for i, code in enumerate(test_codes[:5]):  # Evaluate first 5 for demo
        result = evaluate_single_function(code, wrapper, conformal)
        if "error" not in result:
            function_results.append(result)
            print(f"Function {result['function_name']}: P(error)={result['p_error_global']:.3f}, "
                  f"CI={result['confidence_interval']}, Tests failed: {result['test_failures']}/{result['total_tests']}")
    
    results = {
        "auc": auc,
        "average_precision": ap,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "calibration": calibration,
        "function_results": function_results
    }
    
    save_dir = ".research/iteration1/images"
    create_evaluation_plots(results, save_dir)
    
    results_file = os.path.join(save_dir, "evaluation_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Evaluation completed! Results saved to {save_dir}/")
    
    return results


if __name__ == "__main__":
    evaluate_models()
