import copy
import numpy as np
import wandb
from utils import _softmax, _precision, _recall, _f1_score


def _calculate_exact_match_3(
    val_labels, val_preds, inverse_val_labels, inverse_val_preds, INDEX_TO_LABEL
):
    """
    Option 1: Đánh giá có hướng, tuy nhiên các nhãn vô hướng chỉ tính một sample.

    Ví dụ: Golden: NewAdverse(E1, E2)
    Nếu Predict: Anta(E1, E2) và NewAdverse(E2, E1)
    => Evaluate cho NewAdverse: 1TP
    => Evaluate cho Anta: 1FP
    Predict: NewAdverse(E1, E2) HOẶC NewAdverse(E2, E1), HOẶC cả hai NewAdverse(E1, E2) và NewAdverse(E2, E1)
    => Evaluate đều chỉ tính 1TP cho NewAdverse
    """
    sample_result = {
        label: {"TP": 0, "FP": 0, "FN": 0} for label in INDEX_TO_LABEL.values()
    }

    if val_labels != val_preds:
        sample_result[INDEX_TO_LABEL[val_labels]]["FN"] += 1
        sample_result[INDEX_TO_LABEL[val_preds]]["FP"] += 1

    if inverse_val_labels != inverse_val_preds:
        sample_result[INDEX_TO_LABEL[inverse_val_labels]]["FN"] += 1
        sample_result[INDEX_TO_LABEL[inverse_val_preds]]["FP"] += 1

    # If New Adverse/New Effect, only count 1 TP if either side is correct
    if val_labels == inverse_val_labels and (INDEX_TO_LABEL[val_labels] == "New Effect" or INDEX_TO_LABEL[val_labels] == "New Adverse"):
        if val_labels == val_preds:
            sample_result[INDEX_TO_LABEL[val_labels]]["TP"] += 1
        elif inverse_val_labels == inverse_val_preds:
            sample_result[INDEX_TO_LABEL[inverse_val_labels]]["TP"] += 1
    # Else, add TP for each correct prediction
    else:
        if val_labels == val_preds:
            sample_result[INDEX_TO_LABEL[val_labels]]["TP"] += 1
        if inverse_val_labels == inverse_val_preds:
            sample_result[INDEX_TO_LABEL[inverse_val_labels]]["TP"] += 1
    return sample_result


# Option 2: Match all 4 components (D1, D2, Interaction, Direction)
def _calculate_exact_match_4(
    val_labels, val_preds, inverse_val_labels, inverse_val_preds, INDEX_TO_LABEL
):
    """
    Option 2: Đánh giá có hướng, một sample đúng khi khớp cả 4 thành phần D1, D2, Interaction, Direction.

    Ví dụ: Exact match mức 4
    Golden: NewAdverse(E1, E2)
    (1) Nếu Predict: Anta(E1, E2) và NewAdverse(E2, E1)
    => Evaluate cho NewAdverse: 1TP & 1FN
    => Evaluate cho Anta: 1FP

    (2) Nếu Predict: NewAdverse(E1, E2) HOẶC NewAdverse(E2, E1), chiều ngược lại là NoInteraction
    => Evaluate đều chỉ tính 1TP cho NewAdverse

    (3) Nếu Predict: NewAdverse(E1, E2) và NewAdverse(E2, E1)
    => Evaluate tính 2TP cho NewAdverse
    """
    sample_result = {
        label: {"TP": 0, "FP": 0, "FN": 0} for label in INDEX_TO_LABEL.values()
    }
    if val_labels == val_preds:
        sample_result[INDEX_TO_LABEL[val_labels]]["TP"] += 1
    else:
        sample_result[INDEX_TO_LABEL[val_labels]]["FN"] += 1
        sample_result[INDEX_TO_LABEL[val_preds]]["FP"] += 1

    if inverse_val_labels == inverse_val_preds:
        sample_result[INDEX_TO_LABEL[inverse_val_labels]]["TP"] += 1
    else:
        sample_result[INDEX_TO_LABEL[inverse_val_labels]]["FN"] += 1
        sample_result[INDEX_TO_LABEL[inverse_val_preds]]["FP"] += 1
    return sample_result

# Option 3: Indirect evaluation
def _calculate_exact_match_5(
    val_labels,
    val_preds,
    inverse_val_labels,
    inverse_val_preds,
    val_outputs,
    inverse_val_outputs,
    INDEX_TO_LABEL,
):
    """
    Option 3: Đánh giá vô hướng, chỉ cần khớp 3 thành phần D1, D2, Interaction.

    Ví dụ: Exact match mức 3

    Golden: NewAdverse(E1, E2)
    (1) Nếu Predict: Anta(E1, E2) và NewAdverse(E2, E1)
    => Evaluate cho NewAdverse: 1TP
    => Evaluate cho Anta: 1FP
    (2) Nếu Predict: NewAdverse(E1, E2) HOẶC NewAdverse(E2, E1), chiều ngược lại là NoInteraction HOẶC NewAdverse(E1, E2) và NewAdverse(E2, E1)
    => Evaluate đều chỉ tính 1TP cho NewAdverse
    Với kiểu exact match mức 3 này, thì với trường hợp đoán sai hướng của anta, syn vẫn được tính là 1 TP hết nhé.

    """
    sample_result = {
        label: {"TP": 0, "FP": 0, "FN": 0} for label in INDEX_TO_LABEL.values()
    }

    # Find best label
    val_positive = val_labels > 0
    inverse_val_positive = inverse_val_labels > 0
    if (val_positive or inverse_val_positive) and (
        val_positive and inverse_val_positive
    ):
        # both positive: New adverse case, or never happen if rm new adverse
        label = val_labels
    elif (val_positive and not inverse_val_positive) or (
        not val_positive and inverse_val_positive
    ):
        # one positive
        label = val_labels if val_positive else inverse_val_labels
    else:
        # no positive
        label = val_labels

    # Find best pred
    if val_preds != inverse_val_labels:
        if val_preds and inverse_val_preds:
            # Case between positive preds
            softmax_val = _softmax(val_outputs)
            softmax_inverse = _softmax(inverse_val_outputs)
            stacked_arrays = np.stack([softmax_val, softmax_inverse], axis=0)
            # Get the max along the first axis (axis=0)
            pred = np.argmax(np.max(stacked_arrays, axis=0))
        elif (not bool(val_preds) or not bool(inverse_val_preds)) and (
            bool(val_preds) or bool(inverse_val_preds)
        ):
            pred = val_preds if val_preds else inverse_val_preds
        else:
            pred = val_preds
    else:
        pred = val_preds

    if pred == label:
        sample_result[INDEX_TO_LABEL[pred]]["TP"] += 1
    else:
        sample_result[INDEX_TO_LABEL[pred]]["FP"] += 1
        sample_result[INDEX_TO_LABEL[label]]["FN"] += 1
    return sample_result


def _post_processing_rule(
    val_all_preds, inverse_val_all_preds, val_all_outputs, inverse_val_all_outputs
):
    def _check_and_choose(pred_output_a, pred_output_b):
        if (
            np.argmax(pred_output_a) == np.argmax(pred_output_b)
            and np.argmax(pred_output_a) == 3
        ):
            return 3, 3
        elif np.argmax(pred_output_a) == 3:
            logit_a = pred_output_a[np.argmax(pred_output_a)]
            logit_b = pred_output_b[np.argmax(pred_output_b)]
            if logit_a > logit_b:
                return 3, 3
            else:
                return 0, np.argmax(pred_output_b)
        elif np.argmax(pred_output_b) == 3:
            logit_a = pred_output_a[np.argmax(pred_output_a)]
            logit_b = pred_output_b[np.argmax(pred_output_b)]
            if logit_b > logit_a:
                return 3, 3
            else:
                return np.argmax(pred_output_a), 0
        else:
            return np.argmax(pred_output_a), np.argmax(pred_output_b)

    val_preds = copy.deepcopy(val_all_preds)
    inverse_val_preds = copy.deepcopy(inverse_val_all_preds)
    for i in range(len(val_all_preds)):
        val_preds[i], inverse_val_preds[i] = _check_and_choose(
            val_all_outputs[i], inverse_val_all_outputs[i]
        )
    return val_preds, inverse_val_preds

def _calc_macro_f1(results):
    f1s = []
    for key, value in results.items():
        if key == "No Interaction":
            continue

        denominator = 2 * value["TP"] + value["FP"] + value["FN"]
        if denominator > 0:
            f1 = 2 * value["TP"] / denominator
            f1s.append(f1)
        else:
            f1s.append(0.0)
    return sum(f1s) * 1.0 / len(f1s)

def _calc_micro_f1(results):
    TP = 0
    FP = 0
    FN = 0
    for key, value in results.items():
        if key == "No Interaction":
            continue
        TP += value["TP"]
        FP += value["FP"]
        FN += value["FN"]
    
    return _f1_score(TP, FP, FN)

def _calc_micro_precision(results):
    TP = 0
    FP = 0
    for key, value in results.items():
        if key == "No Interaction":
            continue
        TP += value["TP"]
        FP += value["FP"]
    
    return _precision(TP, FP)

def _calc_micro_recall(results):
    TP = 0
    FN = 0
    for key, value in results.items():
        if key == "No Interaction":
            continue
        TP += value["TP"]
        FN += value["FN"]
    
    return _recall(TP, FN)

def _calc_f1_for_negative_class(results, class_name="No Interaction"):
    if (
        class_name not in results
        or (
            results[class_name]["TP"]
            + results[class_name]["FP"]
            + results[class_name]["FN"]
        )
        == 0
    ):
        return 0.0
    return (
        2
        * results[class_name]["TP"]
        / (
            2 * results[class_name]["TP"]
            + results[class_name]["FP"]
            + results[class_name]["FN"]
        )
    )


def get_evaluation_metrics(
    all_labels,
    all_preds,
    all_outputs,
    label_mapping,
    options=[1],
    is_test=False,
    epoch=None,
    prefix=None,
):
    if not isinstance(label_mapping, dict) or not label_mapping:
        raise ValueError("A valid `label_mapping` dictionary must be provided.")
    
    INDEX_TO_LABEL = {v: k for k, v in label_mapping.items()}

    length = len(all_labels)
    val_all_labels = all_labels[:length//2]
    val_all_preds = all_preds[:length//2]
    val_all_outputs = all_outputs[:length//2]

    inverse_val_all_labels = all_labels[length//2:]
    inverse_val_all_preds = all_preds[length//2:]
    inverse_val_all_outputs = all_outputs[length//2:]

    results = {}
    for option in options:
        results[option] = {
            label: {"TP": 0, "FP": 0, "FN": 0} for label in INDEX_TO_LABEL.values()
        }

        for i in range(len(val_all_labels)):
            if option == 1:
                sample_result = _calculate_exact_match_3(
                    val_all_labels[i],
                    val_all_preds[i],
                    inverse_val_all_labels[i],
                    inverse_val_all_preds[i],
                    INDEX_TO_LABEL,
                )
            
            elif option == 2:
                sample_result = _calculate_exact_match_4(
                    val_all_labels[i],
                    val_all_preds[i],
                    inverse_val_all_labels[i],
                    inverse_val_all_preds[i],
                    INDEX_TO_LABEL,
                )
            
            elif option == 3:
                sample_result = _calculate_exact_match_5(
                    val_all_labels[i],
                    val_all_preds[i],
                    inverse_val_all_labels[i],
                    inverse_val_all_preds[i],
                    val_all_outputs[i],
                    inverse_val_all_outputs[i],
                    INDEX_TO_LABEL,
                )

            for key, value in sample_result.items():
                results[option][key]["TP"] += value["TP"]
                results[option][key]["FP"] += value["FP"]
                results[option][key]["FN"] += value["FN"]

    # get support number
    support = {label: 0 for label in INDEX_TO_LABEL.values()}

    for i in range(len(val_all_labels)):
        support[INDEX_TO_LABEL[val_all_labels[i]]] += 1

    for option, value in results.items():
        for key, value in results[option].items():
            macro = _calc_macro_f1(results[option])
            micro = _calc_micro_f1(results[option])
            micro_precision = _calc_micro_precision(results[option])
            micro_recall = _calc_micro_recall(results[option])
            f1_negative_class = _calc_f1_for_negative_class(
                results[option], class_name="No Interaction"
            )

        results[option]["macro"] = macro
        results[option]["micro"] = micro
        results[option]["micro_precision"] = micro_precision
        results[option]["micro_recall"] = micro_recall
        results[option]["f1_no_interaction"] = f1_negative_class
    for option in results:
        print(f"Option {option}: {results[option]}")
    
    # logger - thu thập tất cả metrics vào một dict
    if wandb.run is not None:
        log_dict = {}
        for option, value in results.items():
            for key, value in results[option].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        # Sử dụng prefix nếu có, không thì dùng is_test
                        if prefix:
                            log_dict[f"{prefix}/{option}/{key}/{k}"] = v
                        elif is_test:
                            log_dict[f"{option}/{key}/{k}"] = v
                        else:
                            log_dict[f"val/{option}/{key}/{k}"] = v
                else:
                    if prefix:
                        log_dict[f"{prefix}/{option}/{key}"] = value
                    elif is_test:
                        log_dict[f"{option}/{key}"] = value
                    else:
                        log_dict[f"val/{option}/{key}"] = value
        
        # Log tất cả cùng lúc với epoch step
        if epoch is not None:
            wandb.log(log_dict, step=epoch)
        else:
            wandb.log(log_dict)

    return results
