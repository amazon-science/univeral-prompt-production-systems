import re
import json
import torch


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


def compare_models_layers(model_1, model_2):
    import pandas

    max_diff = None
    per_layer_max_diff = {}
    layer_idx_col = []
    layer_name_col = []
    layer_diff_col = []
    layer_diff_over_average = []
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        assert key_item_1[0] == key_item_2[0]

        match = re.match(r"encoder.layer.([0-9]+)\.(.*)", key_item_1[0])
        if match is not None:
            layer_name = match.group(2)

            _diff_mean = torch.abs(key_item_1[1] - key_item_2[1]).mean().item()
            if max_diff is None or _diff_mean > max_diff:
                max_diff = _diff_mean

            current_max = per_layer_max_diff.get(layer_name)
            if current_max is None or current_max < _diff_mean:
                per_layer_max_diff[layer_name] = _diff_mean

            layer_idx_col.append(int(match.group(1)))
            layer_name_col.append(layer_name)
            layer_diff_col.append(_diff_mean)
            layer_diff_over_average.append(
                _diff_mean / torch.abs(key_item_1[1]).mean().item()
            )

    layer_diff_over_layer_max = []
    layer_diff_over_max = []
    for idx, layer_name in enumerate(layer_name_col):
        max = per_layer_max_diff[layer_name]
        layer_diff_over_layer_max.append(layer_diff_col[idx] / max)

        layer_diff_over_max.append(layer_diff_col[idx] / max_diff)

    data_dict = {
        "layer_idx": layer_idx_col,
        "layer_name": layer_name_col,
        "layer_diff_mean": layer_diff_col,
        "layer_diff_over_layer_max": layer_diff_over_layer_max,
        "layer_diff_over_average": layer_diff_over_average,
        "layer_diff_over_max": layer_diff_over_max,
    }

    return pandas.DataFrame.from_dict(data_dict)


def save_json(content, path, indent=4, **json_dump_kwargs):
    for epoch in content.keys():
        content[epoch] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in content[epoch].items()}
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def return_average_metric(metric):
    if any(['rouge2' in k for k in metric.keys()]):
        R2 = [v for k, v in metric.items() if 'rouge2' in k]
        R2 = sum(R2)/len(R2)
        metric['avg_rouge2'] = R2
    if any(['meteor' in k for k in metric.keys()]):
        MET = [v for k, v in metric.items() if 'meteor' in k]
        MET = sum(MET) / len(MET)
        metric['avg_meteor'] = MET
    if any(['bleu' in k for k in metric.keys()]):
        bleu = [v for k, v in metric.items() if 'bleu' in k]
        bleu = sum(bleu) / len(bleu)
        metric['avg_bleu'] = bleu
    return metric