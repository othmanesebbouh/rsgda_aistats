import os
import numpy as np


def condition(dic, filterby):
    all_good = True
    for key, values in filterby.items():
        if key not in dic.keys():
            continue
        elif dic[key] not in values:
            all_good = False
            break
    return all_good


def filter_dicts_and_scores(dicts, scores, filterby):
    return [(dic, score) for dic, score in zip(dicts, scores) if condition(dic, filterby)]


def create_result_list(dicts, scores, filterby, metrics):
    dicts_and_scores = filter_dicts_and_scores(dicts, scores, filterby)
    result = []
    for dic, score in dicts_and_scores:
        score_dict = {}
        for metric in metrics:
            score_dict[metric] = []
            for i in range(len(score)):
                score_dict[metric].append(score[i][metric])
        result.append((dic, score_dict))
    
    return result


def filter_div(scores, dicts, div_criteria):
    keep = [True] * len(scores)
    for i, score in enumerate(scores):
        for key, (comparison_type, threshold) in div_criteria.items():
            if comparison_type == -1:
                if score[-1][key] < threshold:
                    keep[i] = False
                    break
            else:
                if score[-1][key] > threshold:
                    keep[i] = False
                    break
            
    scores = [scores[i] for i in range(len(scores)) if keep[i]]
    dicts = [dicts[i] for i in range(len(dicts)) if keep[i]]
    return scores, dicts, keep


def plot(ax, xs, ys, plot_type='semilog', ylim=(None, None), skip_steps=None):
    if plot_type == 'semilog':
        ax.semilogy(xs, ys)
    elif plot_type == 'skip':
        ys = np.array([a for i, a in enumerate(ys) if i % skip_steps == 0])
        xs = np.array([a for i, a in enumerate(xs) if i % skip_steps == 0])
        ax.semilogy(xs, ys)
    else:
        ax.plot(xs, ys)
    
    ax.set_ylim(ylim)
    

def combine_metrics(results, metrics_to_combine, metric_function, new_metric_name):
    for result in results:
        args = (result[1][metric] for metric in metrics_to_combine)
        result[1][new_metric_name] = metric_function(*args)