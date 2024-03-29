from catalyst.contrib.utils.plotly import _get_tensorboard_scalars, _get_scatter
from typing import Dict, List, Optional, Union
from collections import defaultdict
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


def plot_tensorboard_log_custom(
        logdir: Union[str, Path],
        step: Optional[str] = "batch",
        metrics: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
) -> None:
    """
    @TODO: Docs. Contribution is welcome.

    Adapted from
    https://github.com/belskikh/kekas/blob/v0.1.23/kekas/utils.py#L193
    """
    init_notebook_mode()
    logdir = Path(logdir)

    logdirs = {
        x.name.replace("_log", ""): x
        for x in logdir.glob("**/*")
        if x.is_dir() and str(x).endswith("_log")
    }

    scalars_per_loader = {
        key: _get_tensorboard_scalars(inner_logdir, metrics, step)
        for key, inner_logdir in logdirs.items()
    }

    scalars_per_metric = defaultdict(dict)
    for key, value in scalars_per_loader.items():
        for key2, value2 in value.items():
            scalars_per_metric[key2][key] = value2
    metric_data = []
    for metric_name, metric_logs in scalars_per_metric.items():

        for key, value in metric_logs.items():
            if key == 'valid':
                try:
                    data_ = _get_scatter(value, f"{key}/{metric_name}")
                    metric_data.append(data_)
                except:  # noqa: E722
                    pass

    layout = go.Layout(
        title='accuracy',
        height=height,
        width=width,
        yaxis={"hoverformat": ".5f"},
    )
    iplot(go.Figure(data=metric_data, layout=layout))


def plot_metrics_custom(
        logdir: Union[str, Path],
        step: Optional[str] = "epoch",
        metrics: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
) -> None:
    """
    Plots your learning results.

    Args:
        logdir: the logdir that was specified during training.
        step: 'batch' or 'epoch' - what logs to show: for batches or
            for epochs
        metrics: list of metrics to plot. The loss should be specified as
            'loss', learning rate = '_base/lr' and other metrics should be
            specified as names in metrics dict
            that was specified during training
        height: the height of the whole resulting plot
        width: the width of the whole resulting plot

    """
    assert step in [
        "batch",
        "epoch",
    ], f"Step should be either 'batch' or 'epoch', got '{step}'"
    metrics = metrics or ["loss"]
    plot_tensorboard_log_custom(logdir, step, metrics, height, width)
