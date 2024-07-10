import os
import motmetrics as mm
from loguru import logger

def evaluate(cfg, output_dir):
    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()
    gt_dir = "/Users/macbookair/Work/Multicam multi object tracking/Implementation/evaluation/gt"
    ts_dir = "/Users/macbookair/Work/Multicam multi object tracking/Implementation/evaluation/ts"
    accs = []
    names = []

    for file in os.listdir(gt_dir):
        print(file)
        gt_file = os.path.join(gt_dir, file)
        ts_file = os.path.join(ts_dir, file)
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")
        names.append(os.path.splitext(os.path.basename(ts_file))[0])
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5))
    summary = mh.compute_many(accs, metrics=metrics, generate_overall=True)#, name=names)
    logger.info(f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')


if __name__ == "__main__":
    evaluate(None, None)