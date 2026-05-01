"""Training helper functions for jpy-ml."""
from ultralytics import YOLO
import os, csv

_jpy_epoch_log = []

def _on_epoch_end(trainer):
    """Ultralytics callback: called after each epoch."""
    global _jpy_epoch_log
    entry = {
        'epoch': trainer.epoch + 1,
    }
    # Loss items: box_loss, cls_loss, dfl_loss
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        items = trainer.loss_items
        if hasattr(items, '__len__') and len(items) >= 3:
            entry['box_loss'] = float(items[0])
            entry['cls_loss'] = float(items[1])
            entry['dfl_loss'] = float(items[2])
    if hasattr(trainer, 'loss') and trainer.loss is not None:
        entry['total_loss'] = float(trainer.loss)
    if hasattr(trainer, 'best_fitness') and trainer.best_fitness is not None:
        entry['fitness'] = float(trainer.best_fitness)
    _jpy_epoch_log.append(entry)

def jpy_train(model_path, kwargs, enable_logging=False):
    """Run training and return result dict. If enable_logging=True, collect epoch metrics."""
    global _jpy_epoch_log
    _jpy_epoch_log = []

    model = YOLO(model_path)
    if enable_logging:
        model.add_callback("on_fit_epoch_end", _on_epoch_end)

    model.train(**kwargs)

    trainer = model.trainer
    result = {
        'best_model': str(trainer.best) if trainer and trainer.best else '',
        'last_model': str(trainer.last) if trainer and trainer.last else '',
        'save_dir': str(trainer.save_dir) if trainer else '',
        'epochs_completed': (trainer.epoch + 1) if trainer else 0,
        'best_fitness': float(trainer.best_fitness) if trainer and hasattr(trainer, 'best_fitness') and trainer.best_fitness else 0.0,
        'epoch_log': list(_jpy_epoch_log),
    }
    return result

def jpy_read_training_metrics(save_dir):
    """Read results.csv from training output."""
    csv_path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items() if v.strip()}
            rows.append(cleaned)
    return rows
