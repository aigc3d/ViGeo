import csv
import math
from pathlib import Path


ID_COLUMNS = {'task', 'dataset', 'benchmark'}


def make_summary_row(task, dataset, benchmark, metrics, columns):
    row = {column: '' for column in columns}
    row['task'] = task
    row['dataset'] = dataset
    row['benchmark'] = benchmark
    for metric, value in metrics.items():
        if metric in row:
            row[metric] = value
    return row


def read_summary_rows(csv_path, columns):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            return []
        return [{column: row.get(column, '') for column in columns} for row in reader]


def merge_summary_rows(csv_path, new_rows, columns):
    new_rows = [{column: row.get(column, '') for column in columns} for row in new_rows]
    existing_rows = read_summary_rows(csv_path, columns)

    new_keys = {(row['task'], row['dataset'], row['benchmark']) for row in new_rows}
    merged_rows = [
        row for row in existing_rows
        if (row.get('task'), row.get('dataset'), row.get('benchmark')) not in new_keys
    ]
    merged_rows.extend(new_rows)
    return sorted(merged_rows, key=lambda row: (row['task'], row['benchmark'], row['dataset']))


def format_table_value(column, value):
    if column in ID_COLUMNS or value == '' or value is None:
        return value

    try:
        number = float(value)
    except (TypeError, ValueError):
        return value

    if not math.isfinite(number):
        return ''
    return f'{number:.3f}'


def format_table_rows(rows, columns):
    return [
        {column: format_table_value(column, row.get(column, '')) for column in columns}
        for row in rows
    ]


def write_summary_table(output_dir, output_prefix, columns, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    rows = merge_summary_rows(csv_path, rows, columns)
    if not rows:
        return

    rows = format_table_rows(rows, columns)

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary table ({len(rows)} rows): {csv_path}")


def has_valid_metrics(metrics, primary_metric=None):
    if not metrics:
        return False

    metric_name = primary_metric or next(iter(metrics))
    value = metrics.get(metric_name, float('nan'))
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def format_metrics(metrics):
    return ", ".join([f"{key}: {float(value):.4f}" for key, value in metrics.items()])


def print_summary(dataset, metrics, line_formatter=None):
    print(f"\n[SUMMARY] {dataset.upper()}")
    for key, value in metrics.items():
        if line_formatter is None:
            print(f"{key.ljust(10)}: {value:.4f}")
        else:
            print(line_formatter(key, value))
    print("-" * 30)
