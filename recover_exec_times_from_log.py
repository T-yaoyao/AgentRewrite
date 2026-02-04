import csv
import re
import sys
from pathlib import Path


def parse_log_to_csv(log_path: Path, output_csv: Path) -> None:
    """
    从执行日志中提取已完成 SQL 的执行时间并写入 CSV。

    识别的行格式示例：
      [1/157] 完成 id=1，最终执行时间=0.492091 秒

    输出 CSV 列：
      - id
      - execution_time_s
    """
    pattern = re.compile(r"完成 id=(\d+)，最终执行时间=([0-9.]+) 秒")

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append(
                    {"id": m.group(1), "execution_time_s": m.group(2)}
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "execution_time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"已从日志 {log_path} 提取 {len(rows)} 条记录，写入 {output_csv}"
    )


def main():
    if len(sys.argv) < 2:
        print(
            "用法: python recover_exec_times_from_log.py /path/to/log.txt [output.csv]",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path = Path(sys.argv[1]).expanduser().resolve()
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2]).expanduser().resolve()
    else:
        # 默认输出到当前目录下 log 文件名加后缀
        output_csv = log_path.with_suffix(".exec_times.csv")

    parse_log_to_csv(log_path, output_csv)


if __name__ == "__main__":
    main()




