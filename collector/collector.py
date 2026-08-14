import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil


class DataCollector:

    COLUMNS = [
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "swap_percent",
        "disk_usage_percent",
        "disk_read_bytes_delta",
        "disk_write_bytes_delta",
        "disk_read_count_delta",
        "disk_write_count_delta",
        "bytes_sent_delta",
        "bytes_recv_delta",
        "packets_sent_delta",
        "packets_recv_delta",
        "process_count",
        "new_processes",
        "new_unique_processes",
        "tcp_total",
        "tcp_established",
        "tcp_time_wait",
        "tcp_close_wait",
        "tcp_listen",
        "hour",
        "weekday",
    ]

    def __init__(
        self,
        csv_file="dataset.csv",
        interval=5,
    ):
        self.csv_file = Path(csv_file)
        self.interval = interval

        self.prev_disk = psutil.disk_io_counters()
        self.prev_net = psutil.net_io_counters()

        self.prev_processes = set()
        self.prev_names = set()

        self._init_csv()


    def _init_csv(self):

        if self.csv_file.exists():
            return

        pd.DataFrame(
            columns=self.COLUMNS
        ).to_csv(
            self.csv_file,
            index=False,
        )


    def _collect_disk(self):

        disk = psutil.disk_io_counters()

        result = {
            "disk_read_bytes_delta": (
                disk.read_bytes - self.prev_disk.read_bytes
            ),

            "disk_write_bytes_delta": (
                disk.write_bytes - self.prev_disk.write_bytes
            ),

            "disk_read_count_delta": (
                disk.read_count - self.prev_disk.read_count
            ),

            "disk_write_count_delta": (
                disk.write_count - self.prev_disk.write_count
            ),
        }

        self.prev_disk = disk

        return result


    def _collect_network(self):

        net = psutil.net_io_counters()

        result = {
            "bytes_sent_delta": (
                net.bytes_sent - self.prev_net.bytes_sent
            ),

            "bytes_recv_delta": (
                net.bytes_recv - self.prev_net.bytes_recv
            ),

            "packets_sent_delta": (
                net.packets_sent - self.prev_net.packets_sent
            ),

            "packets_recv_delta": (
                net.packets_recv - self.prev_net.packets_recv
            ),
        }

        self.prev_net = net

        return result


    def _collect_processes(self):

        current_processes = set()
        current_names = set()

        for process in psutil.process_iter(["pid", "name"]):

            try:
                current_processes.add(process.info["pid"])
                current_names.add(process.info["name"])
            except Exception:
                pass

        result = {
            "process_count": len(current_processes),

            "new_processes": len(
                current_processes - self.prev_processes
            ),

            "new_unique_processes": len(
                current_names - self.prev_names
            ),
        }

        self.prev_processes = current_processes
        self.prev_names = current_names

        return result


    def _collect_tcp(self):

        connections = psutil.net_connections(
            kind="tcp"
        )

        statuses = {
            "ESTABLISHED": 0,
            "TIME_WAIT": 0,
            "CLOSE_WAIT": 0,
            "LISTEN": 0,
        }

        for connection in connections:
            if connection.status in statuses:
                statuses[connection.status] += 1

        return {
            "tcp_total": len(connections),
            "tcp_established": statuses["ESTABLISHED"],
            "tcp_time_wait": statuses["TIME_WAIT"],
            "tcp_close_wait": statuses["CLOSE_WAIT"],
            "tcp_listen": statuses["LISTEN"],
        }


    def collect(self):

        now = datetime.now()

        row = {
            "timestamp": now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "cpu_percent": psutil.cpu_percent(interval=None),

            "memory_percent": psutil.virtual_memory().percent,

            "swap_percent": psutil.swap_memory().percent,

            "disk_usage_percent": psutil.disk_usage("/").percent,

            "hour": now.hour,

            "weekday": now.weekday(),
        }

        row.update(self._collect_disk())
        row.update(self._collect_network())
        row.update(self._collect_processes())
        row.update(self._collect_tcp())

        return row

    def collect_dataset(self, samples):

        print("Сбор данных начат:")

        pd.DataFrame(
            columns=self.COLUMNS
        ).to_csv(
            self.csv_file,
            index=False,
        )

        rows = []

        for i in range(samples):
            row = self.collect()
            rows.append(row)

            self.save(row)

            print(
                datetime.now().strftime("%H:%M:%S"),
                "данные сохранены",
            )

            if i < samples - 1:
                time.sleep(self.interval)

        return pd.DataFrame(rows)


    def save(self, row):

        pd.DataFrame([row]).to_csv(
            self.csv_file,
            mode="a",
            header=False,
            index=False,
        )


    def run(self):
        
        print("Сбор данных начат:")

        while True:
            row = self.collect()
            self.save(row)

            print(
                datetime.now().strftime("%H:%M:%S"),
                "данные сохранены",
            )

            time.sleep(self.interval)