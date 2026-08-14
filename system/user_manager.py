from pathlib import Path
import time

import pandas as pd

from collector.collector import DataCollector

from system.system import UEBATrainer
from system.system import UEBADetector

from system.reporter import ReportBuilder
from system.reporter import ReportManager


class UEBATrainManager:

    def __init__(
        self,
        results_dir="results",
        device="cpu",
    ):

        self.device = device

        self.report_manager = ReportManager(
            result_dir=results_dir
        )


    def _copy_file(
        self,
        source,
        destination,
    ):

        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(
                f"Файл не найден: {source}"
            )

        destination.write_bytes(
            source.read_bytes()
        )


    def _collect_dataset(
        self,
        path,
        samples,
        interval,
    ):

        collector = DataCollector(
            csv_file=path,
            interval=interval,
        )

        return collector.collect_dataset( samples )


    def train(
        self,
        train_path=None,
        collect=False,
        samples=1000,
        interval=5,
        epochs=50,
        hidden_dims=(64,16),
        batch_size=256,
        train_ratio=0.8,
        lr=1e-3,
    ):

        run_dir = self.report_manager.create_run()

        if train_path is not None:

            train_path = Path(train_path)

            self._copy_file(
                train_path,
                run_dir / "train.csv",
            )

            df = pd.read_csv(train_path)

        elif collect:

            train_file = run_dir / "train.csv"

            df = self._collect_dataset(
                train_file,
                samples,
                interval
            )

        else:
            raise ValueError(
                "Нужно указать train_path или collect=True"
            )

        trainer = UEBATrainer(
            hidden_dims=hidden_dims,
            batch_size=batch_size,
            train_ratio=train_ratio,
            lr=lr,
            device=self.device
        )

        trainer.fit(df, epochs=epochs)

        trainer.save(run_dir / "model.joblib")

        self.report_manager.save_parameters(
            trainer.detector_params,
            run_dir,
            "detector_params.json",
        )

        return run_dir



class BaseDetectionManager:

    def __init__(
        self,
        results_dir="results",
        device="cpu",
        top_features=10,
    ):
        self.device = device

        self.report_builder = ReportBuilder(
            top_features=top_features,
        )

        self.report_manager = ReportManager(
            result_dir=results_dir,
        )

    def _copy_file(self, source, destination):
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(
                f"Файл не найден: {source}"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(
            source.read_bytes()
        )

    def _load_detector(self, model_path):
        return UEBADetector.load(
            model_path,
            device=self.device,
        )



class UEBADetectionManager(BaseDetectionManager):

    def detect(self, sample, model_path):
        run_dir = self.report_manager.create_run()

        self._copy_file(
            model_path,
            run_dir / "model.joblib",
        )

        detector = self._load_detector(model_path)
        result = detector.detect(sample)

        report = self.report_builder.build(result)

        self.report_manager.save_report(
            report,
            run_dir,
        )

        return result


    def detect_batch(
        self,
        data_path,
        model_path,
        batch_size=256,
    ):
        run_dir = self.report_manager.create_run()

        data_path = Path(data_path)
        model_path = Path(model_path)

        self._copy_file(
            data_path,
            run_dir / "inference.csv",
        )

        self._copy_file(
            model_path,
            run_dir / "model.joblib",
        )

        df = pd.read_csv(data_path)

        detector = self._load_detector(model_path)

        result = detector.detect_batch(
            df,
            batch_size=batch_size,
        )

        report = self.report_builder.build_batch(result)

        self.report_manager.save_report(
            report,
            run_dir,
        )

        return result



class UEBAOnlineDetectionManager(BaseDetectionManager):

    def _build_online_report(
        self,
        detections,
        model_path,
        interval,
    ):
        alarms = [item["alarm"] for item in detections]
        risks = [item["risk"] for item in detections]
        errors = [item["error"] for item in detections]

        return {
            "mode": "online",
            "model": str(model_path),
            "interval": interval,
            "objects": len(detections),
            "alarms": sum(alarms),
            "max_risk": max(risks, default=0.0),
            "max_error": max(errors, default=0.0),
            "detections": detections,
        }

    def _print_detection(
        self,
        sample,
        result,
        detection_report,
    ):
        print(
            f"{sample['timestamp']} | "
            f"error={result['error']:.6f} | "
            f"risk={result['risk']:.6f} | "
            f"alarm={result['alarm']}"
        )

        if not result["alarm"]:
            return

        print("Топ признаки:")

        for item in detection_report["top_features"]:
            print(
                f"{item['feature']}: "
                f"{item['contribution']:.2f}%"
            )

    def run(
        self,
        model_path,
        interval=5,
    ):
        run_dir = self.report_manager.create_run()

        model_path = Path(model_path)

        self._copy_file(
            model_path,
            run_dir / "model.joblib",
        )

        detector = self._load_detector(model_path)

        inference_path = run_dir / "inference.csv"
        report_path = run_dir / "report.json"

        collector = DataCollector(
            csv_file=inference_path,
            interval=interval,
        )

        detections = []

        print("Online detection запущен")
        print(f"Модель: {model_path}")
        print(f"Интервал: {interval} сек")
        print(f"Результаты: {run_dir}")
        print("Для остановки нажмите Ctrl+C")

        try:
            while True:
                sample = collector.collect()
                collector.save(sample)

                result = detector.detect(
                    pd.Series(sample)
                )

                detection_report = self.report_builder.build(
                    result,
                    timestamp=sample["timestamp"],
                )

                detections.append(detection_report)

                report = self._build_online_report(
                    detections=detections,
                    model_path=model_path,
                    interval=interval,
                )

                self.report_manager.save_report(
                    report,
                    run_dir,
                    "report.json",
                )

                self._print_detection(
                    sample,
                    result,
                    detection_report,
                )

                time.sleep(interval)

        except KeyboardInterrupt:
            print()
            print("Online detection остановлен.")
            print(f"Отчёт сохранён: {report_path}")

        return run_dir
