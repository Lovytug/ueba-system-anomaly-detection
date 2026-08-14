import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


class ReportBuilder:
    def __init__(
        self,
        top_features=10,
    ):
        self.top_features = top_features


    def _get_top_features(
        self,
        feature_error,
        feature_names,
        top=None,
    ):
        if top is None:
            top = self.top_features

        feature_error = pd.Series(
            feature_error,
            index=feature_names,
            dtype=float,
        )

        total = feature_error.sum()
        if total <= 0:
            contribution = pd.Series(0.0,index=feature_names)

        else:
            contribution = feature_error / total * 100

        contribution = (
            contribution
            .sort_values(
                ascending=False
            )
            .head(top)
            .round(2)
        )

        return [
            {
                "feature": feature,
                "contribution": float(value),
            }
            for feature, value in contribution.items()
        ]


    def append_dataframe(
        self,
        row,
        run_dir,
        filename,
    ):

        path = Path(run_dir) / filename
        
        # еслу нету, создали дерект
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame([row])
        
        df.to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )
        
        return path


    def build(
        self,
        detection_result,
        timestamp=None,
    ):
 
        feature_names = detection_result["feature_names"]
        feature_error = detection_result["explanation"]["feature_error"]

        if len(feature_error.shape) > 1:
            feature_error = feature_error[0]

        top_features = self._get_top_features(
            feature_error=feature_error,
            feature_names=feature_names,
        )

        report = {
            "timestamp": (        timestamp
            if timestamp is not None
            else datetime.now().isoformat()
        ),
            "objects": 1,

            "alarm": bool(detection_result["alarm"]),

            "risk": float(detection_result["risk"]),

            "error": float(detection_result["error"]),

            "top_features": top_features,
        }

        return report


    def build_batch(
        self,
        detection_result,
    ):

        alarms = detection_result["alarm"]

        risks = detection_result["risk"]
        
        errors = detection_result["errors"]

        feature_errors = detection_result["explanation"]["feature_error"]

        feature_names = detection_result["feature_names"]

        detections = []
        for i in range(len(errors)):
            top_features = self._get_top_features(
                feature_error=feature_errors[i],
                feature_names=feature_names,
            )

            detections.append(
                {
                    "index": i,

                    "error": float(errors[i]),

                    "risk": float(risks[i]),

                    "alarm": bool(alarms[i]),

                    "top_features": top_features,
                }
            )

        report = {

            "timestamp": datetime.now().isoformat(),

            "objects": len(errors),

            "alarms": int(sum(alarms)),

            "max_risk": float(max(risks)),

            "max_error": float(max(errors)),

            "detections": detections,
        }

        return report



class ReportManager:

    def __init__(
        self,
        result_dir="results",
    ):

        self.result_dir = Path(result_dir)


    def create_run(self):

        run_dir = (
            self.result_dir
            / datetime.now().strftime("%Y-%m-%d")
            / datetime.now().strftime("%H-%M-%S")
        )

        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir


    def save_report(
        self,
        report,
        run_dir,
        filename="report.json",
    ):

        path = run_dir / filename

        with open(path, "w", encoding="utf8") as f:

            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
            )

        return path


    def save_parameters(
        self,
        params,
        run_dir,
        filename="params.json",
    ):

        path = run_dir / filename

        with open(path, "w", encoding="utf8") as f:

            json.dump(
                params,
                f,
                indent=4,
                ensure_ascii=False,
                default=lambda obj: obj.item()
                if isinstance(obj, np.generic)
                else obj,
            )

        return path


    def save_dataframe(
        self,
        df,
        run_dir,
        filename,
    ):

        path = run_dir / filename

        df.to_csv(path, index=False)

        return path


    def append_dataframe(
        self,
        row,
        run_dir,
        filename,
    ):
        path = run_dir / filename

        df = pd.DataFrame([row])

        df.to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )

        return path


    def save_text(
        self,
        text,
        run_dir,
        filename="log.txt",
    ):

        path = run_dir / filename

        with open(path, "w", encoding="utf8",) as f:
            f.write(text)

        return path
