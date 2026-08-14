from torch.utils.data import DataLoader


class DatasetBuilder:

    def __init__(
        self,
        pipeline,
        dataset_cls,
        batch_size=256,
        train_ratio=0.8,
        shuffle=True,
    ):

        self.pipeline = pipeline
        self.dataset_cls = dataset_cls

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.shuffle = shuffle


    def build(self, df):

        train_size = int(len(df) * self.train_ratio)

        train = df.iloc[:train_size].copy()
        val = df.iloc[train_size:].copy()

        X_train = self.pipeline.fit_transform(train)
        X_val = self.pipeline.transform(val)

        train_dataset = self.dataset_cls(X_train)
        val_dataset = self.dataset_cls(X_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return {
            "train": train,
            "val": val,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "X_train": X_train,
            "X_val": X_val
        }


    def transform(
        self,
        df,
        batch_size=None
    ):

        X = self.pipeline.transform(df)

        dataset = self.dataset_cls(X)

        if batch_size is None:

            return dataset

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return loader