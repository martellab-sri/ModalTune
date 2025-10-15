from typing import Any

import torch


class Aggregator(torch.nn.Module):
    """
    To register the subclasses based on the name
    To be used as
    @Aggregator.register("some_name")
    class SomeClass(BaseModel)

    To build that class, one can use
    Aggregator.create("some_name")
    """

    subclasses = {}

    def __init__(self) -> None:
        super().__init__()
        self.mode = "classifier"

    @classmethod
    def register(cls, subclass_name: str):
        def decorator(subclass: Any):
            subclass.subclasses[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_name: str, **params):
        if subclass_name not in cls.subclasses:
            raise ValueError("Unknown subclass name {}".format(subclass_name))
        print("-" * 50)
        print(
            f"For class: {cls.__name__}, Selected subclass: ({subclass_name}):{cls.subclasses[subclass_name]}"
        )
        print("-" * 50)

        return cls.subclasses[subclass_name](**params)

    def return_logits(self, h):
        """
        features: 1 x n_dim
        """
        if self.mode == "feature":
            return h
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        if self.mode == "classifier":
            return logits.squeeze()
        elif self.mode == "survival":
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat
        else:
            raise NotImplementedError
