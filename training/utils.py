import re
from pathlib import Path


def infer_model_type(model_name_or_path: str or Path) -> str:
    """
    infer_model_type - infer the simpletransformers model type from the model name or path
        https://simpletransformers.ai/docs/ner-specifics/

    :param str model_name_or_path: the name or path of the model
    :return str: the model type
    """
    model_name_or_path = Path(model_name_or_path).name
    model_type = re.search(
        r"(bert|xlnet|roberta|mobilebert|deberta-v2|albert)",
        model_name_or_path,
        re.IGNORECASE,
    )
    if model_type:
        return model_type.group(1)
    else:
        raise ValueError(f"model type not recognized: {model_name_or_path}")
