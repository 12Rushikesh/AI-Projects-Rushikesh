CONFIDENCE_MAP = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.85
}

def confidence_to_float(conf):
    if isinstance(conf, (int, float)):
        return float(conf)
    if isinstance(conf, str):
        return CONFIDENCE_MAP.get(conf.lower(), 0.0)
    return 0.0
