# fer_live/utils.py
def apply_torch_cpu_safety():
    import torch
    _orig_load = torch.load
    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    torch.load = _cpu_load

    try:
        from torch.serialization import add_safe_globals
        from timm.models.efficientnet import EfficientNet
        add_safe_globals([EfficientNet])
    except Exception:
        pass

    try:
        from timm.models._efficientnet_blocks import DepthwiseSeparableConv
        if not hasattr(DepthwiseSeparableConv, "conv_s2d"):
            DepthwiseSeparableConv.conv_s2d = None
    except Exception:
        pass
