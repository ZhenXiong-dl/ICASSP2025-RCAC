from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from comfy_extras.nodes_mask import MaskComposite, SolidMask

from .attention_couple import AttentionCouple

class ComfyCouple:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "orientation": (["horizontal", "vertical"],),
                "center": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
    )

    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(
            self,
            model,
            positive_1,
            positive_2,
            negative,
            orientation,
            center,
            width,
            height,
    ):
        mask_rect_first_x = None
        mask_rect_first_y = None
        mask_rect_first_width = None
        mask_rect_first_height = None

        mask_rect_second_x = None
        mask_rect_second_y = None
        mask_rect_second_width = None
        mask_rect_second_height = None

        if orientation == "horizontal":
            width_first = int(width * center)

            mask_rect_first_x = width_first
            mask_rect_first_y = 0
            mask_rect_first_width = width - width_first
            mask_rect_first_height = height
            mask_rect_second_x = 0
            mask_rect_second_y = 0
            mask_rect_second_width = width_first
            mask_rect_second_height = height
        elif orientation == "vertical":
            height_first = int(height * center)

            mask_rect_first_x = 0
            mask_rect_first_y = height_first
            mask_rect_first_width = width
            mask_rect_first_height = height - height_first
            mask_rect_second_x = 0
            mask_rect_second_y = 0
            mask_rect_second_width = width
            mask_rect_second_height = height_first

        solid_mask_zero = SolidMask().solid(0.0, width, height)[0]

        solid_mask_first = SolidMask().solid(1.0, mask_rect_first_width, mask_rect_first_height)[0]
        solid_mask_second = SolidMask().solid(1.0, mask_rect_second_width, mask_rect_second_height)[0]

        mask_composite_first = MaskComposite().combine(solid_mask_zero, solid_mask_first, mask_rect_first_x, mask_rect_first_y, "add")[0]
        mask_composite_second = MaskComposite().combine(solid_mask_zero, solid_mask_second, mask_rect_second_x, mask_rect_second_y, "add")[0]

        conditioning_mask_first = ConditioningSetMask().append(positive_1, mask_composite_second, "default", 1.0)[0]
        conditioning_mask_second = ConditioningSetMask().append(positive_2, mask_composite_first, "default", 1.0)[0]

        positive_combined = ConditioningCombine().combine(conditioning_mask_first, conditioning_mask_second)[0]

        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention")

NODE_CLASS_MAPPINGS = {
    "Comfy Couple": ComfyCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Couple": "Comfy Couple",
}
