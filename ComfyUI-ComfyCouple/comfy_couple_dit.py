from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from comfy_extras.nodes_mask import MaskComposite, SolidMask

from .attention_couple_dit import AttentionCouple, AttentionCoupleMiddle

class ComfyCouple4:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "positive_3": ("CONDITIONING",),
                "positive_4": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                # "orientation": (["horizontal", "vertical"],),
                "divide_X": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                "divide_Y": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
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
            positive_3,
            positive_4,
            negative,
            divide_X,
            divide_Y,
            width,
            height,
    ):
        mask_rect1_x = None
        mask_rect1_y = None
        mask_rect1_width = None
        mask_rect1_height = None

        mask_rect2_x = None
        mask_rect2_y = None
        mask_rect2_width = None
        mask_rect2_height = None

        w1= int(width * divide_X)
        h1= int(height * divide_Y)

        mask_rect1_x = 0
        mask_rect1_y = 0
        mask_rect1_width =w1
        mask_rect1_height = h1
        mask_rect2_x = w1
        mask_rect2_y = 0
        mask_rect2_width = width - w1
        mask_rect2_height = h1

        mask_rect3_x = 0
        mask_rect3_y = h1
        mask_rect3_width = w1
        mask_rect3_height = height - h1
        mask_rect4_x = w1
        mask_rect4_y = h1
        mask_rect4_width = width - w1 
        mask_rect4_height =  height - h1

        solid_mask_zero = SolidMask().solid(0.0, width, height)[0]

        solid_mask1 = SolidMask().solid(1.0, mask_rect1_width, mask_rect1_height)[0]
        solid_mask2 = SolidMask().solid(1.0, mask_rect2_width, mask_rect2_height)[0]
        solid_mask3 = SolidMask().solid(1.0, mask_rect3_width, mask_rect3_height)[0]
        solid_mask4 = SolidMask().solid(1.0, mask_rect4_width, mask_rect4_height)[0]

        mask_composite1 = MaskComposite().combine(solid_mask_zero, solid_mask1, mask_rect1_x, mask_rect1_y, "add")[0]
        mask_composite2 = MaskComposite().combine(solid_mask_zero, solid_mask2, mask_rect2_x, mask_rect2_y, "add")[0]
        mask_composite3 = MaskComposite().combine(solid_mask_zero, solid_mask3, mask_rect3_x, mask_rect3_y, "add")[0]
        mask_composite4 = MaskComposite().combine(solid_mask_zero, solid_mask4, mask_rect4_x, mask_rect4_y, "add")[0]

        conditioning_mask1 = ConditioningSetMask().append(positive_1, mask_composite1, "default", 1.0)[0]
        conditioning_mask2 = ConditioningSetMask().append(positive_2, mask_composite2, "default", 1.0)[0]
        conditioning_mask3 = ConditioningSetMask().append(positive_3, mask_composite3, "default", 1.0)[0]
        conditioning_mask4 = ConditioningSetMask().append(positive_4, mask_composite4, "default", 1.0)[0]

        # positive_combined = ConditioningCombine().combine(conditioning_mask1, conditioning_mask2,conditioning_mask3,conditioning_mask4)[0]
        positive_combined=conditioning_mask1 + conditioning_mask2+conditioning_mask3+conditioning_mask4
        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention")
    
POS_LIM=9
class ComfyCoupleN:
    @classmethod
    def INPUT_TYPES(cls):
        inputs={
            "required": {
                "model": ("MODEL",),
                "positive_count": ("INT", {"default": 9, "min": 0, "max": POS_LIM, "step": 2}),
                "negative": ("CONDITIONING",),
                # "orientation": (["horizontal", "vertical"],),
                # "divide_X": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                # "divide_Y": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            }
        }
        # POS_count=inputs["required"]["positive_count"]
        for i in range(1, POS_LIM+1):
            inputs["required"][f"positive_{i}"]=("CONDITIONING",)
        x=int(pow(POS_LIM,0.5))
        y=int(POS_LIM/x)
        for i in  range(1, x):
            #  "divide_X": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
            inputs["required"][f"divideX_{i}"] = ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.00001})
        for i in  range(1, y):
            inputs["required"][f"divideY_{i}"] =  ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.00001})

        return inputs

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
            negative,
            positive_count,
            width,
            height,
            **kwargs
    ):
        divideX_count=int(pow(positive_count,0.5))
        divideY_count=int(positive_count/divideX_count)
        positive_prompts = [kwargs.get(f"positive_{i}") for i in range(1, positive_count + 1)]
        divideX_line = [kwargs.get(f"divideX_{i}") for i in range(1, divideX_count)]
        divideY_line = [kwargs.get(f"divideY_{i}") for i in range(1, divideY_count)]

        divideX_line.append(1.0)
        divideY_line.append(1.0)
        rects=[]
        w0=0
        h0=0
        for divideY in divideY_line:
            for divideX in divideX_line:
                w1= int(width * divideX)
                h1= int(height * divideY)
                rect_width=w1-w0
                rect_height=h1-h0
                rect_x=w0
                rect_y=h0
                w0=w1
                rects.append((rect_x, rect_y, rect_width, rect_height))
            h0=h1
            w0=0
        

        solid_mask_zero = SolidMask().solid(0.0, width, height)[0]

        # solid_mask1 = SolidMask().solid(1.0, mask_rect1_width, mask_rect1_height)[0]
        # mask_composite1 = MaskComposite().combine(solid_mask_zero, solid_mask1, mask_rect1_x, mask_rect1_y, "add")[0]
        all_cond=[]
        for i,rect in enumerate(rects):
            solid_mask = SolidMask().solid(1.0, rect[2], rect[3])[0]
            mask_composite = MaskComposite().combine(solid_mask_zero, solid_mask,  rect[0],  rect[1], "add")[0]
            conditioning_mask = ConditioningSetMask().append(positive_prompts[i], mask_composite, "default", 1.0)[0]
            all_cond.extend(conditioning_mask)
        positive_combined =all_cond

        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention")
    
class ComfyCouple2:

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
    

class ComfyCouple4_1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "positive_3": ("CONDITIONING",),
                "positive_4": ("CONDITIONING",),
                "positive_all": ("CONDITIONING",),
                "block_nums": ("INT", {"default": 21, "min": 0, "max": 40, "step": 1}),
                "negative": ("CONDITIONING",),
                "divide_X": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                "divide_Y": ("FLOAT", {"default": 0.5, "min": 0, "max": 0.99, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
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
            positive_3,
            positive_4,
            positive_all,
            negative,
            divide_X,
            divide_Y,
            width,
            height,
            block_nums
    ):
        mask_rect1_x = None
        mask_rect1_y = None
        mask_rect1_width = None
        mask_rect1_height = None

        mask_rect2_x = None
        mask_rect2_y = None
        mask_rect2_width = None
        mask_rect2_height = None

        w1= int(width * divide_X)
        h1= int(height * divide_Y)

        mask_rect1_x = 0
        mask_rect1_y = 0
        mask_rect1_width =w1
        mask_rect1_height = h1
        mask_rect2_x = w1
        mask_rect2_y = 0
        mask_rect2_width = width - w1
        mask_rect2_height = h1

        mask_rect3_x = 0
        mask_rect3_y = h1
        mask_rect3_width = w1
        mask_rect3_height = height - h1
        mask_rect4_x = w1
        mask_rect4_y = h1
        mask_rect4_width = width - w1 
        mask_rect4_height =  height - h1

        solid_mask_zero = SolidMask().solid(0.0, width, height)[0]

        solid_mask1 = SolidMask().solid(1.0, mask_rect1_width, mask_rect1_height)[0]
        solid_mask2 = SolidMask().solid(1.0, mask_rect2_width, mask_rect2_height)[0]
        solid_mask3 = SolidMask().solid(1.0, mask_rect3_width, mask_rect3_height)[0]
        solid_mask4 = SolidMask().solid(1.0, mask_rect4_width, mask_rect4_height)[0]

        mask_composite1 = MaskComposite().combine(solid_mask_zero, solid_mask1, mask_rect1_x, mask_rect1_y, "add")[0]
        mask_composite2 = MaskComposite().combine(solid_mask_zero, solid_mask2, mask_rect2_x, mask_rect2_y, "add")[0]
        mask_composite3 = MaskComposite().combine(solid_mask_zero, solid_mask3, mask_rect3_x, mask_rect3_y, "add")[0]
        mask_composite4 = MaskComposite().combine(solid_mask_zero, solid_mask4, mask_rect4_x, mask_rect4_y, "add")[0]

        conditioning_mask1 = ConditioningSetMask().append(positive_1, mask_composite1, "default", 1.0)[0]
        conditioning_mask2 = ConditioningSetMask().append(positive_2, mask_composite2, "default", 1.0)[0]
        conditioning_mask3 = ConditioningSetMask().append(positive_3, mask_composite3, "default", 1.0)[0]
        conditioning_mask4 = ConditioningSetMask().append(positive_4, mask_composite4, "default", 1.0)[0]

        # positive_combined = ConditioningCombine().combine(conditioning_mask1, conditioning_mask2,conditioning_mask3,conditioning_mask4)[0]
        positive_combined=conditioning_mask1 + conditioning_mask2+conditioning_mask3+conditioning_mask4
        return AttentionCoupleMiddle().attention_couple(model, positive_combined,positive_all, negative,block_nums, "Attention")

NODE_CLASS_MAPPINGS = {
    "Comfy Couple Dit 2 blocks": ComfyCouple2,
    "Comfy Couple Dit 4 blocks": ComfyCouple4,
    "Comfy Couple Dit 4 blocks + 1":ComfyCouple4_1,
    "Comfy Couple Dit n blocks": ComfyCoupleN
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Couple Dit 2 blocks": "Comfy Couple Dit 2 blocks",
    "Comfy Couple Dit 4 blocks": "Comfy Couple Dit 4 blocks",
    "Comfy Couple Dit n blocks":"Comfy Couple Dit n blocks",
    "Comfy Couple Dit 4 blocks + 1": "Comfy Couple Dit 4 blocks + 1"
}
