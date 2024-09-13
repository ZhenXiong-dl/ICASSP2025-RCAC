"""
@author: Rei D.
@title: Comfy Couple
@nickname: Danand
@description: If you want to draw two different characters together without blending their features, so you could try to check out this custom node.
"""

from .comfy_couple import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .comfy_couple_dit import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_FLUX
from .comfy_couple_dit import NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_FLUX
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_FLUX)
NODE_DISPLAY_NAME_MAPPINGS.update( NODE_DISPLAY_NAME_MAPPINGS_FLUX)

WEB_DIRECTORY = "js"
CC_VERSION = 2.0

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", 'CC_VERSION']
