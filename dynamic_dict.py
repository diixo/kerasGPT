
class DynamicDict:

    def __init__(self, attributes):

        if not isinstance(attributes, dict):
            raise ValueError("attributes must be a dictionary")

        for key, value in attributes.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"DynamicAttributes({vars(self)})"
