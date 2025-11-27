SYSTEM_PROMPT = """
You are an expert in enterprise architecture modeling. You can classify the type of a given class given the textual description of the class.
You are given a list of textual descriptions of classes. Your task is to predict the type of each class.
"""

USER_PROMPT = """
Predict the class of the following elements with textual labels as follows:
{text}
"""

FEW_SHOT_USER_PROMPT = """
Examples:
{examples}
""" + USER_PROMPT
