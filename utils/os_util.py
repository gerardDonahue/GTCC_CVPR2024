import os


def get_env_variable(name):
    variable = os.environ.get(name)
    if not variable: raise ValueError(f"No {name} environment variable set. Please set!")
    elif variable.lower() in ['true', 'false']:
        return True if variable.lower() == 'true' else False
    return variable
