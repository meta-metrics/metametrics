from typing import List, Any

def validate_argument_list(arg: Any, valid_arguments: List[Any]) -> Any:
    """
    Validate that the given argument is in the list of valid arguments.

    Args:
        valid_arguments (List[Any]): A list of valid arguments.
        arg (Any): The argument to validate.

    Returns:
        Any: The validated argument if it is in the list of valid arguments.

    Raises:
        ValueError: If the argument is not in the list of valid arguments.
    """
    if arg in valid_arguments:
        return arg
    else:
        raise ValueError(f"Invalid argument '{arg}'. Valid options are: {valid_arguments}")
    
def validate_real(arg: Any, valid_min: float = None, valid_max: float = None) -> float:
    """
    Validate that the given floating-point argument is within the specified range.

    Args:
        arg (Any): The argument to be validated.
        valid_min (float, optional): The minimum valid value (inclusive). Defaults to None.
        valid_max (float, optional): The maximum valid value (inclusive). Defaults to None.

    Returns:
        float: The validated floating-point argument.

    Raises:
        ValueError: If the argument is not within the specified range.
    """
    if not isinstance(arg, float) and not isinstance(arg, int):
        raise ValueError(f"Invalid argument '{arg}'. Must be a real value.")
    if valid_min is not None and arg < valid_min:
        raise ValueError(f"Invalid argument '{arg}'. Must be greater than or equal to {valid_min}.")
    if valid_max is not None and arg > valid_max:
        raise ValueError(f"Invalid argument '{arg}'. Must be less than or equal to {valid_max}.")
    
    return arg

def validate_int(arg: Any, valid_min: int = None, valid_max: int = None) -> int:
    """
    Validate that the given integer argument is within the specified range.

    Args:
        arg (Any): The argument to be validated.
        valid_min (int, optional): The minimum valid value (inclusive). Defaults to None.
        valid_max (int, optional): The maximum valid value (inclusive). Defaults to None.

    Returns:
        int: The validated integer argument.

    Raises:
        ValueError: If the argument is not within the specified range.
    """
    if not isinstance(arg, int):
        raise ValueError(f"Invalid argument '{arg}'. Must be a int value.")
    if valid_min is not None and arg < valid_min:
        raise ValueError(f"Invalid argument '{arg}'. Must be greater than or equal to {valid_min}.")
    if valid_max is not None and arg > valid_max:
        raise ValueError(f"Invalid argument '{arg}'. Must be less than or equal to {valid_max}.")
    
    return arg
    
def validate_bool(arg: Any) -> bool:
    """
    Validate that the given argument is a boolean value.

    Args:
        arg (Any): The argument to be validated.

    Returns:
        bool: The validated boolean argument.

    Raises:
        ValueError: If the argument is not a boolean.
    """
    if isinstance(arg, bool):
        return arg
    else:
        raise ValueError(f"Invalid argument '{arg}'. Must be a boolean value (True or False).")


