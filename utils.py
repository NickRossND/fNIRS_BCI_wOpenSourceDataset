# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import os


def mkdir(path):
    """
    Create directory if it doesn't exist.
    Utility function for creating results directories.
    
    Args:
        path: Directory path to create
        
    Returns:
        True if directory was created, False if it already existed
    """
    path = path.strip()  # Remove leading/trailing whitespace
    path = path.rstrip("\\")  # Remove trailing backslash (Windows path handling)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)  # Create directory and all parent directories
        print(path + ' create success')
        return True
    else:
        print(path + ' already exist')
        return False

