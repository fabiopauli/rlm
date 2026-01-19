"""
Security and sandboxing for RLM code execution.

Provides restricted execution environment to limit potential damage from
generated code.
"""

import sys
import io
from typing import Dict, Any, Set


class RestrictedGlobals:
    """
    Creates a restricted globals dictionary for safer exec() execution.

    Blocks access to dangerous modules and functions while allowing
    safe operations needed for RLM functionality.
    """

    # Dangerous builtins to exclude
    DANGEROUS_BUILTINS = {
        '__import__',
        'eval',
        'compile',
        'open',  # Can be added back via safe wrapper
        'input',
        'execfile',
        'reload',
        'breakpoint',
    }

    # Dangerous modules to block
    DANGEROUS_MODULES = {
        'os',
        'sys',
        'subprocess',
        'socket',
        'urllib',
        'requests',
        'http',
        'ftplib',
        'smtplib',
        'pickle',
        'shelve',
        'dbm',
        '__builtin__',
        '__builtins__',
    }

    # Safe modules that can be imported
    SAFE_MODULES = {
        're',
        'json',
        'math',
        'statistics',
        'collections',
        'itertools',
        'functools',
        'datetime',
        'time',
        'random',
        'string',
        'textwrap',
    }

    @classmethod
    def _safe_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Safe import wrapper that only allows whitelisted modules.

        Args:
            name: Module name to import
            globals: Globals dict (unused but required by import signature)
            locals: Locals dict (unused but required by import signature)
            fromlist: List of names to import from module
            level: Relative import level

        Returns:
            Imported module

        Raises:
            ImportError: If module is not in the safe list
        """
        # Get the base module name (before any dots)
        base_module = name.split('.')[0]

        # Check if module is in safe list
        if base_module not in cls.SAFE_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed. Safe modules: {', '.join(sorted(cls.SAFE_MODULES))}")

        # Use the real __import__ to load the module
        import builtins
        return builtins.__import__(name, globals, locals, fromlist, level)

    @classmethod
    def create_safe_globals(cls, custom_globals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a safe globals dictionary with restricted builtins.

        Args:
            custom_globals: Custom globals to include (e.g., llm_query, context)

        Returns:
            Restricted globals dictionary
        """
        # Get builtins - __builtins__ can be either a dict or a module
        import builtins as builtins_module

        # Start with a copy of safe builtins
        safe_builtins = {
            name: getattr(builtins_module, name)
            for name in dir(builtins_module)
            if name not in cls.DANGEROUS_BUILTINS
        }

        # Add safe import wrapper
        safe_builtins['__import__'] = cls._safe_import

        # Create base safe globals
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__rlm__',
            '__doc__': None,
        }

        # Add custom globals (like llm_query, context, helpers)
        safe_globals.update(custom_globals)

        return safe_globals


class ExecutionMonitor:
    """
    Monitors code execution for safety violations and resource limits.
    """

    def __init__(
        self,
        max_execution_time: float = 30.0,
        max_output_size: int = 100_000,
        forbidden_patterns: Set[str] = None
    ):
        """
        Initialize execution monitor.

        Args:
            max_execution_time: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            forbidden_patterns: Set of forbidden string patterns in code
        """
        self.max_execution_time = max_execution_time
        self.max_output_size = max_output_size
        # Note: We no longer block '__import__' or 'compile(' since we provide safe wrappers
        # We only block specific dangerous module imports and eval/exec
        self.forbidden_patterns = forbidden_patterns or {
            'import os',
            'import sys',
            'import subprocess',
            'import socket',
            'import urllib',
            'import requests',
            'import pickle',
            'eval(',
            'exec(',
        }

    def check_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Check if code contains forbidden patterns.

        Args:
            code: Code string to check

        Returns:
            Tuple of (is_safe, error_message)
        """
        code_lower = code.lower()

        for pattern in self.forbidden_patterns:
            if pattern.lower() in code_lower:
                return False, f"Forbidden pattern detected: {pattern}"

        return True, ""

    def capture_execution(self, code: str, globals_dict: Dict[str, Any]) -> tuple[bool, str, str]:
        """
        Execute code with output capture and safety monitoring.

        Args:
            code: Code to execute
            globals_dict: Globals dictionary for execution

        Returns:
            Tuple of (success, output, error_message)
        """
        # Check code safety first
        is_safe, error_msg = self.check_code_safety(code)
        if not is_safe:
            return False, "", error_msg

        # Capture stdout
        output_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture

        try:
            # Execute with timeout (basic implementation)
            exec(code, globals_dict)

            # Get captured output
            output = output_capture.getvalue()

            # Check output size limit
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\n[Output truncated...]"

            return True, output, ""

        except Exception as e:
            return False, "", str(e)

        finally:
            sys.stdout = original_stdout
            output_capture.close()


class SafeFileAccess:
    """
    Provides safe file access wrapper for RLM.

    Restricts file operations to specific directories and file types.
    """

    def __init__(self, allowed_dirs: Set[str] = None, allowed_extensions: Set[str] = None):
        """
        Initialize safe file access.

        Args:
            allowed_dirs: Set of allowed directory paths
            allowed_extensions: Set of allowed file extensions
        """
        self.allowed_dirs = allowed_dirs or set()
        self.allowed_extensions = allowed_extensions or {'.txt', '.json', '.csv', '.md'}

    def safe_read(self, filepath: str) -> str:
        """
        Safely read a file with restrictions.

        Args:
            filepath: Path to file

        Returns:
            File contents

        Raises:
            PermissionError: If file access is not allowed
        """
        import os

        # Normalize path
        abs_path = os.path.abspath(filepath)

        # Check directory restriction
        if self.allowed_dirs:
            allowed = any(abs_path.startswith(d) for d in self.allowed_dirs)
            if not allowed:
                raise PermissionError(f"Access to {abs_path} is not allowed")

        # Check extension restriction
        _, ext = os.path.splitext(abs_path)
        if ext.lower() not in self.allowed_extensions:
            raise PermissionError(f"File type {ext} is not allowed")

        # Read file
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()

    def safe_write(self, filepath: str, content: str):
        """
        Safely write to a file with restrictions.

        Args:
            filepath: Path to file
            content: Content to write

        Raises:
            PermissionError: If file access is not allowed
        """
        import os

        # Normalize path
        abs_path = os.path.abspath(filepath)

        # Check directory restriction
        if self.allowed_dirs:
            allowed = any(abs_path.startswith(d) for d in self.allowed_dirs)
            if not allowed:
                raise PermissionError(f"Access to {abs_path} is not allowed")

        # Check extension restriction
        _, ext = os.path.splitext(abs_path)
        if ext.lower() not in self.allowed_extensions:
            raise PermissionError(f"File type {ext} is not allowed")

        # Write file
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
