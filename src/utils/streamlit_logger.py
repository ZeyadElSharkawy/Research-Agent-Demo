"""
Streamlit Logger - Captures agent activities for real-time display in Streamlit UI
"""

import io
import sys
from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional
import threading

class RealTimeStreamlitLogger:
    """Thread-safe logger that captures print statements and updates UI in real-time"""
    
    def __init__(self):
        self.logs: List[Tuple[str, str]] = []  # (log_level, message)
        self.lock = threading.Lock()
        self.callback: Optional[Callable] = None
    
    def clear(self):
        """Clear all logs"""
        with self.lock:
            self.logs = []
    
    def set_callback(self, callback: Optional[Callable]):
        """Set a callback function to be called when new logs are added"""
        with self.lock:
            self.callback = callback
    
    def add_log(self, message: str, level: str = "info"):
        """Add a log message and trigger callback if set"""
        with self.lock:
            self.logs.append((level, message))
            # Call the callback immediately if set
            if self.callback:
                try:
                    self.callback(level, message)
                except Exception as e:
                    # Don't let callback errors break logging
                    print(f"Logger callback error: {e}", file=sys.__stdout__)
    
    def get_logs(self):
        """Get all logs"""
        with self.lock:
            return self.logs.copy()
    
    def get_level(self, line: str) -> str:
        """Determine log level based on emoji/prefix"""
        if "❌" in line or "failed" in line.lower() or "error" in line.lower():
            return "error"
        elif "⚠️" in line or "warning" in line.lower():
            return "warning"
        elif "✅" in line or "complete" in line.lower():
            return "success"
        elif any(emoji in line for emoji in ["🔍", "📚", "🎯", "🧠", "📋", "🧭", "🚀"]):
            return "agent"
        return "info"


class RealTimeStdout:
    """Custom stdout that logs to both console and our logger in real-time"""
    
    def __init__(self, logger: RealTimeStreamlitLogger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout
        self.buffer = ""
    
    def write(self, text):
        """Write to both original stdout and capture for logger"""
        # Write to original stdout
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Add to buffer
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():
                level = self.logger.get_level(line)
                self.logger.add_log(line, level)
    
    def flush(self):
        """Flush the buffer"""
        self.original_stdout.flush()
        # Process any remaining buffer content
        if self.buffer.strip():
            level = self.logger.get_level(self.buffer)
            self.logger.add_log(self.buffer, level)
            self.buffer = ""
    
    def __getattr__(self, attr):
        """Delegate other attributes to original stdout"""
        return getattr(self.original_stdout, attr)


class StreamlitLogger:
    """Backward compatible wrapper"""
    
    def __init__(self):
        self._real_logger = RealTimeStreamlitLogger()
    
    def clear(self):
        self._real_logger.clear()
    
    def add_log(self, message: str, level: str = "info"):
        self._real_logger.add_log(message, level)
    
    def get_logs(self):
        return self._real_logger.get_logs()
    
    def set_callback(self, callback: Optional[Callable]):
        self._real_logger.set_callback(callback)
    
    @contextmanager
    def capture_logs(self, real_time: bool = False):
        """Context manager to capture stdout and convert to logs
        
        Args:
            real_time: If True, uses real-time stdout capture. If False, uses buffered capture.
        """
        if real_time:
            # Real-time capture - redirects stdout immediately
            old_stdout = sys.stdout
            sys.stdout = RealTimeStdout(self._real_logger, sys.__stdout__)
            
            try:
                yield self
            finally:
                # Flush any remaining content
                sys.stdout.flush()
                # Restore stdout
                sys.stdout = old_stdout
        else:
            # Buffered capture (original behavior)
            captured_output = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                yield self
            finally:
                sys.stdout = old_stdout
                output = captured_output.getvalue()
                
                for line in output.split('\n'):
                    if line.strip():
                        level = self._real_logger.get_level(line)
                        self._real_logger.add_log(line, level)

# Global logger instance
_logger = StreamlitLogger()

def get_logger():
    """Get the global logger instance"""
    return _logger


